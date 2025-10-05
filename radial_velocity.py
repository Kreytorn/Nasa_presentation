# rv_infer_standalone.py
# Standalone inference helpers for your RV model.
# - predict_from_arrays(model_path, time, rv, rv_err=None, threshold=0.5)
# - predict_from_files(model_path, file_paths, threshold=0.5)
#
# File parsing supports: IPAC .tbl (Astropy), CSV, or whitespace-separated text.

import sys, subprocess, io, re, warnings
from pathlib import Path

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

warnings.filterwarnings("ignore")

# Deps (install if missing)
try:
    import numpy as np
    import pandas as pd
except Exception:
    _pip_install(["numpy", "pandas"])
    import numpy as np
    import pandas as pd

try:
    import lightgbm as lgb
except Exception:
    _pip_install(["lightgbm"])
    import lightgbm as lgb

try:
    from astropy.io import ascii as astro_ascii
    from astropy.timeseries import LombScargle
except Exception:
    _pip_install(["astropy"])
    from astropy.io import ascii as astro_ascii
    from astropy.timeseries import LombScargle


# ==== config: must match training features ====
FEATURE_COLS = [
    "period","power1","power2","power3","amp",
    "rms","mad","skew","n_obs","span_days"
]


# ---------- core utils ----------
def load_model(model_path: str | Path) -> lgb.Booster:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return lgb.Booster(model_file=str(model_path))

def _features_from_series(time: np.ndarray, rv: np.ndarray, rv_err: np.ndarray | None = None):
    """Build the same features as training from a single star's RV curve."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(rv, dtype=float)
    if len(t) < 3 or np.std(y) == 0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        return None

    # normalize like training
    y = (y - np.median(y)) / (np.std(y) + 1e-9)

    # Lomb–Scargle periodogram (unweighted to mirror training)
    try:
        freq, power = LombScargle(t, y).autopower()
    except Exception:
        return None
    if len(freq) == 0:
        return None

    idx = np.argsort(power)[-3:]
    topf = freq[idx]; topp = power[idx]
    bestf = topf[-1]
    period = 1.0 / bestf if bestf > 0 else np.nan

    # sinusoid amplitude at bestf (least squares)
    phi = 2 * np.pi * bestf * t
    A = np.vstack([np.sin(phi), np.cos(phi), np.ones_like(phi)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = float(np.sqrt(coef[0]**2 + coef[1]**2))

    feats = {
        "period": float(period),
        "power1": float(topp[-1]),
        "power2": float(topp[-2]) if len(topp) > 1 else 0.0,
        "power3": float(topp[-3]) if len(topp) > 2 else 0.0,
        "amp":    amp,
        "rms":    float(np.std(y)),
        "mad":    float(np.median(np.abs(y - np.median(y)))),
        "skew":   float(pd.Series(y).skew()),
        "n_obs":  int(len(y)),
        "span_days": float(np.max(t) - np.min(t)),
    }
    return feats

def _map_cols(df: pd.DataFrame):
    """Find time/rv/rv_err column names in arbitrary tables."""
    cl = {c.lower().strip(): c for c in df.columns}
    def pick(cands):
        for k in cands:
            if k in cl: return cl[k]
        for k in cl:
            if any(sub in k for sub in cands): return cl[k]
        return None
    tcol = pick(['bjd_tdb','bjd','time','jd','jd_utc','mjd','date'])
    rcol = pick(['rv','radial_velocity','vrad','mnvel','vel','velocity','v_r'])
    ecol = pick(['rv_err','sigma_rv','e_rv','erv','rv_error','sig_rv','stdev','unc_rv'])
    return tcol, rcol, ecol

def _read_one_file(path: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Read one RV file: IPAC .tbl preferred, else CSV, else whitespace.
    Returns (df, star_id_guess). df has columns [time, rv, rv_err].
    """
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    data = None
    name = fp.name.lower()

    # IPAC first for .tbl
    if name.endswith(".tbl"):
        try:
            tab = astro_ascii.read(str(fp), format="ipac", guess=True, fast_reader=False)
            data = tab.to_pandas()
        except Exception:
            data = None

    # CSV fallback
    if data is None:
        try:
            data = pd.read_csv(fp)
        except Exception:
            # whitespace fallback
            data = pd.read_csv(fp, delim_whitespace=True, comment="#")

    if data is None or data.empty:
        raise ValueError(f"Could not parse any rows from: {fp}")

    tcol, rcol, ecol = _map_cols(data)
    if tcol is None or rcol is None:
        raise ValueError(f"Could not find time/rv columns in: {fp}")

    df = pd.DataFrame({
        "time": pd.to_numeric(data[tcol], errors="coerce"),
        "rv":   pd.to_numeric(data[rcol], errors="coerce"),
    })
    if ecol and ecol in data.columns:
        df["rv_err"] = pd.to_numeric(data[ecol], errors="coerce")
    else:
        df["rv_err"] = np.nan

    df = df.dropna(subset=["time","rv"]).sort_values("time")
    if len(df) < 3:
        raise ValueError(f"Need at least 3 valid (time, rv) points in: {fp}")

    # star_id guess from filename like UID_xxx_RVC_###.tbl
    m = re.match(r'^(.*)_RVC_', fp.stem, flags=re.IGNORECASE)
    star_id = m.group(1) if m else fp.stem
    return df, star_id


# ---------- public API ----------
def predict_from_arrays(model_path: str | Path,
                        time, rv, rv_err=None,
                        threshold: float = 0.5) -> dict:
    """
    Predict from raw arrays (one star).
    Returns dict with prob, label, features, n_obs, span_days.
    """
    booster = load_model(model_path)
    time = np.asarray(time, dtype=float)
    rv   = np.asarray(rv,   dtype=float)
    if rv_err is None:
        rv_err = np.full_like(rv, np.nan, dtype=float)
    else:
        rv_err = np.asarray(rv_err, dtype=float)

    feats = _features_from_series(time, rv, rv_err)
    if feats is None:
        return {"ok": False, "msg": "Not enough signal/points to compute features (need ≥3 and non-zero std)."}

    X = pd.DataFrame([feats])[FEATURE_COLS]
    prob = float(booster.predict(X)[0])
    pred = int(prob >= float(threshold))
    return {
        "ok": True,
        "probability": prob,
        "pred_label": pred,   # 1=planet, 0=not
        "threshold": float(threshold),
        "features": feats,
        "n_obs": int(feats["n_obs"]),
        "span_days": float(feats["span_days"]),
    }

def predict_from_files(model_path: str | Path,
                       file_paths: list[str | Path],
                       threshold: float = 0.5) -> pd.DataFrame:
    """
    Predict for one or many files. Returns a DataFrame with:
    [file, star_id, prob, pred_label, n_obs, span_days, error]
    """
    booster = load_model(model_path)
    rows = []
    for path in file_paths:
        rec = {"file": str(path), "star_id": None,
               "prob": np.nan, "pred_label": np.nan,
               "n_obs": np.nan, "span_days": np.nan,
               "error": ""}
        try:
            df, sid = _read_one_file(path)
            rec["star_id"] = sid
            feats = _features_from_series(df["time"].values, df["rv"].values, df["rv_err"].values)
            if feats is None:
                rec["error"] = "insufficient data / zero variance"
            else:
                X = pd.DataFrame([feats])[FEATURE_COLS]
                prob = float(booster.predict(X)[0])
                rec["prob"] = prob
                rec["pred_label"] = int(prob >= float(threshold))
                rec["n_obs"] = int(feats["n_obs"])
                rec["span_days"] = float(feats["span_days"])
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out