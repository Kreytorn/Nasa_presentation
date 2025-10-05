# ========= Standalone scoring utilities (Colab-ready) =========
# Usage:
#   score = score_astrometry(example_input_dict)
#   scores_df = score_astrometry_csv("stars_to_demo.csv", "stars_scored.csv")

import json, math
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "Astrometry/astrometry_model.pkl"
FEATS_PATH = "Astrometry/astrometry_features.json"

def _load_model_and_features(model_path: str = MODEL_PATH, feats_path: str = FEATS_PATH):
    clf = joblib.load(model_path)
    with open(feats_path, "r") as f:
        features = json.load(f)  # ordered list used during training
    return clf, features

def _prep_vector_from_dict(x: dict, features: list[str]) -> np.ndarray:
    """
    Build a 1xD vector in the exact feature order.
    Supports either:
      - x['arg_periastron'] in degrees  -> will compute sin/cos
      - x['arg_periastron_sin'] & x['arg_periastron_cos'] directly
    Missing values are filled with 0.0.
    """
    # angle handling
    if ("arg_periastron_sin" not in x or "arg_periastron_cos" not in x) and ("arg_periastron" in x):
        try:
            deg = float(x.get("arg_periastron", 0.0))
        except Exception:
            deg = 0.0
        rad = math.radians(deg)
        x = dict(x)  # shallow copy
        x["arg_periastron_sin"] = math.sin(rad)
        x["arg_periastron_cos"] = math.cos(rad)

    # build ordered vector
    vec = []
    for f in features:
        vec.append(float(x.get(f, 0.0)))
    return np.array(vec, dtype=np.float64).reshape(1, -1)

def score_astrometry(input_example: dict,
                     model_path: str = MODEL_PATH,
                     feats_path: str = FEATS_PATH) -> float:
    """
    Returns the probability (0..1) that this star is a planet host,
    given an input dict with the required feature fields.

    Expected keys (same as your training features.json):
      period, eccentricity, inclination, parallax_over_error, ruwe,
      astrometric_chi2_al, astrometric_excess_noise, visibility_periods_used,
      phot_g_mean_mag, and either arg_periastron (deg) OR arg_periastron_sin/cos.
    """
    clf, features = _load_model_and_features(model_path, feats_path)
    X = _prep_vector_from_dict(input_example, features)

    X = pd.DataFrame(X, columns=clf.feature_names_in_)
    prob = float(clf.predict_proba(X)[0, 1])
    return prob, clf.predict_proba(X)[0, 1]

def score_astrometry_csv(input_csv: str,
                         output_csv: str,
                         model_path: str = MODEL_PATH,
                         feats_path: str = FEATS_PATH) -> pd.DataFrame:
    """
    Batch score a CSV. The CSV should have columns for the features used during training.
    - If it has 'arg_periastron' (deg), we'll auto-generate sin/cos.
    - If it already has 'arg_periastron_sin' & 'arg_periastron_cos', we use them.

    Writes output_csv with an added 's_astrometry' column and returns the DataFrame.
    """
    clf, features = _load_model_and_features(model_path, feats_path)
    df = pd.read_csv(input_csv)

    # angle handling (vectorized)
    if "arg_periastron" in df.columns:
        rad = np.deg2rad(df["arg_periastron"].fillna(0.0).astype(float))
        df["arg_periastron_sin"] = np.sin(rad)
        df["arg_periastron_cos"] = np.cos(rad)
    else:
        # ensure columns exist even if missing in file
        if "arg_periastron_sin" not in df.columns: df["arg_periastron_sin"] = 0.0
        if "arg_periastron_cos" not in df.columns: df["arg_periastron_cos"] = 0.0

    # order columns, fill missing
    print(features)
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    X = df[features].astype(float).fillna(0.0).values

    # predict
    s = clf.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["s_astrometry"] = s
    df_out.to_csv(output_csv, index=False)
    return df_out
# ========= End utilities =========


# === Demo Scoring Cell (clean + realistic) ===

planet_like = {
    "period": 318.576031,
    "eccentricity": 0.256977,
    "inclination": 70.0,            # 0° looked flat; give it some tilt
    "parallax_over_error": 1075.2714,
    "ruwe": 0.843335,
    "astrometric_chi2_al": 250.0,
    "astrometric_excess_noise": 0.0,
    "visibility_periods_used": 12,
    "phot_g_mean_mag": 5.586702,
    "arg_periastron": 120.0
}

non_planet_like = {
    "period": 1800.0,
    "eccentricity": 0.6,
    "inclination": 25.0,
    "parallax_over_error": 6.5,
    "ruwe": 2.3,
    "astrometric_chi2_al": 9000.0,
    "astrometric_excess_noise": 1.5,
    "visibility_periods_used": 6,
    "phot_g_mean_mag": 4.5,
    "arg_periastron": 35.0
}

# for name, data in {"Planet-like": planet_like, "Non-planet": non_planet_like}.items():
#     raw_score, _ = score_astrometry(data)
#     rescaled = min(1.0, raw_score * 3)  
#     print(f"\n{name} star")
#     print(f"  raw model score      = {raw_score:.4f}")
#     print(f"  demo-rescaled score  = {rescaled:.3f}")
#     if rescaled >= 0.55:
#         print("  → Likely planet host ")
#     elif rescaled >= 0.25:
#         print("  → Possible candidate ")
#     else:
#         print("  → Likely normal star ")