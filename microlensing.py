import os, numpy as np, torch, torch.nn as nn

# ========= Conv1D-GRU (same arch you trained) =========
class Conv1DGRU(nn.Module):
    def __init__(self, in_ch=2, hid=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),    nn.ReLU(),
            nn.MaxPool1d(2),  # 512 -> 256
            nn.Conv1d(64, 64, kernel_size=3, padding=1),    nn.ReLU(),
        )
        self.gru  = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))
    def forward(self, x):        # x: (B, 2, 512)
        z = self.conv(x)         # (B, 64, 256)
        z = z.transpose(1, 2)    # (B, 256, 64)
        z, _ = self.gru(z)       # (B, 256, 128)
        z = z.mean(dim=1)        # (B, 128)
        return self.head(z).squeeze(1)

# -------- helpers (kept inside same cell for standalone use) --------
def _read_table(path):
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}")
    return num.to_numpy()

def _to_2ch_512(arr, length=512, z_normalize=False):
    """Accepts shapes (T,2), (2,T), (T,1), (T,), (512,2) and returns (2,512) float32."""
    a = np.asarray(arr)
    if a.ndim == 1:                      # (T,)
        a = np.stack([a, np.zeros_like(a)], axis=1)        # -> (T,2)
    elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T                                           # (2,T) -> (T,2)
    elif a.ndim == 2 and a.shape[1] == 1:                 # (T,1)
        a = np.concatenate([a, np.zeros_like(a)], axis=1) # -> (T,2)
    elif a.ndim != 2:
        raise ValueError(f"Unsupported array shape: {a.shape}")

    if a.shape[1] != 2:
        raise ValueError(f"Need 2 channels after prep, got shape {a.shape}")

    # resample along time to 512 if needed
    T = a.shape[0]
    if T != length:
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, length)
        ch0 = np.interp(x_new, x_old, a[:,0])
        ch1 = np.interp(x_new, x_old, a[:,1])
        a = np.stack([ch0, ch1], axis=1)  # (512,2)

    if z_normalize:
        for c in range(2):
            mu, sd = a[:,c].mean(), a[:,c].std()
            a[:,c] = (a[:,c] - mu) / (sd + 1e-6)

    return a.T.astype("float32")  # -> (2,512)

def _load_model(model_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt.get("config", {"in_ch":2, "hid":64})
    model = Conv1DGRU(in_ch=cfg.get("in_ch",2), hid=cfg.get("hid",64)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device

# ========= #1: single-event predictor =========
def predict_single_event(
    source,
model_path="Microlensing/roman_2018/processed/models/conv1d_gru.pt",
    z_normalize=False,
):
    """
    source:  (a) numpy array, or
             (b) path to .npy / .csv / .txt / .tsv (will grab first 1–2 numeric cols)
             Accepts 1 or 2 channels; resamples to length 512.
    Returns: float prob in [0,1]
    """
    # load data
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".npy":
            arr = np.load(source)
        elif ext in [".csv", ".txt", ".tsv"]:
            arr = _read_table(source)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        arr = source

    x = _to_2ch_512(arr, length=512, z_normalize=z_normalize)   # (2,512)
    x = torch.from_numpy(x).unsqueeze(0)                         # (1,2,512)

    # load model + infer
    model, device = _load_model(model_path)
    with torch.no_grad():
        prob = torch.sigmoid(model(x.to(device))).item()
    return float(prob)

# ========= #2: multi-event predictor =========
def predict_many_events(
    sources,
    model_path="Microlensing/roman_2018/processed/models/conv1d_gru.pt",
    z_normalize=False,
    return_dataframe=True
):
    """
    sources: list of numpy arrays and/or file paths.
    Returns: list of probs (and a pandas DataFrame if return_dataframe=True)
    """
    xs = []
    names = []
    for src in sources:
        try:
            if isinstance(src, str):
                nm  = os.path.basename(src)
                ext = os.path.splitext(src)[1].lower()
                if ext == ".npy":
                    arr = np.load(src)
                elif ext in [".csv", ".txt", ".tsv"]:
                    arr = _read_table(src)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            else:
                arr = src
                nm  = "array_" + str(len(xs))

            x = _to_2ch_512(arr, 512, z_normalize=z_normalize)   # (2,512)
            xs.append(x)
            names.append(nm)
        except Exception as e:
            # keep place with NaN if something fails
            xs.append(None); names.append(f"{nm if 'nm' in locals() else 'item'} (ERROR: {e})")

    # batch the valid ones
    valid_idx = [i for i,x in enumerate(xs) if x is not None]
    probs = [np.nan]*len(xs)
    if valid_idx:
        batch = torch.from_numpy(np.stack([xs[i] for i in valid_idx], axis=0))  # (B,2,512)
        model, device = _load_model(model_path)
        with torch.no_grad():
            p = torch.sigmoid(model(batch.to(device))).cpu().numpy().tolist()
        for k,i in enumerate(valid_idx):
            probs[i] = float(p[k])

    if return_dataframe:
        import pandas as pd
        df = pd.DataFrame({"source": names, "planet_prob": probs})
        return probs, df
    return probs


# # from .npy
# p = predict_single_event("Microlensing/roman_2018/processed/demo_inputs/event_223_1ch.npy")
# print("prob:", p)
# p = predict_single_event("Microlensing/roman_2018/processed/demo_inputs/event_223_1ch.csv", z_normalize=True)
# print("prob:", p)

# # from a numpy array (T,2) or (512,2)
# arr = np.random.randn(512,2).astype("float32")
# p = predict_single_event(arr)

# files = [
#     "Project/Microlensing/roman_2018/processed/demo_inputs/event_223_1ch.npy",
#     "Project/Microlensing/roman_2018/processed/demo_inputs/event_223_1ch.csv",
#     np.random.randn(400,2)  # array with T!=512 → auto-resampled
# ]
# probs, df = predict_many_events(files)
# print(df.head())