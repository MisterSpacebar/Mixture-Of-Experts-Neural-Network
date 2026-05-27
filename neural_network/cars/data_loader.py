"""
EV Fleet Data Loader
---------------------
Reads all 226 vehicle daily-telemetry CSVs from data/gov_evs/vehicle/,
joins each row with static vehicle metadata from vehicle_reference_json.json,
and returns normalised PyTorch tensors ready for training.

Target  : Total Energy Consumption (kWh per day)
Features: 9 daily telemetry columns + 6 static vehicle attributes = 15 total
"""

import json
import os
import glob

import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.join(_HERE, "..", "..")
_VEHICLE_DIR = os.path.normpath(os.path.join(_REPO_ROOT, "data", "gov_evs", "vehicle"))
_REF_JSON    = os.path.normpath(os.path.join(_REPO_ROOT, "data", "gov_evs", "data_reference", "vehicle_reference_json.json"))

# ---------------------------------------------------------------------------
# Column selections
# ---------------------------------------------------------------------------
DAILY_FEATURES = [
    "Total Distance",
    "Driving Time",
    "Total Run Time",
    "SOC Used",
    "Average Ambient Temperature",
]

STATIC_FEATURES = [
    "Weight Class",
    "Model Year",
    "Rated Energy",
]

TARGET = "Total Energy Consumption"


def load_reference() -> pd.DataFrame:
    """Load vehicle_reference_json.json → DataFrame indexed by Vehicle ID."""
    with open(_REF_JSON, encoding="utf-8") as f:
        records = json.load(f)
    ref = pd.DataFrame(records)
    ref["Vehicle ID"] = ref["Vehicle ID"].str.upper().str.strip()
    ref = ref[["Vehicle ID"] + STATIC_FEATURES].copy()
    for col in STATIC_FEATURES:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    return ref.set_index("Vehicle ID")


def load_all_vehicles() -> pd.DataFrame:
    """Concatenate all vehicle CSVs into a single DataFrame."""
    csvs = sorted(
        f for f in glob.glob(os.path.join(_VEHICLE_DIR, "*.csv"))
        if not os.path.basename(f).startswith(".")
    )
    frames = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception:
            pass
    combined = pd.concat(frames, ignore_index=True)
    combined["Vehicle ID"] = combined["Vehicle ID"].str.upper().str.strip()
    return combined


def build_dataset(val_split: float = 0.2, seed: int = 42):
    """
    Returns
    -------
    train_dataset, val_dataset : TensorDataset
    feature_names              : list[str]
    n_features                 : int
    """
    ref     = load_reference()
    daily   = load_all_vehicles()

    # Join static attributes onto each daily row
    merged = daily.join(ref, on="Vehicle ID", how="left")

    # Cast all feature + target columns to numeric
    for col in DAILY_FEATURES + STATIC_FEATURES + [TARGET]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Drop rows missing the target or any feature
    feature_cols = DAILY_FEATURES + STATIC_FEATURES
    merged = merged.dropna(subset=feature_cols + [TARGET])

    # Drop rows with non-positive energy (invalid sensor readings)
    merged = merged[merged[TARGET] > 0]

    print(f"Clean rows after dropping NaN: {len(merged):,}")

    X_raw = merged[feature_cols].values.astype("float32")
    y_raw = merged[[TARGET]].values.astype("float32")

    # Standardise features (mean=0, std=1)
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - X_mean) / X_std

    # Log-transform target (energy is right-skewed) then standardise
    import numpy as np
    y_log  = np.log1p(y_raw)
    y_mean = y_log.mean()
    y_std  = y_log.std() + 1e-8
    y_norm = (y_log - y_mean) / y_std

    X_tensor = torch.from_numpy(X_norm)
    y_tensor = torch.from_numpy(y_norm)

    full_dataset = TensorDataset(X_tensor, y_tensor)

    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_ds, val_ds, feature_cols, X_tensor.shape[1]


if __name__ == "__main__":
    train_ds, val_ds, features, n_feat = build_dataset()
    print(f"Features ({n_feat}): {features}")
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
