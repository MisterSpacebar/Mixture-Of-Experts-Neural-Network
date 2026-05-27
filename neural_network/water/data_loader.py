"""
Water Station Data Loader
--------------------------
Reads all 8 water-station CSVs from data/water_data/water_stations_raw/,
combines them with a station_id label, drops rows missing any core feature
or the target, then returns normalised PyTorch tensors.

Target  : Turbidity (FNU)  — water clarity; spikes after rainfall/runoff
Features: 8 columns
    Temperature (C), Specific Conductance (uS/cm), Salinity (PPT),
    Pressure (psia), Depth (m), ODO (%Sat), ODO (mg/L), station_id (0-7)
"""

import os
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.join(_HERE, "..", "..")
_STATION_DIR = os.path.normpath(os.path.join(_REPO_ROOT, "data", "water_data", "water_stations_raw"))

# ---------------------------------------------------------------------------
# Column selections
# ---------------------------------------------------------------------------
SENSOR_FEATURES = [
    "Temperature (C)",
    "Specific Conductance (uS/cm)",
    "Salinity (PPT)",
    "Pressure (psia)",
    "Depth (m)",
    "ODO (%Sat)",
    "ODO (mg/L)",
]
FEATURE_COLS = SENSOR_FEATURES + ["station_id"]  # 8 total

TARGET = "Turbidity (FNU)"


def load_all_stations() -> pd.DataFrame:
    """Concatenate all L0-L7 station CSVs, tagging each with a station_id."""
    csvs = sorted(glob.glob(os.path.join(_STATION_DIR, "*.csv")))
    frames = []
    for station_id, path in enumerate(csvs):
        df = pd.read_csv(path, low_memory=False)
        df["station_id"] = float(station_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_dataset(val_split: float = 0.2, seed: int = 42):
    """
    Returns
    -------
    train_dataset, val_dataset : TensorDataset
    feature_names              : list[str]
    n_features                 : int
    """
    df = load_all_stations()

    # Cast all relevant columns to numeric
    for col in SENSOR_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing target or any feature
    df = df.dropna(subset=FEATURE_COLS + [TARGET])

    # Drop physically invalid readings
    df = df[df[TARGET] >= 0]
    df = df[df["Depth (m)"] > 0]

    print(f"Clean rows: {len(df):,}")

    X_raw = df[FEATURE_COLS].values.astype("float32")
    y_raw = df[[TARGET]].values.astype("float32")

    # Standardise features
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - X_mean) / X_std

    # Log-transform target (turbidity is heavily right-skewed), then standardise
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
        generator=torch.Generator().manual_seed(seed),
    )

    return train_ds, val_ds, FEATURE_COLS, X_tensor.shape[1]


if __name__ == "__main__":
    train_ds, val_ds, features, n_feat = build_dataset()
    print(f"Features ({n_feat}): {features}")
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
