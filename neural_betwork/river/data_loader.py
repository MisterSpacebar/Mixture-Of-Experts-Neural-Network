"""
River / Canal Data Loader
--------------------------
Reads the 3 monthly USGS canal CSVs (Jul/Aug/Sep 2025) from
data/river_data/, concatenates them, and returns normalised
PyTorch tensors ready for training.

Target  : gage_height_ft_navd88  — water level (ft), driven by rainfall/tide
Features: 10 columns
    water_temp_top_c, water_temp_bottom_c,
    salinity_top_ppt, salinity_bottom_ppt,
    specific_conductance_top_us_cm_25c, specific_conductance_bottom_us_cm_25c,
    stream_water_level_elevation_ft_ngvd29,
    airport_temp_min_c, airport_temp_max_c, airport_rain_in
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
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_HERE, "..", "..")
_DATA_DIR  = os.path.normpath(os.path.join(_REPO_ROOT, "data", "river_data"))

# ---------------------------------------------------------------------------
# Column selections
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "water_temp_top_c",
    "water_temp_bottom_c",
    "salinity_top_ppt",
    "salinity_bottom_ppt",
    "specific_conductance_top_us_cm_25c",
    "specific_conductance_bottom_us_cm_25c",
    "stream_water_level_elevation_ft_ngvd29",
    "airport_temp_min_c",
    "airport_temp_max_c",
    "airport_rain_in",
]

TARGET = "gage_height_ft_navd88"


def load_all_months() -> pd.DataFrame:
    """Concatenate all monthly river CSVs into one DataFrame."""
    csvs = sorted(glob.glob(os.path.join(_DATA_DIR, "*.csv")))
    frames = []
    for path in csvs:
        df = pd.read_csv(path, low_memory=False)
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
    df = load_all_months()

    # Cast to numeric
    for col in FEATURE_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing target or any feature
    df = df.dropna(subset=FEATURE_COLS + [TARGET])

    print(f"Clean rows: {len(df):,}")

    X_raw = df[FEATURE_COLS].values.astype("float32")
    y_raw = df[[TARGET]].values.astype("float32")

    # Standardise features (mean=0, std=1)
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - X_mean) / X_std

    # Standardise target directly (gage height can be negative, so no log)
    y_mean = y_raw.mean()
    y_std  = y_raw.std() + 1e-8
    y_norm = (y_raw - y_mean) / y_std

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
