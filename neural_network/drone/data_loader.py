"""
data_loader.py — Submarine drone water-quality missions

Three missions:
  0  March 15  2024  (8 CSVs)   — full sensor suite
  1  October 25 2024 (1 CSV)    — full sensor suite
  2  March 18  2025  (20 CSVs)  — missing Chlorophyll / pH sensors

Uses only the 13 columns present in all three missions.
Target: Turbidity FNU  (log1p-transformed, always >= 0)
"""

import os
import glob

import numpy as np
import pandas as pd

BASE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "submarine_drone_data"
)

# Columns present in every mission after renaming
FEATURE_COLS = [
    "Depth m",
    "nLF Cond µS/cm",
    "ODO % sat",
    "ODO mg/L",
    "Pressure psi a",
    "Sal psu",
    "SpCond µS/cm",
    "TDS mg/L",
    "Temp °C",
    "Vertical Position m",
    "Latitude",
    "Longitude",
    "mission_id",
]
TARGET = "Turbidity FNU"

# March 2025 used different column names for some sensors
RENAME_2025 = {
    "Turb (FNU)":   "Turbidity FNU",
    "Temp (C)":     "Temp °C",
    "latitude":     "Latitude",
    "longitude":    "Longitude",
    # "ODO % local" is NOT renamed — Mar2025 already has "ODO % sat" so
    # renaming would create duplicate columns; ODO % local is simply dropped
    # when we select FEATURE_COLS below.
}

MISSION_LABELS = {
    0: "Mar-2024",
    1: "Oct-2024",
    2: "Mar-2025",
}


def _load_mission(folder, mission_id, rename=None):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if rename:
            df = df.rename(columns=rename)
        df["mission_id"] = float(mission_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_dataset():
    m24  = _load_mission(os.path.join(BASE, "March 15th 2024"),   mission_id=0)
    oct24 = _load_mission(os.path.join(BASE, "October 25th 2024"), mission_id=1)
    m25  = _load_mission(os.path.join(BASE, "March 18th 2025"),   mission_id=2, rename=RENAME_2025)

    needed = FEATURE_COLS + [TARGET]
    combined = pd.concat([m24, oct24, m25], ignore_index=True)
    combined = combined[needed].copy()
    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.dropna(subset=needed)
    # Turbidity can read slightly negative due to sensor noise in clear water;
    # clip to 0 rather than dropping those rows (log1p requires >= 0)
    combined[TARGET] = combined[TARGET].clip(lower=0)
    combined = combined.reset_index(drop=True)

    print(f"Rows after cleaning: {len(combined)}")
    for mid, label in MISSION_LABELS.items():
        n = (combined["mission_id"] == mid).sum()
        print(f"  Mission {mid} ({label}): {n} rows")

    X = combined[FEATURE_COLS].values.astype("float32")
    y = np.log1p(combined[TARGET].values.astype("float32"))

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8

    return (
        (X_train - X_mean) / X_std,
        (X_val   - X_mean) / X_std,
        y_train,
        y_val,
        X_mean,
        X_std,
        combined,   # raw frame kept for inspect_model
    )


if __name__ == "__main__":
    Xtr, Xv, ytr, yv, _, _, df = build_dataset()
    print(f"Train: {Xtr.shape}  Val: {Xv.shape}")
