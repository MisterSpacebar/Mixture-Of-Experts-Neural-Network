"""
Inspect & evaluate the saved WaterQualityNet model.
Shows architecture, parameter count, and human-readable predictions
vs actual Turbidity (FNU) on a sample of real sensor data.

Run from anywhere:
    python neural_network/water/output/inspect_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neural_network import WaterQualityNet          # noqa: E402
from data_loader import (                           # noqa: E402
    load_all_stations, FEATURE_COLS, SENSOR_FEATURES, TARGET
)

PT_FILE = os.path.join(os.path.dirname(__file__), "water_quality_net.pt")

if not os.path.exists(PT_FILE):
    print(f"ERROR: {PT_FILE} not found. Run neural_network.py first.")
    sys.exit(1)

# -------------------------------------------------------------------
# 1. Architecture & parameter count
# -------------------------------------------------------------------
INPUT_DIM = len(FEATURE_COLS)   # 8

state = torch.load(PT_FILE, weights_only=True)

print("=" * 65)
print("STATE DICT  (layer → shape → params)")
print("=" * 65)
total = 0
for name, tensor in state.items():
    n = tensor.numel()
    total += n
    print(f"  {name:<45s}  {str(tuple(tensor.shape)):<18s}  {n:>8,}")
print("-" * 65)
print(f"  {'TOTAL':<45s}  {'':18s}  {total:>8,}")

model = WaterQualityNet(input_dim=INPUT_DIM)
model.load_state_dict(state)
model.eval()

print()
print("=" * 65)
print("ARCHITECTURE")
print("=" * 65)
print(model)

# -------------------------------------------------------------------
# 2. Load real data, refit normalisation stats, run predictions
# -------------------------------------------------------------------
print()
print("=" * 65)
print("LOADING REAL SENSOR DATA …")
print("=" * 65)

df = load_all_stations()
for col in SENSOR_FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
df = df.dropna(subset=FEATURE_COLS + [TARGET])
df = df[df[TARGET] >= 0]
df = df[df["Depth (m)"] > 0]

# Refit the same normalisation used during training
X_raw = df[FEATURE_COLS].values.astype("float32")
y_raw = df[[TARGET]].values.astype("float32")

X_mean = X_raw.mean(axis=0)
X_std  = X_raw.std(axis=0) + 1e-8
X_norm = (X_raw - X_mean) / X_std

y_log  = np.log1p(y_raw)
y_mean = y_log.mean()
y_std  = y_log.std() + 1e-8

# -------------------------------------------------------------------
# 3. Sample 20 rows spread across all stations
# -------------------------------------------------------------------
sample = (
    df.groupby("station_id", group_keys=False)
    .apply(lambda g: g.sample(min(3, len(g)), random_state=42))
    .reset_index(drop=True)
    .head(20)
)
# Ensure station_id survived the groupby
if "station_id" not in sample.columns:
    sample = df.sample(20, random_state=42).reset_index(drop=True)

X_s = (sample[FEATURE_COLS].values.astype("float32") - X_mean) / X_std
X_t = torch.from_numpy(X_s)

with torch.no_grad():
    y_pred_norm = model(X_t).numpy().flatten()

# Back-transform: un-standardise → expm1 (undo log1p)
y_pred_fnu = np.expm1(y_pred_norm * y_std + y_mean)
y_true_fnu = sample[TARGET].values

mae  = np.mean(np.abs(y_pred_fnu - y_true_fnu))
rmse = np.sqrt(np.mean((y_pred_fnu - y_true_fnu) ** 2))

print()
print("=" * 65)
print("SAMPLE PREDICTIONS  (Turbidity in FNU)")
print("=" * 65)
print(f"  {'Station':<12} {'Actual FNU':>12} {'Predicted FNU':>14} {'Error':>10}")
print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*10}")
for i, row in sample.iterrows():
    sid   = f"L{int(row['station_id'])}"
    act   = y_true_fnu[i]
    pred  = y_pred_fnu[i]
    err   = pred - act
    print(f"  {sid:<12} {act:>12.2f} {pred:>14.2f} {err:>+10.2f}")

print()
print(f"  MAE  (mean absolute error) : {mae:.2f} FNU")
print(f"  RMSE (root mean sq error)  : {rmse:.2f} FNU")

# -------------------------------------------------------------------
# 4. Turbidity scale reference
# -------------------------------------------------------------------
print()
print("=" * 65)
print("TURBIDITY (FNU) REFERENCE SCALE")
print("=" * 65)
print("  < 1      Crystal clear (drinking water standard)")
print("  1 – 10   Very clear natural water")
print("  10 – 50  Slightly turbid (coastal/estuarine)")
print("  50 – 300 Moderately turbid (after rain / runoff)")
print("  > 300    Highly turbid (storm surge / sediment event)")
