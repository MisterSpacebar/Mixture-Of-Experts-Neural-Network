"""
Inspect & evaluate the saved RiverLevelNet model.
Shows architecture, parameter count, and human-readable predictions
vs actual gage height (ft NAVD88) on a sample of real canal readings.

Run from anywhere:
    python neural_network/river/output/inspect_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neural_network import RiverLevelNet            # noqa: E402
from data_loader import load_all_months, FEATURE_COLS, TARGET  # noqa: E402

PT_FILE = os.path.join(os.path.dirname(__file__), "river_level_net.pt")

if not os.path.exists(PT_FILE):
    print(f"ERROR: {PT_FILE} not found. Run neural_network.py first.")
    sys.exit(1)

# -------------------------------------------------------------------
# 1. Architecture & parameter count
# -------------------------------------------------------------------
INPUT_DIM = len(FEATURE_COLS)   # 10

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

model = RiverLevelNet(input_dim=INPUT_DIM)
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
print("LOADING REAL RIVER DATA …")
print("=" * 65)

df = load_all_months()
for col in FEATURE_COLS + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=FEATURE_COLS + [TARGET])

# Refit same normalisation used during training
X_raw = df[FEATURE_COLS].values.astype("float32")
y_raw = df[[TARGET]].values.astype("float32")

X_mean = X_raw.mean(axis=0)
X_std  = X_raw.std(axis=0) + 1e-8

y_mean = y_raw.mean()
y_std  = y_raw.std() + 1e-8

# -------------------------------------------------------------------
# 3. Sample 20 rows spread across the 3 months
# -------------------------------------------------------------------
sample = (
    df.groupby("month", group_keys=False)
    .apply(lambda g: g.sample(min(7, len(g)), random_state=42))
    .reset_index(drop=True)
    .head(20)
)
# Ensure month survived the groupby
if "month" not in sample.columns:
    sample = df.sample(20, random_state=42).reset_index(drop=True)

X_s = (sample[FEATURE_COLS].values.astype("float32") - X_mean) / X_std
X_t = torch.from_numpy(X_s)

with torch.no_grad():
    y_pred_norm = model(X_t).numpy().flatten()

# Back-transform: un-standardise → actual feet
y_pred_ft = y_pred_norm * y_std + y_mean
y_true_ft = sample[TARGET].values

mae  = np.mean(np.abs(y_pred_ft - y_true_ft))
rmse = np.sqrt(np.mean((y_pred_ft - y_true_ft) ** 2))

print()
print("=" * 65)
print("SAMPLE PREDICTIONS  (Gage Height in ft, NAVD88)")
print("=" * 65)
print(f"  {'Date/Time':<22} {'Month':>6} {'Actual ft':>10} {'Predicted ft':>13} {'Error':>8}")
print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*13} {'-'*8}")
for i, row in sample.iterrows():
    dt   = str(row["datetime"])[:16]
    mo   = int(row["month"])
    act  = y_true_ft[i]
    pred = y_pred_ft[i]
    err  = pred - act
    print(f"  {dt:<22} {mo:>6} {act:>10.3f} {pred:>13.3f} {err:>+8.3f}")

print()
print(f"  MAE  (mean absolute error) : {mae:.4f} ft  ({mae * 30.48:.2f} cm)")
print(f"  RMSE (root mean sq error)  : {rmse:.4f} ft  ({rmse * 30.48:.2f} cm)")

# -------------------------------------------------------------------
# 4. Gage height context
# -------------------------------------------------------------------
print()
print("=" * 65)
print("GAGE HEIGHT (ft NAVD88) CONTEXT — USGS site 2286328")
print("=" * 65)
obs_min = float(y_raw.min())
obs_max = float(y_raw.max())
obs_mean = float(y_raw.mean())
print(f"  Observed range in dataset : {obs_min:.3f} ft  →  {obs_max:.3f} ft")
print(f"  Observed mean             : {obs_mean:.3f} ft")
print()
print("  Flood stage reference (typical South FL canals):")
print("    Normal operation  :  -0.5 to  1.0 ft")
print("    Elevated / caution:   1.0 to  2.5 ft")
print("    Flood concern     :   > 2.5 ft")
