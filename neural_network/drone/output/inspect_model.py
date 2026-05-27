"""
inspect_model.py — Human-readable inspection of drone_quality_net.pt

Run from any directory:
    python neural_network/drone/output/inspect_model.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neural_network import DroneQualityNet          # noqa: E402
from neural_network.drone.data_loader   import build_dataset, FEATURE_COLS, MISSION_LABELS, TARGET  # noqa: E402

PT_FILE = os.path.join(os.path.dirname(__file__), "drone_quality_net.pt")

if not os.path.exists(PT_FILE):
    print(f"ERROR: {PT_FILE} not found. Run neural_network.py first.")
    sys.exit(1)

# ── 1. State dict ─────────────────────────────────────────────────────────────
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

# ── 2. Architecture ────────────────────────────────────────────────────────────
INPUT_DIM = state["layers.0.0.weight"].shape[1]
model = DroneQualityNet(input_dim=INPUT_DIM)
model.load_state_dict(state)
model.eval()

print()
print("=" * 65)
print("ARCHITECTURE")
print("=" * 65)
print(model)

# ── 3. Load real data ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("LOADING REAL DRONE DATA …")
print("=" * 65)

X_train, X_val, y_train, y_val, X_mean, X_std, df = build_dataset()

# Sample ~7 rows per mission (up to 20 total)
sample = (
    df.groupby("mission_id", group_keys=False)
    .apply(lambda g: g.sample(min(7, len(g)), random_state=42))
    .reset_index(drop=True)
    .head(20)
)
if "mission_id" not in sample.columns:
    sample = df.sample(20, random_state=42).reset_index(drop=True)

X_s = (sample[FEATURE_COLS].values.astype("float32") - X_mean) / X_std
with torch.no_grad():
    preds_log = model(torch.from_numpy(X_s)).squeeze(1).numpy()

actual_fnu = sample[TARGET].values
pred_fnu   = np.expm1(preds_log)

# ── 4. Prediction table ────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SAMPLE PREDICTIONS  (Turbidity in FNU)")
print("=" * 65)

col_w = 12
hdr   = f"  {'Mission':<12}  {'Depth m':>8}  {'Actual FNU':>12}  {'Predicted FNU':>14}  {'Error':>10}"
sep   = "  " + "-"*12 + "  " + "-"*8 + "  " + "-"*12 + "  " + "-"*14 + "  " + "-"*10
print(hdr)
print(sep)

for _, row in sample.iterrows():
    i      = int(_)
    mid    = int(row["mission_id"])
    label  = MISSION_LABELS.get(mid, str(mid))
    depth  = row["Depth m"]
    act    = actual_fnu[i]
    pred   = pred_fnu[i]
    err    = pred - act
    sign   = "+" if err >= 0 else ""
    print(f"  {label:<12}  {depth:>8.2f}  {act:>12.2f}  {pred:>14.2f}  {sign}{err:>9.2f}")

mae  = float(np.mean(np.abs(pred_fnu - actual_fnu)))
rmse = float(np.sqrt(np.mean((pred_fnu - actual_fnu) ** 2)))
print()
print(f"  MAE  (mean absolute error) : {mae:.2f} FNU")
print(f"  RMSE (root mean sq error)  : {rmse:.2f} FNU")

# ── 5. Mission summary ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MISSION SUMMARY")
print("=" * 65)
for mid, label in MISSION_LABELS.items():
    rows = df[df["mission_id"] == mid]
    if len(rows) == 0:
        continue
    lo  = rows[TARGET].min()
    hi  = rows[TARGET].max()
    med = rows[TARGET].median()
    print(f"  {label}  |  {len(rows):>4} rows  |  "
          f"range {lo:.1f}–{hi:.1f} FNU  |  median {med:.1f} FNU")

# ── 6. Turbidity reference ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("TURBIDITY (FNU) REFERENCE SCALE")
print("=" * 65)
for band, desc in [
    ("< 1",      "Crystal clear (drinking water standard)"),
    ("1 – 10",   "Very clear natural water"),
    ("10 – 50",  "Slightly turbid (coastal / estuarine)"),
    ("50 – 300", "Moderately turbid (after rain / runoff)"),
    ("> 300",    "Highly turbid (storm surge / sediment event)"),
]:
    print(f"  {band:<10}  {desc}")
