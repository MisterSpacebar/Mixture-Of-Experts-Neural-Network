"""
inspect_moe.py — Human-readable inspection of moe_net.pt

Run from any directory:
    python mixture_of_experts/output/inspect_moe.py

Key output:
  • State-dict summary
  • Full architecture
  • Expert utilisation heatmap (which expert handles each domain)
  • Per-domain MAE in normalised (z-score) space
  • Sample predictions per domain
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_network  import MoENet                                         # noqa
from unified_loader import (                                             # noqa
    build_combined_dataset, INPUT_DIM, N_EXPERTS, DOMAIN_NAMES,
)

PT_FILE = os.path.join(os.path.dirname(__file__), "moe_net.pt")

if not os.path.exists(PT_FILE):
    print(f"ERROR: {PT_FILE} not found. Run moe_network.py first.")
    sys.exit(1)

# ── 1. State dict ─────────────────────────────────────────────────────────────
state = torch.load(PT_FILE, weights_only=True)

print("=" * 70)
print("STATE DICT  (layer → shape → params)")
print("=" * 70)
total = 0
for name, tensor in state.items():
    n = tensor.numel()
    total += n
    print(f"  {name:<52s}  {str(tuple(tensor.shape)):<18s}  {n:>8,}")
print("-" * 70)
print(f"  {'TOTAL':<52s}  {'':18s}  {total:>8,}")

# ── 2. Architecture ────────────────────────────────────────────────────────────
model = MoENet(input_dim=INPUT_DIM, n_experts=N_EXPERTS)
model.load_state_dict(state)
model.eval()

print()
print("=" * 70)
print("ARCHITECTURE")
print("=" * 70)
print(model)

# ── 3. Load data ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("LOADING COMBINED DATASET …")
print("=" * 70)
X_train, X_val, y_train, y_val, d_train, d_val = build_combined_dataset()

X_val_t = torch.from_numpy(X_val)
with torch.no_grad():
    preds_t, gate_w_t = model(X_val_t)

preds_np  = preds_t.numpy()
gate_np   = gate_w_t.numpy()   # (N_val, N_EXPERTS)
errors_np = preds_np - y_val

# ── 4. Expert utilisation heatmap ─────────────────────────────────────────────
print()
print("=" * 70)
print("EXPERT UTILISATION BY DOMAIN")
print("(average gate weight a domain sends to each expert — rows sum to 1.0)")
print("=" * 70)

E = N_EXPERTS
expert_hdr = "".join(f"  Expert-{i}" for i in range(E))
print(f"  {'Domain':<20s}{expert_hdr}   Top Expert")
print("  " + "-" * (20 + E * 10 + 12))

top_expert_counts = np.zeros((len(DOMAIN_NAMES), E), dtype=int)

for did, name in DOMAIN_NAMES.items():
    mask = (d_val == did)
    if mask.sum() == 0:
        continue
    avg_w   = gate_np[mask].mean(axis=0)       # average weight per expert
    top_exp = int(gate_np[mask].argmax(axis=1).mean().round())   # modal top expert
    top_str = f"Expert-{top_exp}"
    weights = "".join(f"  {w:>8.3f}" for w in avg_w)
    print(f"  {name:<20s}{weights}   {top_str}")

# ── 5. Per-domain accuracy ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("PER-DOMAIN ACCURACY  (in normalised z-score space)")
print("=" * 70)
print(f"  {'Domain':<20s}  {'Val rows':>10}  {'MAE':>10}  {'RMSE':>10}")
print("  " + "-" * 56)
for did, name in DOMAIN_NAMES.items():
    mask = (d_val == did)
    n    = mask.sum()
    if n == 0:
        continue
    mae  = float(np.abs(errors_np[mask]).mean())
    rmse = float(np.sqrt((errors_np[mask] ** 2).mean()))
    print(f"  {name:<20s}  {n:>10,}  {mae:>10.4f}  {rmse:>10.4f}")

note = "(1.0 = 1 standard deviation in each domain's own target space)"
print(f"\n  Note: {note}")

# ── 6. Sample predictions per domain ──────────────────────────────────────────
print()
print("=" * 70)
print("SAMPLE PREDICTIONS  (5 rows per domain, normalised target space)")
print("=" * 70)
print(f"  {'Domain':<20s}  {'Actual':>10}  {'Predicted':>10}  {'Error':>10}  "
      f"{'Top Expert':>12}")
print("  " + "-" * 68)

rng = np.random.default_rng(7)
for did, name in DOMAIN_NAMES.items():
    mask  = np.where(d_val == did)[0]
    if len(mask) == 0:
        continue
    samp  = rng.choice(mask, min(5, len(mask)), replace=False)
    for i in samp:
        act   = float(y_val[i])
        pred  = float(preds_np[i])
        err   = pred - act
        top_e = int(gate_np[i].argmax())
        sign  = "+" if err >= 0 else ""
        print(f"  {name:<20s}  {act:>10.3f}  {pred:>10.3f}  {sign}{err:>9.3f}  "
              f"  Expert-{top_e}")
    print()

# ── 7. What to look for ────────────────────────────────────────────────────────
print("=" * 70)
print("HOW TO READ THE EXPERT UTILISATION TABLE")
print("=" * 70)
print("""  Ideal MoE: each domain is routed predominantly to ONE expert (0.6+)
  and different domains prefer DIFFERENT experts.

  Expert collapse: one expert gets >0.8 weight for ALL domains — the gate
  has not learned to specialise (add more training data or raise LB_WEIGHT).

  Balanced gate: all experts near 0.25 for every domain — the gate is not
  routing at all, acting as a simple ensemble average.
""")
