"""
compare_models.py — Side-by-side comparison of standalone domain models vs MoE.

Run from workspace root:
    python comparison/compare_models.py

For every domain it:
  • evaluates the standalone model on the domain's own validation split
  • evaluates the MoE model on the same validation data (features padded to 14-dim)
  • reports MAE and RMSE in normalised z-score units (consistent across domains)
  • shows the MoE's expert-routing table so you can see which experts handle what

Normalised z-score space means:
    0.0  = perfect prediction
    0.5  = moderate error (half a standard deviation in the target)
    1.0  = poor (error equal to one full standard deviation)
"""

import importlib.util
import os
import sys

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.join(_HERE, "..")
ROOT_NN = os.path.join(ROOT, "neural_network")
ROOT_MoE = os.path.join(ROOT, "mixture_of_experts")

sys.path.insert(0, ROOT_MoE)
from unified_loader import _pad_and_tag, INPUT_DIM, N_EXPERTS, MAX_FEATURES  # noqa
from moe_network     import MoENet                                            # noqa

# ── Domain configuration ──────────────────────────────────────────────────────
DOMAINS = [
    # (folder, domain_id, ModelClass name, .pt filename,   target description)
    ("cars",  0, "EVFleetNet",      "ev_fleet_net.pt",       "z-score(log₁ₚ kWh)"),
    ("water", 1, "WaterQualityNet", "water_quality_net.pt",  "z-score(log₁ₚ FNU)"),
    ("river", 2, "RiverLevelNet",   "river_level_net.pt",    "z-score(ft NAVD88)"),
    ("drone", 3, "DroneQualityNet", "drone_quality_net.pt",  "z-score(log₁ₚ FNU)"),
]

DOMAIN_LABELS = {0: "Cars (EV Fleet)", 1: "Water Stations",
                 2: "River Canal",     3: "Drone Missions"}


# ── Import helpers ────────────────────────────────────────────────────────────
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subset_to_numpy(subset):
    X = subset.dataset.tensors[0][subset.indices].numpy()
    y = subset.dataset.tensors[1][subset.indices].numpy().ravel()
    return X, y


def _batched_inference_standalone(model, X_np, batch=4096):
    """Run a standalone model (returns a plain tensor) in batches."""
    preds = []
    for i in range(0, len(X_np), batch):
        chunk = torch.from_numpy(X_np[i:i+batch])
        with torch.no_grad():
            preds.append(model(chunk).squeeze(-1).numpy())
    return np.concatenate(preds)


def _batched_inference_moe(model, X_np, batch=4096):
    """Run the MoE model (returns pred + gate_w) in batches."""
    preds, gates = [], []
    for i in range(0, len(X_np), batch):
        chunk = torch.from_numpy(X_np[i:i+batch])
        with torch.no_grad():
            p, g = model(chunk)
            preds.append(p.numpy())
            gates.append(g.numpy())
    return np.concatenate(preds), np.vstack(gates)


# ── Load MoE ──────────────────────────────────────────────────────────────────
MoE_PT = os.path.join(ROOT_MoE, "output", "moe_net.pt")
if not os.path.exists(MoE_PT):
    print(f"ERROR: {MoE_PT} not found. Run mixture_of_experts/moe_network.py first.")
    sys.exit(1)

moe_state = torch.load(MoE_PT, weights_only=True)
moe_model  = MoENet(input_dim=INPUT_DIM, n_experts=N_EXPERTS)
moe_model.load_state_dict(moe_state)
moe_model.eval()

# ── Per-domain evaluation ─────────────────────────────────────────────────────
results = {}   # domain_id → dict with metrics + gate weights

for folder, did, cls_name, pt_name, target_desc in DOMAINS:
    print(f"Evaluating {DOMAIN_LABELS[did]} …", end=" ", flush=True)

    # -- load data
    dl_mod = _load_module(os.path.join(ROOT_NN, folder, "data_loader.py"),
                          f"dl_{folder}")
    nn_mod = _load_module(os.path.join(ROOT_NN, folder, "neural_network.py"),
                          f"nn_{folder}")

    result = dl_mod.build_dataset()

    if folder == "drone":
        X_val, y_val_raw = result[1], result[3]   # log1p(turbidity)
        y_train          = result[2]               # log1p(turbidity) train split
        ym = float(y_train.mean())
        ys = float(y_train.std()) + 1e-8
        y_val_zs = (y_val_raw - ym) / ys          # z-score for MoE comparison
    else:
        X_val, y_val_zs = _subset_to_numpy(result[1])   # already z-scored

    # -- standalone model
    pt_path   = os.path.join(ROOT_NN, folder, "output", pt_name)
    sa_state  = torch.load(pt_path, weights_only=True)
    input_dim = sa_state["layers.0.0.weight"].shape[1]
    sa_model  = getattr(nn_mod, cls_name)(input_dim=input_dim, output_dim=1)
    sa_model.load_state_dict(sa_state)
    sa_model.eval()

    if folder == "drone":
        # standalone predicts log1p; z-score to same space as MoE
        sa_preds_raw = _batched_inference_standalone(sa_model, X_val)
        sa_preds_zs  = (sa_preds_raw - ym) / ys
    else:
        sa_preds_zs = _batched_inference_standalone(sa_model, X_val)

    # -- MoE on same val data (padded to 14 features + domain_id)
    X_moe          = _pad_and_tag(X_val, did)
    moe_preds, gw  = _batched_inference_moe(moe_model, X_moe.astype("float32"))

    # -- metrics
    def _mae_rmse(preds, actual):
        e = preds - actual
        return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))

    sa_mae,  sa_rmse  = _mae_rmse(sa_preds_zs, y_val_zs)
    moe_mae, moe_rmse = _mae_rmse(moe_preds,   y_val_zs)

    results[did] = {
        "label":       DOMAIN_LABELS[did],
        "target_desc": target_desc,
        "n_val":       len(y_val_zs),
        "sa_mae":      sa_mae,  "sa_rmse":  sa_rmse,
        "moe_mae":     moe_mae, "moe_rmse": moe_rmse,
        "delta_mae":   moe_mae - sa_mae,
        "gate_weights": gw.mean(axis=0),   # avg gate weight per expert
    }
    print(f"done  ({len(y_val_zs):,} val rows)")


# ── Print comparison table ────────────────────────────────────────────────────
W = 72
print()
print("=" * W)
print("MODEL COMPARISON: Standalone Domain Network vs Mixture-of-Experts")
print("Metrics in normalised z-score space  |  0.0=perfect  0.5=moderate  1.0=poor")
print("=" * W)

hdr = (f"  {'Domain':<22s}  {'Standalone':>22s}  {'MoE':>22s}  {'Δ MAE':>8s}")
sub = (f"  {'':22s}  {'MAE':>10s}  {'RMSE':>10s}  {'MAE':>10s}  {'RMSE':>10s}  {'':>8s}")
print(hdr)
print(sub)
print("  " + "-" * (W - 2))

totals = {"sa_mae": 0, "sa_rmse": 0, "moe_mae": 0, "moe_rmse": 0, "n": 0}

for did in sorted(results):
    r = results[did]
    d = r["delta_mae"]
    flag = "  ✓ tied" if abs(d) < 0.005 else (f"  ↑ MoE +{d:.3f}" if d > 0 else f"  ↓ MoE {d:.3f}")
    print(f"  {r['label']:<22s}  {r['sa_mae']:>10.4f}  {r['sa_rmse']:>10.4f}"
          f"  {r['moe_mae']:>10.4f}  {r['moe_rmse']:>10.4f}  {flag}")
    for k in ("sa_mae", "sa_rmse", "moe_mae", "moe_rmse"):
        totals[k] += r[k]
    totals["n"] += 1

n = totals["n"]
print("  " + "-" * (W - 2))
print(f"  {'Average':<22s}  {totals['sa_mae']/n:>10.4f}  {totals['sa_rmse']/n:>10.4f}"
      f"  {totals['moe_mae']/n:>10.4f}  {totals['moe_rmse']/n:>10.4f}")

# ── Winner summary ────────────────────────────────────────────────────────────
print()
print("=" * W)
print("VERDICT PER DOMAIN")
print("=" * W)
for did in sorted(results):
    r   = results[did]
    d   = r["delta_mae"]
    pct = abs(d) / r["sa_mae"] * 100 if r["sa_mae"] > 0 else 0
    if abs(d) < 0.005:
        verdict = "Tied (within 0.005 MAE)"
    elif d > 0:
        verdict = f"Standalone wins  (MoE is {pct:.1f}% worse, Δ={d:+.4f})"
    else:
        verdict = f"MoE wins  (MoE is {pct:.1f}% better, Δ={d:+.4f})"
    print(f"  {r['label']:<20s}  {verdict}")

# ── Expert routing ────────────────────────────────────────────────────────────
print()
print("=" * W)
print("MoE EXPERT ROUTING (average gate weight per domain — rows sum to 1.0)")
print("=" * W)
e_hdr = "".join(f"  Expert-{i}" for i in range(N_EXPERTS))
print(f"  {'Domain':<22s}{e_hdr}")
print("  " + "-" * (W - 2))
for did in sorted(results):
    r   = results[did]
    gw  = r["gate_weights"]
    top = int(gw.argmax())
    row = "".join(f"  {w:>8.3f}  " for w in gw)
    print(f"  {r['label']:<22s}{row}← top: Expert-{top}")

print()
print("=" * W)
print("WHAT THE ROUTING TABLE TELLS YOU")
print("=" * W)
print("""  Strong specialisation  : one domain sends >0.60 weight to one expert,
                           and different domains prefer different experts.
  Balanced (no routing)  : all domains send ~0.25 to every expert — the gate
                           is acting as a simple ensemble, not routing.
  Current state          : soft routing — experts overlap but show preferences.
                           More data per domain or longer training will sharpen
                           specialisation.
""")
