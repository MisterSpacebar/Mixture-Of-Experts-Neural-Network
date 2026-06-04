"""
visual_compare.py — Detailed graphical comparison of standalone models vs MoE.

Run from workspace root:
    python comparison/visual_compare.py

Saves comparison/output/model_comparison.png  (multi-panel figure)

Panels
------
1. MAE bar chart — standalone vs MoE per domain
2. RMSE bar chart — standalone vs MoE per domain
3. MoE expert routing heatmap
4. Residual distributions per domain (violin) — standalone
5. Residual distributions per domain (violin) — MoE
6. Actual vs Predicted scatter — standalone (sample)
7. Actual vs Predicted scatter — MoE (sample)
8. Δ MAE waterfall (positive = standalone better, negative = MoE better)
"""

import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import torch

# ── Path setup ─────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.join(_HERE, "..")
ROOT_NN  = os.path.join(ROOT, "neural_network")
ROOT_MOE = os.path.join(ROOT, "mixture_of_experts")
OUT_DIR  = os.path.join(_HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT_MOE)
from unified_loader import _pad_and_tag, INPUT_DIM, N_EXPERTS, MAX_FEATURES  # noqa
from moe_network     import MoENet                                            # noqa

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
COLOUR_SA  = "#4C72B0"   # standalone blue
COLOUR_MOE = "#DD8452"   # MoE orange
ALPHA_VIOL = 0.55

# ── Domain config ──────────────────────────────────────────────────────────
DOMAINS = [
    ("cars",  0, "EVFleetNet",      "ev_fleet_net.pt"),
    ("water", 1, "WaterQualityNet", "water_quality_net.pt"),
    ("river", 2, "RiverLevelNet",   "river_level_net.pt"),
    ("drone", 3, "DroneQualityNet", "drone_quality_net.pt"),
]
LABELS   = {0: "Cars\n(EV Fleet)", 1: "Water\nStations",
            2: "River\nCanal",     3: "Drone\nMissions"}
SHORT    = {0: "Cars", 1: "Water", 2: "River", 3: "Drone"}
SAMPLE_N = 600   # points per scatter panel


# ── Helpers ────────────────────────────────────────────────────────────────
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subset_to_numpy(subset):
    X = subset.dataset.tensors[0][subset.indices].numpy()
    y = subset.dataset.tensors[1][subset.indices].numpy().ravel()
    return X, y


def _infer_sa(model, X_np, batch=4096):
    out = []
    for i in range(0, len(X_np), batch):
        t = torch.from_numpy(X_np[i:i+batch])
        with torch.no_grad():
            out.append(model(t).squeeze(-1).numpy())
    return np.concatenate(out)


def _infer_moe(model, X_np, batch=4096):
    preds, gates = [], []
    for i in range(0, len(X_np), batch):
        t = torch.from_numpy(X_np[i:i+batch])
        with torch.no_grad():
            p, g = model(t)
            preds.append(p.numpy())
            gates.append(g.numpy())
    return np.concatenate(preds), np.vstack(gates)


# ── Load MoE ───────────────────────────────────────────────────────────────
MoE_PT   = os.path.join(ROOT_MOE, "output", "moe_net.pt")
moe_state = torch.load(MoE_PT, weights_only=True)
moe_model = MoENet(input_dim=INPUT_DIM, n_experts=N_EXPERTS)
moe_model.load_state_dict(moe_state)
moe_model.eval()

# ── Collect results ────────────────────────────────────────────────────────
print("Collecting predictions …")
results = {}

for folder, did, cls_name, pt_name in DOMAINS:
    dl_mod = _load_module(os.path.join(ROOT_NN, folder, "data_loader.py"), f"dl_{folder}")
    nn_mod = _load_module(os.path.join(ROOT_NN, folder, "neural_network.py"), f"nn_{folder}")

    result = dl_mod.build_dataset()

    if folder == "drone":
        X_val, y_val_raw = result[1], result[3]
        y_train = result[2]
        ym = float(y_train.mean());  ys = float(y_train.std()) + 1e-8
        y_val_zs = (y_val_raw - ym) / ys
    else:
        X_val, y_val_zs = _subset_to_numpy(result[1])

    pt_path  = os.path.join(ROOT_NN, folder, "output", pt_name)
    sa_state = torch.load(pt_path, weights_only=True)
    input_dim_sa = sa_state["layers.0.0.weight"].shape[1]
    sa_model = getattr(nn_mod, cls_name)(input_dim=input_dim_sa, output_dim=1)
    sa_model.load_state_dict(sa_state)
    sa_model.eval()

    if folder == "drone":
        sa_raw  = _infer_sa(sa_model, X_val)
        sa_pred = (sa_raw - ym) / ys
    else:
        sa_pred = _infer_sa(sa_model, X_val)

    X_moe = _pad_and_tag(X_val, did).astype("float32")
    moe_pred, gate_w = _infer_moe(moe_model, X_moe)

    sa_err  = sa_pred  - y_val_zs
    moe_err = moe_pred - y_val_zs

    rng     = np.random.default_rng(42 + did)
    si      = rng.choice(len(y_val_zs), min(SAMPLE_N, len(y_val_zs)), replace=False)

    results[did] = dict(
        y         = y_val_zs,
        sa_pred   = sa_pred,   sa_err = sa_err,
        moe_pred  = moe_pred,  moe_err = moe_err,
        gate_w    = gate_w,
        sa_mae    = float(np.abs(sa_err).mean()),
        sa_rmse   = float(np.sqrt((sa_err**2).mean())),
        moe_mae   = float(np.abs(moe_err).mean()),
        moe_rmse  = float(np.sqrt((moe_err**2).mean())),
        sample_idx = si,
    )
    print(f"  {SHORT[did]:8s}  SA_MAE={results[did]['sa_mae']:.4f}  "
          f"MoE_MAE={results[did]['moe_mae']:.4f}")

# ── Build figure ───────────────────────────────────────────────────────────
print("\nRendering figure …")

fig = plt.figure(figsize=(20, 26), facecolor="#1a1a2e")
fig.suptitle("Standalone Domain Networks  vs  Mixture-of-Experts\n"
             "All metrics in normalised z-score space  (0 = perfect, 1 = 1 std-dev error)",
             fontsize=15, fontweight="bold", color="white", y=0.995)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38,
                       top=0.97, bottom=0.03, left=0.06, right=0.97)

domain_ids  = sorted(results)
x           = np.arange(len(domain_ids))
bar_w       = 0.35
tick_labels = [LABELS[d] for d in domain_ids]
ax_kw       = dict(facecolor="#16213e", labelcolor="white",
                   titlecolor="white", tick_params=dict(colors="white"))

def _style(ax, title):
    ax.set_facecolor("#16213e")
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

# ── Panel 1: MAE bars ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sa_maes  = [results[d]["sa_mae"]  for d in domain_ids]
moe_maes = [results[d]["moe_mae"] for d in domain_ids]
b1 = ax1.bar(x - bar_w/2, sa_maes,  bar_w, label="Standalone", color=COLOUR_SA,  alpha=0.88)
b2 = ax1.bar(x + bar_w/2, moe_maes, bar_w, label="MoE",        color=COLOUR_MOE, alpha=0.88)
ax1.set_xticks(x); ax1.set_xticklabels(tick_labels, fontsize=9)
ax1.set_ylabel("MAE (z-score units)")
ax1.legend(facecolor="#2a2a4a", labelcolor="white", edgecolor="#555577")
for bar in list(b1) + list(b2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=8)
_style(ax1, "Mean Absolute Error (lower = better)")

# ── Panel 2: RMSE bars ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sa_rmses  = [results[d]["sa_rmse"]  for d in domain_ids]
moe_rmses = [results[d]["moe_rmse"] for d in domain_ids]
b3 = ax2.bar(x - bar_w/2, sa_rmses,  bar_w, label="Standalone", color=COLOUR_SA,  alpha=0.88)
b4 = ax2.bar(x + bar_w/2, moe_rmses, bar_w, label="MoE",        color=COLOUR_MOE, alpha=0.88)
ax2.set_xticks(x); ax2.set_xticklabels(tick_labels, fontsize=9)
ax2.set_ylabel("RMSE (z-score units)")
ax2.legend(facecolor="#2a2a4a", labelcolor="white", edgecolor="#555577")
for bar in list(b3) + list(b4):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=8)
_style(ax2, "Root Mean Squared Error (lower = better)")

# ── Panel 3: Expert routing heatmap ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
heat = np.array([results[d]["gate_w"].mean(axis=0) for d in domain_ids])
sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[f"Expert-{i}" for i in range(N_EXPERTS)],
            yticklabels=[SHORT[d] for d in domain_ids],
            ax=ax3, linewidths=0.5, linecolor="#333355",
            cbar_kws={"shrink": 0.8, "label": "Avg gate weight"})
ax3.tick_params(colors="white", labelsize=9)
ax3.collections[0].colorbar.ax.yaxis.label.set_color("white")
ax3.collections[0].colorbar.ax.tick_params(colors="white")
_style(ax3, "MoE Expert Routing Heatmap  (rows sum to 1.0)")

# ── Panel 4: Δ MAE waterfall ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
deltas = [results[d]["moe_mae"] - results[d]["sa_mae"] for d in domain_ids]
colours = [COLOUR_MOE if d < 0 else COLOUR_SA for d in deltas]
bars = ax4.bar(x, deltas, color=colours, alpha=0.88, width=0.55)
ax4.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax4.set_xticks(x); ax4.set_xticklabels(tick_labels, fontsize=9)
ax4.set_ylabel("Δ MAE  (MoE − Standalone)")
for bar, val in zip(bars, deltas):
    va  = "bottom" if val >= 0 else "top"
    off = 0.002 if val >= 0 else -0.002
    ax4.text(bar.get_x() + bar.get_width()/2, val + off,
             f"{val:+.4f}", ha="center", va=va, color="white", fontsize=9)
legend_els = [Patch(facecolor=COLOUR_SA,  label="Standalone wins (MoE worse)"),
              Patch(facecolor=COLOUR_MOE, label="MoE wins (MoE better)")]
ax4.legend(handles=legend_els, facecolor="#2a2a4a", labelcolor="white", edgecolor="#555577")
_style(ax4, "Δ MAE Waterfall  (negative = MoE is better)")

# ── Panel 5 & 6: Residual violin — standalone / MoE ───────────────────────
clip_z = 4.0  # clip extreme residuals for readability
for col, key, title in [(0, "sa_err", "Standalone Residual Distribution"),
                        (1, "moe_err", "MoE Residual Distribution")]:
    ax = fig.add_subplot(gs[2, col])
    data_list  = [np.clip(results[d][key], -clip_z, clip_z) for d in domain_ids]
    vparts = ax.violinplot(data_list, positions=x, widths=0.6, showmedians=True,
                           showextrema=False)
    fill_c = COLOUR_SA if col == 0 else COLOUR_MOE
    for body in vparts["bodies"]:
        body.set_facecolor(fill_c)
        body.set_alpha(ALPHA_VIOL)
    vparts["cmedians"].set_color("white")
    ax.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Residual (z-score units)")
    ax.set_ylim(-clip_z - 0.3, clip_z + 0.3)
    _style(ax, title)

# ── Panel 7 & 8: Actual vs Predicted scatter ──────────────────────────────
for col, pred_key, title in [(0, "sa_pred", "Standalone: Actual vs Predicted"),
                              (1, "moe_pred", "MoE: Actual vs Predicted")]:
    ax = fig.add_subplot(gs[3, col])
    domain_colours = sns.color_palette("husl", len(domain_ids))
    handles = []
    for i, did in enumerate(domain_ids):
        si   = results[did]["sample_idx"]
        y_s  = results[did]["y"][si]
        p_s  = results[did][pred_key][si]
        ax.scatter(y_s, p_s, s=14, alpha=0.45, color=domain_colours[i], label=SHORT[did])
        handles.append(Patch(facecolor=domain_colours[i], label=SHORT[did]))
    # Perfect-prediction line
    all_y = np.concatenate([results[d]["y"][results[d]["sample_idx"]] for d in domain_ids])
    lim   = max(abs(all_y.min()), abs(all_y.max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="white", linewidth=0.9,
            linestyle="--", alpha=0.6, label="Perfect")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual (z-score)"); ax.set_ylabel("Predicted (z-score)")
    ax.legend(handles=handles, facecolor="#2a2a4a", labelcolor="white",
              edgecolor="#555577", markerscale=1.5, fontsize=9)
    _style(ax, title)

# ── Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "model_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved → {out_path}")
