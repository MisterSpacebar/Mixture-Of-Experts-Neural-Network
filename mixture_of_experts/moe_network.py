"""
moe_network.py — Mixture-of-Experts network trained on all four domains.

Architecture
------------
  Input (14 dims: 13 padded features + domain_id)
        ↓
  Gate  (14 → 64 → ReLU → 4 → Softmax)      ← learns which expert handles what
        ↓                         ↓
  Expert-0  Expert-1  Expert-2  Expert-3     ← each a 3-layer funnel
  14→128→256→128→1  (×4)
        ↓
  output = Σ  gate_weight_i × expert_i(x)

A load-balancing auxiliary loss (0.01 × variance of expert utilisation)
prevents expert collapse — where the gate routes everything to one expert.

Saves weights to output/moe_net.pt
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from unified_loader import build_combined_dataset, INPUT_DIM, N_EXPERTS, DOMAIN_NAMES  # noqa

# ── Hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN       = [128, 256, 128]
DROPOUT      = 0.3
BATCH_SIZE   = 512
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LB_WEIGHT    = 0.01    # load-balancing loss coefficient

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
PT_FILE = os.path.join(OUT_DIR, "moe_net.pt")


# ── Model ─────────────────────────────────────────────────────────────────────
class MoENet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, n_experts=N_EXPERTS,
                 hidden=None, dropout=DROPOUT):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN

        # Build experts
        self.experts = nn.ModuleList([
            self._make_expert(input_dim, hidden, dropout)
            for _ in range(n_experts)
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
            nn.Softmax(dim=-1),
        )

    @staticmethod
    def _make_expert(input_dim, hidden, dropout):
        dims   = [input_dim] + hidden
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        layers.append(nn.Linear(hidden[-1], 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        gate_w      = self.gate(x)                                         # (B, E)
        expert_outs = torch.stack([e(x).squeeze(-1) for e in self.experts], dim=1)  # (B, E)
        prediction  = (gate_w * expert_outs).sum(dim=1)                    # (B,)
        return prediction, gate_w


# ── Training helpers ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_mse  = 0.0
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            preds, gate_w = model(X_batch)
            mse_loss = criterion(preds, y_batch)

            # Load-balancing: penalise unequal expert utilisation
            expert_load = gate_w.mean(0)                          # (E,) mean gate weight
            lb_loss     = ((expert_load - 1.0 / N_EXPERTS) ** 2).sum()
            loss        = mse_loss + LB_WEIGHT * lb_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_mse  += mse_loss.item() * len(y_batch)
            total_loss += loss.item()     * len(y_batch)

    n = len(loader.dataset)
    return total_mse / n, total_loss / n


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LOADING COMBINED DATASET")
    print("=" * 60)
    X_train, X_val, y_train, y_val, d_train, d_val = build_combined_dataset()

    X_tr_t = torch.from_numpy(X_train)
    X_v_t  = torch.from_numpy(X_val)
    y_tr_t = torch.from_numpy(y_train)
    y_v_t  = torch.from_numpy(y_val)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_v_t,  y_v_t),
                              batch_size=BATCH_SIZE, shuffle=False)

    # Print domain breakdown in val set
    print("\nVal-set domain breakdown:")
    for did, name in DOMAIN_NAMES.items():
        n = (d_val == did).sum()
        print(f"  {name:<20s}: {n:,} rows")

    model     = MoENet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Training {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  {len(y_train):,} train rows")
    print(f"\n{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}")
    print("-" * 36)

    for epoch in range(1, EPOCHS + 1):
        tr_mse,  _ = run_epoch(model, train_loader, criterion, optimizer)
        val_mse, _ = run_epoch(model, val_loader,   criterion)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_mse:>12.4f}  {val_mse:>12.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), PT_FILE)
    print(f"\nSaved → {PT_FILE}")
