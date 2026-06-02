"""
neural_network.py — Drone water-quality turbidity prediction

Architecture: 13 → 128 → 256 → 128 → 1
Smaller than the other models to match the dataset size (~2 k rows).
Target: Turbidity FNU (log1p-scaled)
Saves weights to output/drone_quality_net.pt
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_dataset, FEATURE_COLS  # noqa: E402

# ── Hyper-parameters ─────────────────────────────────────────────────────────
INPUT_DIM  = len(FEATURE_COLS)   # 13
OUTPUT_DIM = 1
HIDDEN     = [128, 256, 128]
DROPOUT    = 0.4
BATCH_SIZE = 32
EPOCHS     = 80
LR         = 1e-3
WEIGHT_DECAY = 1e-4

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
PT_FILE = os.path.join(OUT_DIR, "drone_quality_net.pt")


# ── Model ─────────────────────────────────────────────────────────────────────
class DroneQualityNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        dims = [input_dim] + HIDDEN
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ))
        layers.append(nn.Linear(HIDDEN[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ── Training ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            preds = model(X_batch).squeeze(1)
            loss  = criterion(preds, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    print("Loading drone mission data …")
    X_train, X_val, y_train, y_val, X_mean, X_std, _ = build_dataset()

    X_train_t = torch.from_numpy(X_train)
    X_val_t   = torch.from_numpy(X_val)
    y_train_t = torch.from_numpy(y_train)
    y_val_t   = torch.from_numpy(y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),
                              batch_size=BATCH_SIZE, shuffle=False)

    model     = DroneQualityNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training {EPOCHS} epochs  |  train rows: {len(y_train)}  val rows: {len(y_val)}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}")
    print("-" * 36)

    for epoch in range(1, EPOCHS + 1):
        tr_loss  = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader,   criterion)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>12.4f}  {val_loss:>12.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), PT_FILE)
    print(f"\nSaved → {PT_FILE}")
