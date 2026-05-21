"""
Feedforward Neural Network — Water Quality (Turbidity Prediction)
------------------------------------------------------------------
Architecture: 5 hidden layers, funnel pattern (128 → 256 → 512 → 256 → 128)
Target      : Turbidity (FNU)
Features    : Temperature, Conductance, Salinity, Pressure, Depth,
              ODO %Sat, ODO mg/L, station_id  (8 total)
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WaterQualityNet(nn.Module):
    """
    5-layer funnel FFNN: input → 128 → 256 → 512 → 256 → 128 → output
    BatchNorm stabilises training; Dropout (p=0.3) prevents overfitting.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.3):
        super().__init__()

        def block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

        self.layers = nn.Sequential(
            block(input_dim, 128),
            block(128, 256),
            block(256, 512),
            block(512, 256),
            block(256, 128),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            total_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import build_dataset

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIM = 1
    BATCH_SIZE = 512   # larger batch suits the ~800k-row dataset
    EPOCHS     = 30
    LR         = 1e-3

    # --- Load real water data ---
    train_set, val_set, feature_names, INPUT_DIM = build_dataset(val_split=0.2, seed=42)
    print(f"\nFeatures ({INPUT_DIM}): {feature_names}")
    print(f"Train: {len(train_set):,} rows  |  Val: {len(val_set):,} rows\n")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model, loss, optimiser ---
    model     = WaterQualityNet(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}\n")

    # --- Training ---
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # --- Save ---
    out_dir   = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "water_quality_net.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
