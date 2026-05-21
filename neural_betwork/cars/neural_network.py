"""
Feedforward Neural Network for EV Fleet Tabular Data
-----------------------------------------------------
Architecture: 5 hidden layers, funnel pattern (128 → 256 → 512 → 256 → 128)

Diminishing Returns Guide
--------------------------
Layers: Strong gains at 1–3 hidden layers. Returns drop off after ~5–6 for basic
        FFNNs (vanishing gradients without residual connections). Each layer past
        that adds compute with marginal accuracy improvement.

Nodes:  Doubling 64→128→256 shows clear gains on most tabular tasks.
        256→512 is modest. 512→1024 is typically negligible.
        Diminishing returns kick in hard around 512 nodes/layer.

This network sits at the sweet spot: 5 hidden layers, max 512 nodes.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EVFleetNet(nn.Module):
    """
    5-layer funnel FFNN with BatchNorm and Dropout.

    Layer widths:  input → 128 → 256 → 512 → 256 → 128 → output

    BatchNorm stabilises training; Dropout (p=0.3) prevents overfitting.
    Both are disabled automatically during model.eval().
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        def block(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

        self.layers = nn.Sequential(
            block(input_dim, 128),   # Layer 1 — 128 nodes
            block(128, 256),         # Layer 2 — 256 nodes  (strong gain)
            block(256, 512),         # Layer 3 — 512 nodes  (strong gain)
            block(512, 256),         # Layer 4 — 256 nodes  (moderate gain)
            block(256, 128),         # Layer 5 — 128 nodes  (compression)
            nn.Linear(128, output_dim),  # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(X_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Entry point — train on real EV fleet data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import build_dataset

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    OUTPUT_DIM  = 1
    BATCH_SIZE  = 128
    EPOCHS      = 40
    LR          = 1e-3

    # --- Real EV fleet dataset ---
    train_set, val_set, feature_names, INPUT_DIM = build_dataset(val_split=0.2, seed=42)
    print(f"\nFeatures ({INPUT_DIM}): {feature_names}")
    print(f"Train: {len(train_set):,} rows  |  Val: {len(val_set):,} rows\n")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, num_workers=0)

    # --- Model, loss, optimiser ---
    model     = EVFleetNet(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
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
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ev_fleet_net.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
