"""
Inspect a saved EVFleetNet .pt weights file.
Run from any directory:
    python neural_network/cars/output/inspect_model.py
"""

import os
import sys

import torch

# Allow importing EVFleetNet from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neural_network import EVFleetNet  # noqa: E402

PT_FILE = os.path.join(os.path.dirname(__file__), "ev_fleet_net.pt")

if not os.path.exists(PT_FILE):
    print(f"ERROR: {PT_FILE} not found. Run neural_network.py first to generate it.")
    sys.exit(1)

# -------------------------------------------------------------------
# 1. Load raw state dict — shows every weight tensor and its shape
# -------------------------------------------------------------------
state = torch.load(PT_FILE, weights_only=True)

print("=" * 60)
print("RAW STATE DICT  (layer name → tensor shape → parameter count)")
print("=" * 60)
total = 0
for name, tensor in state.items():
    count = tensor.numel()
    total += count
    print(f"  {name:<45s}  {str(tuple(tensor.shape)):<20s}  {count:>8,}")
print("-" * 60)
print(f"  {'TOTAL':<45s}  {'':20s}  {total:>8,}")

# -------------------------------------------------------------------
# 2. Reconstruct the full model and verify weights load cleanly
# -------------------------------------------------------------------
INPUT_DIM  = state["layers.0.0.weight"].shape[1]  # inferred from saved weights
OUTPUT_DIM = 1

model = EVFleetNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
model.load_state_dict(state)
model.eval()

print()
print("=" * 60)
print("FULL MODEL ARCHITECTURE")
print("=" * 60)
print(model)

print()
print("=" * 60)
print("QUICK INFERENCE SANITY CHECK  (random input)")
print("=" * 60)
x = torch.randn(4, INPUT_DIM)          # batch of 4
with torch.no_grad():
    out = model(x)
print(f"  Input shape : {tuple(x.shape)}")
print(f"  Output shape: {tuple(out.shape)}")
print(f"  Sample output values: {out.squeeze().tolist()}")
print()
print("Model loaded and verified successfully.")
