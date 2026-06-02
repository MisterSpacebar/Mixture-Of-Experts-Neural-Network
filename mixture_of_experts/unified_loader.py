"""
unified_loader.py — Loads all four domains into one padded dataset for MoE training.

Domain IDs:  0=cars  1=water  2=river  3=drone

All four domains are already feature-standardised by their own data_loaders.
Targets:
  - cars / water / river  already return z-scored targets → used as-is
  - drone                 returns log1p(turbidity), z-scored here for consistency

Features are padded with zeros to MAX_FEATURES (=13, drone's width) then a
domain_id column is appended, giving INPUT_DIM=14.

Water is capped at MAX_WATER_ROWS to keep training balanced.
"""

import importlib.util
import os
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..", "neural_network")

# ── Constants (importable by other modules) ───────────────────────────────────
MAX_FEATURES   = 13    # drone has the most features
INPUT_DIM      = MAX_FEATURES + 1   # +1 for domain_id   → 14
MAX_WATER_ROWS = 30_000
N_EXPERTS      = 4

DOMAIN_NAMES   = {0: "Cars (EV)", 1: "Water Stations", 2: "River Canal", 3: "Drone Missions"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _import_loader(domain_name: str):
    """Import a domain's data_loader.py without polluting sys.path."""
    path = os.path.join(_ROOT, domain_name, "data_loader.py")
    spec = importlib.util.spec_from_file_location(f"dl_{domain_name}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subset_to_numpy(subset):
    """Extract (X, y) numpy arrays from a torch random_split Subset."""
    X_all = subset.dataset.tensors[0]
    y_all = subset.dataset.tensors[1]
    idx   = subset.indices
    return X_all[idx].numpy(), y_all[idx].numpy().ravel()


def _pad_and_tag(X, domain_id: int, target_cols: int = MAX_FEATURES):
    """Zero-pad X to target_cols features then append domain_id column."""
    if X.shape[1] < target_cols:
        pad = np.zeros((len(X), target_cols - X.shape[1]), dtype="float32")
        X   = np.hstack([X, pad])
    did = np.full((len(X), 1), float(domain_id), dtype="float32")
    return np.hstack([X, did])


# ── Per-domain loaders ────────────────────────────────────────────────────────
def _load_domain(domain_name: str, domain_id: int,
                 max_rows: int | None = None, zscore_y: bool = False):
    print(f"\n── Loading {DOMAIN_NAMES[domain_id]} …")
    mod    = _import_loader(domain_name)
    result = mod.build_dataset()

    if domain_name == "drone":
        # drone returns (X_train, X_val, y_train, y_val, X_mean, X_std, df)
        X_train, X_val = result[0], result[1]
        y_train, y_val = result[2], result[3]
    else:
        # others return (Subset_train, Subset_val, feature_cols, n_features)
        X_train, y_train = _subset_to_numpy(result[0])
        X_val,   y_val   = _subset_to_numpy(result[1])

    # Optional sub-sampling (used for water to balance domain sizes)
    if max_rows and len(X_train) > max_rows:
        rng = np.random.default_rng(42 + domain_id)
        idx = rng.choice(len(X_train), max_rows, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    # Drone target is log1p (not z-scored); z-score it for a consistent target space
    if zscore_y:
        ym = float(y_train.mean())
        ys = float(y_train.std()) + 1e-8
        y_train = (y_train - ym) / ys
        y_val   = (y_val   - ym) / ys

    X_train = _pad_and_tag(X_train, domain_id)
    X_val   = _pad_and_tag(X_val,   domain_id)

    print(f"   train={len(X_train):,}  val={len(X_val):,}  features={X_train.shape[1]}")
    return X_train, X_val, y_train.astype("float32"), y_val.astype("float32")


# ── Public API ────────────────────────────────────────────────────────────────
def build_combined_dataset():
    """
    Returns
    -------
    X_train, X_val : np.ndarray  shape (N, INPUT_DIM=14)
    y_train, y_val : np.ndarray  shape (N,)   — all z-scored targets
    domain_train, domain_val : np.ndarray  shape (N,)  — int domain labels
    """
    domains = [
        ("cars",  0, None,           False),
        ("water", 1, MAX_WATER_ROWS, False),
        ("river", 2, None,           False),
        ("drone", 3, None,           True),   # True → z-score y here
    ]

    splits = {"train": [], "val": []}
    d_splits = {"train": [], "val": []}

    for name, did, max_rows, zsc in domains:
        Xtr, Xv, ytr, yv = _load_domain(name, did, max_rows, zsc)
        splits["train"].append((Xtr, ytr))
        splits["val"].append((Xv,   yv))
        d_splits["train"].append(np.full(len(Xtr), did, dtype="int32"))
        d_splits["val"].append(  np.full(len(Xv),  did, dtype="int32"))

    X_train = np.vstack([x for x, _ in splits["train"]])
    y_train = np.hstack([y for _, y in splits["train"]])
    X_val   = np.vstack([x for x, _ in splits["val"]])
    y_val   = np.hstack([y for _, y in splits["val"]])
    d_train = np.hstack(d_splits["train"])
    d_val   = np.hstack(d_splits["val"])

    # Shuffle training set so domains are interleaved
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_train))
    X_train, y_train, d_train = X_train[idx], y_train[idx], d_train[idx]

    print(f"\nCombined  train={len(X_train):,}  val={len(X_val):,}  input_dim={INPUT_DIM}")
    return X_train, X_val, y_train, y_val, d_train, d_val


if __name__ == "__main__":
    X_tr, X_v, y_tr, y_v, d_tr, d_v = build_combined_dataset()
    print(f"\nX_train {X_tr.shape}  y_train {y_tr.shape}")
    print(f"X_val   {X_v.shape}  y_val   {y_v.shape}")
