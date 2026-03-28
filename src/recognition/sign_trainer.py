"""
Build sign templates from recorded data (no neural network).

Computes per-sign centroids from temporal-averaged landmark features.
Also stores all individual clips for k-NN fallback.

Usage:
    venv/bin/python src/recognition/sign_trainer.py

Reads:   data/recordings/{sign_name}/*.npy
Writes:  models/word_model.npz
         models/word_labels.json
"""

import json
import os

import numpy as np

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "recordings")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "word_model.npz")
LABEL_PATH = os.path.join(MODEL_DIR, "word_labels.json")

KEEP_SLICE  = slice(468, 543)    # hands + pose = 75 landmarks
N_FEATURES  = 75 * 3             # 225


def load_all_clips(data_dir):
    """Load clips, compute temporal mean per clip → (N, 225). Also keep full sequences."""
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and any(f.endswith(".npy") for f in os.listdir(os.path.join(data_dir, d)))
    ])
    label_map = {name: i for i, name in enumerate(labels)}

    means = []   # temporal mean per clip → (225,)
    y = []
    all_seqs = []  # raw sequences for analysis

    for name in labels:
        sign_dir = os.path.join(data_dir, name)
        files = sorted(f for f in os.listdir(sign_dir) if f.endswith(".npy"))
        for fname in files:
            arr = np.load(os.path.join(sign_dir, fname))   # (T, 543, 3)
            arr = arr[:, KEEP_SLICE, :]                      # (T, 75, 3)
            arr = arr.reshape(arr.shape[0], -1)              # (T, 225)
            np.nan_to_num(arr, copy=False, nan=0.0)

            # Temporal mean (only over frames that have data)
            nonzero = np.abs(arr).sum(axis=1) > 1e-6
            if nonzero.any():
                mean = arr[nonzero].mean(axis=0)
            else:
                mean = arr.mean(axis=0)

            means.append(mean)
            y.append(label_map[name])
            all_seqs.append(arr)

    return np.array(means, dtype=np.float32), np.array(y), label_map, all_seqs


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No data dir: {DATA_DIR}")
        return

    X_means, y, label_map, all_seqs = load_all_clips(DATA_DIR)
    n_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    print(f"Classes ({n_classes}): {sorted(label_map.keys())}")
    print(f"Total clips: {len(y)}")
    for name, idx in sorted(label_map.items()):
        count = (y == idx).sum()
        print(f"  {name}: {count} clips")

    # Per-feature normalization stats (computed on temporal means)
    feat_mean = X_means.mean(axis=0)
    feat_std  = X_means.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0

    X_norm = (X_means - feat_mean) / feat_std

    # Per-class centroids
    centroids = np.zeros((n_classes, N_FEATURES), dtype=np.float32)
    for i in range(n_classes):
        centroids[i] = X_norm[y == i].mean(axis=0)

    # Quick leave-one-out accuracy check
    correct = 0
    for i in range(len(y)):
        query = X_norm[i]
        # k-NN with k=3 (exclude self)
        dists = np.linalg.norm(X_norm - query, axis=1)
        dists[i] = np.inf  # exclude self
        nn_idx = np.argsort(dists)[:3]
        nn_labels = y[nn_idx]
        pred = np.bincount(nn_labels, minlength=n_classes).argmax()
        if pred == y[i]:
            correct += 1

    loo_acc = correct / len(y)
    print(f"\nLeave-one-out 3-NN accuracy: {loo_acc:.0%}")

    # Centroid distances (sanity check — should be well-separated)
    print("\nCentroid distances:")
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            d = np.linalg.norm(centroids[i] - centroids[j])
            print(f"  {idx_to_label[i]} ↔ {idx_to_label[j]}: {d:.2f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(MODEL_PATH,
             X_norm=X_norm,           # all normalized clip means
             y=y,                      # labels
             centroids=centroids,      # per-class centroids
             feat_mean=feat_mean,
             feat_std=feat_std,
             n_classes=n_classes)

    with open(LABEL_PATH, "w") as f:
        json.dump(idx_to_label, f, indent=2)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH}")


if __name__ == "__main__":
    main()
