#!/usr/bin/env python3
"""
Train a stacked LSTM on the Kaggle ASL Signs landmark dataset.

Usage:
    python src/recognition/train_landmark_model.py
    python src/recognition/train_landmark_model.py --epochs 30 --batch_size 128

Expects:
    data/train.csv
    data/train_landmark_files/  (parquet files)
    data/sign_to_prediction_index_map.json

Produces:
    models/landmark_model.pt
    models/landmark_labels.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.recognition.landmark_classifier import (
    SELECTED_INDICES, N_LANDMARKS, SEQ_LEN, INPUT_DIM,
    SignLSTM, normalize_sequence, pad_or_truncate,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRAIN_CSV   = os.path.join(DATA_DIR, "train.csv")
LABEL_MAP   = os.path.join(DATA_DIR, "sign_to_prediction_index_map.json")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "landmark_model.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "landmark_labels.json")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ASLLandmarkDataset(Dataset):
    """Loads parquet landmark sequences, selects key landmarks, normalizes."""

    def __init__(self, df, label_map, data_dir, seq_len=SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.data_dir = data_dir
        self.seq_len = seq_len

        # Precompute column names for selected landmarks
        self._x_cols = [f"x_face_{i}" if i < 468
                        else f"x_left_hand_{i - 468}" if i < 489
                        else f"x_pose_{i - 489}" if i < 522
                        else f"x_right_hand_{i - 522}"
                        for i in SELECTED_INDICES]
        self._y_cols = [c.replace("x_", "y_") for c in self._x_cols]
        self._z_cols = [c.replace("x_", "z_") for c in self._x_cols]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row["path"])
        sign = row["sign"]
        label = self.label_map[sign]

        try:
            pq = pd.read_parquet(path)
        except Exception:
            # Return zeros on read failure
            return torch.zeros(self.seq_len, INPUT_DIM), label

        # Extract selected landmarks as (T, N_LANDMARKS, 3)
        try:
            x = pq[self._x_cols].values
            y = pq[self._y_cols].values
            z = pq[self._z_cols].values
        except KeyError:
            # Fallback: try generic column naming
            x, y, z = self._extract_generic(pq)

        seq = np.stack([x, y, z], axis=-1)  # (T, N_LANDMARKS, 3)
        flat = normalize_sequence(seq)       # (T, N_LANDMARKS*3)
        flat = pad_or_truncate(flat, self.seq_len)  # (SEQ_LEN, INPUT_DIM)

        return torch.from_numpy(flat), label

    def _extract_generic(self, pq):
        """Fallback: columns named x_0..x_542, y_0..y_542, z_0..z_542."""
        x_cols = [f"x_{i}" for i in SELECTED_INDICES]
        y_cols = [f"y_{i}" for i in SELECTED_INDICES]
        z_cols = [f"z_{i}" for i in SELECTED_INDICES]

        # Check which naming convention exists
        if x_cols[0] in pq.columns:
            return pq[x_cols].values, pq[y_cols].values, pq[z_cols].values

        # Try face/hand/pose prefixed columns
        raise KeyError(f"Cannot find landmark columns. Available: {list(pq.columns[:10])}")


def detect_column_format(parquet_path):
    """Read one parquet to figure out column naming convention."""
    pq = pd.read_parquet(parquet_path)
    cols = list(pq.columns)
    if "x_face_0" in cols:
        return "prefixed"
    elif "x_0" in cols:
        return "indexed"
    else:
        print(f"Unknown column format. Sample columns: {cols[:20]}")
        return "unknown"


class ASLDatasetIndexed(Dataset):
    """Dataset using x_0..x_542 column naming."""

    def __init__(self, df, label_map, data_dir, seq_len=SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.data_dir = data_dir
        self.seq_len = seq_len
        self._x_cols = [f"x_{i}" for i in SELECTED_INDICES]
        self._y_cols = [f"y_{i}" for i in SELECTED_INDICES]
        self._z_cols = [f"z_{i}" for i in SELECTED_INDICES]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row["path"])
        label = self.label_map[row["sign"]]

        try:
            pq = pd.read_parquet(path, columns=self._x_cols + self._y_cols + self._z_cols)
            x = pq[self._x_cols].values  # (T, N_LANDMARKS)
            y = pq[self._y_cols].values
            z = pq[self._z_cols].values
            seq = np.stack([x, y, z], axis=-1).astype(np.float32)  # (T, N, 3)
            flat = normalize_sequence(seq)
            flat = pad_or_truncate(flat, self.seq_len)
            return torch.from_numpy(flat), label
        except Exception:
            return torch.zeros(self.seq_len, INPUT_DIM), label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, correct_top5, total = 0.0, 0, 0, 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()

        top5 = logits.topk(5, dim=1).indices
        correct_top5 += (top5 == batch_y.unsqueeze(1)).any(dim=1).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total, correct_top5 / total


def main():
    parser = argparse.ArgumentParser(description="Train ASL landmark LSTM")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # ---- Load metadata ----
    if not os.path.exists(TRAIN_CSV):
        print(f"ERROR: {TRAIN_CSV} not found. Download the Kaggle ASL Signs dataset first.")
        sys.exit(1)

    df = pd.read_csv(TRAIN_CSV)
    with open(LABEL_MAP) as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}
    num_classes = len(sign_to_idx)

    print(f"Dataset: {len(df)} sequences, {num_classes} classes")
    print(f"Selected landmarks: {N_LANDMARKS}, input dim: {INPUT_DIM}")
    print(f"Sequence length: {SEQ_LEN}")

    # ---- Detect column format ----
    sample_path = os.path.join(DATA_DIR, df.iloc[0]["path"])
    if not os.path.exists(sample_path):
        print(f"ERROR: Parquet file not found: {sample_path}")
        print("Make sure train_landmark_files/ is extracted in data/")
        sys.exit(1)

    col_fmt = detect_column_format(sample_path)
    print(f"Column format: {col_fmt}")

    # ---- Train/val split (stratified) ----
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["sign"], random_state=42
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ---- Create datasets ----
    DatasetClass = ASLDatasetIndexed if col_fmt == "indexed" else ASLLandmarkDataset
    train_ds = DatasetClass(train_df, sign_to_idx, DATA_DIR, seq_len=SEQ_LEN)
    val_ds = DatasetClass(val_df, sign_to_idx, DATA_DIR, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = SignLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    best_val_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | "
          f"{'Val Loss':>8} {'Val Top1':>8} {'Val Top5':>8} | {'LR':>8} {'Time':>6}")
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"{epoch:5d} | {train_loss:10.4f} {train_acc:9.1%} | "
              f"{val_loss:8.4f} {val_acc:8.1%} {val_top5:8.1%} | "
              f"{lr:8.6f} {dt:5.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "input_dim": INPUT_DIM,
                "hidden_dim": args.hidden,
                "num_layers": args.layers,
                "n_landmarks": N_LANDMARKS,
                "seq_len": SEQ_LEN,
                "val_acc_top1": val_acc,
                "val_acc_top5": val_top5,
                "epoch": epoch,
            }, MODEL_PATH)

            # Save label map (index -> sign name)
            with open(LABELS_PATH, "w") as f:
                json.dump(idx_to_sign, f, indent=2)

            print(f"       ** Saved best model (top1={val_acc:.1%}, top5={val_top5:.1%})")

    print(f"\nTraining complete. Best val top-1: {best_val_acc:.1%}")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABELS_PATH}")


if __name__ == "__main__":
    main()
