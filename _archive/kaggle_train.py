"""
Train ASL Landmark LSTM on Kaggle.
Paste this entire file into a Kaggle notebook cell and run.
Make sure GPU is enabled in notebook settings.

Output files (download from /kaggle/working/):
  - landmark_model.pt
  - landmark_labels.json
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE_INPUT = "/kaggle/input/asl-signs"
OUTPUT_DIR   = "/kaggle/working"

# Landmark selection: 130 key landmarks from the 543 holistic set
# Face lips
LIPS = [61,185,40,39,37,0,267,269,270,409,291,146,91,181,84,17,314,405,321,375,
        78,191,80,81,82,13,312,311,310,415,95,88,178,87,14,317,402,318,324,308]
# Eyes
LEFT_EYE = [263, 249, 390, 373]
RIGHT_EYE = [33, 7, 160, 144]
# Nose
NOSE = [1, 2, 98, 327]
# Face oval
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
             400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
# Hands
LEFT_HAND = list(range(468, 489))
RIGHT_HAND = list(range(522, 543))

FACE_INDICES = sorted(set(LIPS + LEFT_EYE + RIGHT_EYE + NOSE + FACE_OVAL))
SELECTED_INDICES = sorted(FACE_INDICES + LEFT_HAND + RIGHT_HAND)
N_LANDMARKS = len(SELECTED_INDICES)
INPUT_DIM = N_LANDMARKS * 3
SEQ_LEN = 30

print(f"Selected landmarks: {N_LANDMARKS}, input dim: {INPUT_DIM}")

# Type → global index offset mapping
TYPE_OFFSET = {
    "face": 0,        # 0-467
    "left_hand": 468,  # 468-488
    "pose": 489,       # 489-521
    "right_hand": 522, # 522-542
}

# Pre-build a set for fast lookup
SELECTED_SET = set(SELECTED_INDICES)
# Map global_index → position in our selected array
GLOBAL_TO_LOCAL = {g: i for i, g in enumerate(SELECTED_INDICES)}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SignLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=3,
                 num_classes=250, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.fc(out)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet_to_sequence(path):
    """
    Load a long-form parquet (frame, type, landmark_index, x, y, z)
    and convert to (T, N_LANDMARKS, 3) array of selected landmarks.
    """
    df = pd.read_parquet(path)

    # Compute global landmark index
    df["global_idx"] = df["type"].map(TYPE_OFFSET) + df["landmark_index"]

    # Filter to only selected landmarks
    df = df[df["global_idx"].isin(SELECTED_SET)]

    # Map to local index
    df["local_idx"] = df["global_idx"].map(GLOBAL_TO_LOCAL)

    # Get frame range
    frames = sorted(df["frame"].unique())
    n_frames = len(frames)
    frame_map = {f: i for i, f in enumerate(frames)}

    # Build array
    seq = np.full((n_frames, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        t = frame_map[row["frame"]]
        l = int(row["local_idx"])
        seq[t, l, 0] = row["x"]
        seq[t, l, 1] = row["y"]
        seq[t, l, 2] = row["z"]

    return seq


def load_parquet_fast(path):
    """Faster vectorized version of load_parquet_to_sequence."""
    df = pd.read_parquet(path)

    df["global_idx"] = df["type"].map(TYPE_OFFSET) + df["landmark_index"]
    mask = df["global_idx"].isin(SELECTED_SET)
    df = df[mask].copy()
    df["local_idx"] = df["global_idx"].map(GLOBAL_TO_LOCAL)

    frames = sorted(df["frame"].unique())
    n_frames = len(frames)
    frame_map = {f: i for i, f in enumerate(frames)}
    df["frame_idx"] = df["frame"].map(frame_map)

    seq = np.full((n_frames, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    t_idx = df["frame_idx"].values
    l_idx = df["local_idx"].values.astype(int)
    seq[t_idx, l_idx, 0] = df["x"].values
    seq[t_idx, l_idx, 1] = df["y"].values
    seq[t_idx, l_idx, 2] = df["z"].values

    return seq


def normalize_sequence(seq):
    """(T, N, 3) → (T, N*3) normalized."""
    seq = seq.copy()
    np.nan_to_num(seq, copy=False, nan=0.0)
    flat = seq.reshape(seq.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((flat - mean) / std).astype(np.float32)


def pad_or_truncate(seq, target_len=SEQ_LEN):
    T = seq.shape[0]
    if T >= target_len:
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return seq[indices]
    else:
        pad = np.zeros((target_len - T, seq.shape[1]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)


class ASLDataset(Dataset):
    def __init__(self, df, label_map, data_dir):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row["path"])
        label = self.label_map[row["sign"]]

        try:
            seq = load_parquet_fast(path)
            flat = normalize_sequence(seq)
            flat = pad_or_truncate(flat, SEQ_LEN)
            return torch.from_numpy(flat), label
        except Exception as e:
            return torch.zeros(SEQ_LEN, INPUT_DIM), label

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * bx.size(0)
        correct += (logits.argmax(1) == by).sum().item()
        total += bx.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, correct5, total = 0.0, 0, 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        loss = F.cross_entropy(logits, by)
        total_loss += loss.item() * bx.size(0)
        correct += (logits.argmax(1) == by).sum().item()
        correct5 += (logits.topk(5, 1).indices == by.unsqueeze(1)).any(1).sum().item()
        total += bx.size(0)
    return total_loss / total, correct / total, correct5 / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EPOCHS = 50
    BATCH = 64
    LR = 1e-3
    HIDDEN = 256
    LAYERS = 3
    DROPOUT = 0.3
    WORKERS = 2

    # Load metadata
    train_csv = os.path.join(KAGGLE_INPUT, "train.csv")
    label_json = os.path.join(KAGGLE_INPUT, "sign_to_prediction_index_map.json")

    df = pd.read_csv(train_csv)
    with open(label_json) as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}
    num_classes = len(sign_to_idx)

    print(f"Dataset: {len(df)} sequences, {num_classes} classes")

    # Verify a sample parquet loads
    sample = os.path.join(KAGGLE_INPUT, df.iloc[0]["path"])
    seq = load_parquet_fast(sample)
    print(f"Sample parquet: {seq.shape} (frames, landmarks, xyz)")

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["sign"], random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = ASLDataset(train_df, sign_to_idx, KAGGLE_INPUT)
    val_ds = ASLDataset(val_df, sign_to_idx, KAGGLE_INPUT)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=WORKERS, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SignLSTM(INPUT_DIM, HIDDEN, LAYERS, num_classes, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    best_acc = 0.0
    print(f"\n{'Ep':>3} | {'TrLoss':>7} {'TrAcc':>6} | {'VaLoss':>7} {'Top1':>6} {'Top5':>6} | {'LR':>8} {'Time':>5}")
    print("-" * 72)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc, va_top5 = evaluate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"{epoch:3d} | {tr_loss:7.4f} {tr_acc:5.1%} | {va_loss:7.4f} {va_acc:5.1%} {va_top5:5.1%} | {lr:8.6f} {dt:4.0f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            save_path = os.path.join(OUTPUT_DIR, "landmark_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN,
                "num_layers": LAYERS,
                "n_landmarks": N_LANDMARKS,
                "seq_len": SEQ_LEN,
                "selected_indices": SELECTED_INDICES,
                "val_acc_top1": va_acc,
                "val_acc_top5": va_top5,
                "epoch": epoch,
            }, save_path)
            labels_path = os.path.join(OUTPUT_DIR, "landmark_labels.json")
            with open(labels_path, "w") as f:
                json.dump(idx_to_sign, f, indent=2)
            print(f"     ** Saved best (top1={va_acc:.1%}, top5={va_top5:.1%})")

    print(f"\nDone. Best val top-1: {best_acc:.1%}")
    print(f"Download from Output tab:")
    print(f"  {OUTPUT_DIR}/landmark_model.pt")
    print(f"  {OUTPUT_DIR}/landmark_labels.json")


main()
