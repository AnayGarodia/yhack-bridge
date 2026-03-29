"""
Landmark-based ASL word classifier using a stacked LSTM.

Runs MediaPipe Holistic to extract 543 landmarks per frame, selects
130 key landmarks (hands + lips + nose + eyes), buffers 30 frames,
and classifies the sequence into one of 250 ASL signs.

Usage:
    clf = LandmarkClassifier()
    # In your frame loop:
    annotated, sign, conf = clf.process_frame(bgr_frame)
"""

import collections
import json
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .holistic_tracker import HolisticTracker
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.recognition.holistic_tracker import HolisticTracker

# ---------------------------------------------------------------------------
# Landmark selection (130 key landmarks from the 543 holistic set)
# ---------------------------------------------------------------------------
# Face mesh: 468 landmarks (indices 0-467)
#   Lips: indices around mouth
_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
# Eyes (key corners + iris)
_LEFT_EYE = [263, 249, 390, 373]
_RIGHT_EYE = [33, 7, 160, 144]
# Nose
_NOSE = [1, 2, 98, 327]

# Hands: left 468-488 (21), right 522-542 (21)
_LEFT_HAND = list(range(468, 489))    # 21 landmarks
_RIGHT_HAND = list(range(522, 543))   # 21 landmarks

# Combined: 40 lips + 4+4 eyes + 4 nose + 21+21 hands = 94... extend face to get ~130
# Add more face contour points for expressiveness
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# All selected face indices
_FACE_INDICES = sorted(set(_LIPS + _LEFT_EYE + _RIGHT_EYE + _NOSE + _FACE_OVAL))

# Final combined selection
SELECTED_INDICES = sorted(_FACE_INDICES + _LEFT_HAND + _RIGHT_HAND)
N_LANDMARKS = len(SELECTED_INDICES)  # ~130

SEQ_LEN     = 30    # frames to buffer
STRIDE      = 15    # run inference every N frames
INPUT_DIM   = N_LANDMARKS * 3  # x, y, z per landmark

# Model defaults
_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "..", "models", "landmark_model.pt")
_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "landmark_labels.json")


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class SignLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=3,
                 num_classes=250, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)          # (B, T, hidden)
        out = out[:, -1, :]            # last timestep: (B, hidden)
        out = self.drop(out)
        return self.fc(out)            # (B, num_classes)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def extract_selected_landmarks(lm543):
    """
    From a (543, 3) array, select the ~130 key landmarks.
    Returns (N_LANDMARKS, 3) float32 array. NaN stays as NaN.
    """
    return lm543[SELECTED_INDICES]


def normalize_sequence(seq):
    """
    Normalize a (T, N_LANDMARKS, 3) sequence:
      - Replace NaN with 0
      - Subtract per-sequence mean, divide by std
    Returns (T, N_LANDMARKS*3) float32 array, ready for LSTM.
    """
    seq = seq.copy()
    np.nan_to_num(seq, copy=False, nan=0.0)
    flat = seq.reshape(seq.shape[0], -1)  # (T, N*3)

    # Per-feature mean/std over the time axis
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    flat = (flat - mean) / std
    return flat.astype(np.float32)


def pad_or_truncate(seq, target_len=SEQ_LEN):
    """Pad with zeros or truncate to target_len frames."""
    T = seq.shape[0]
    if T >= target_len:
        # Uniform sample if longer
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return seq[indices]
    else:
        pad = np.zeros((target_len - T, seq.shape[1]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class LandmarkClassifier:
    """
    Real-time ASL sign classifier using landmark sequences + LSTM.

    Same interface as WLASLClassifier for drop-in replacement in main.py.
    """

    def __init__(self, model_path=None, labels_path=None,
                 confidence_threshold=0.40):
        self.confidence_threshold = confidence_threshold
        self.ready = False
        self.device = self._best_device()

        model_path = model_path or _MODEL_PATH
        labels_path = labels_path or _LABELS_PATH

        self._labels = self._load_labels(labels_path)
        self._model = self._load_model(model_path)
        self._tracker = None  # lazy-loaded

        if self._model is not None:
            self.ready = True
            print(f"[LandmarkClassifier] LSTM loaded on {self.device}")
            print(f"[LandmarkClassifier] {N_LANDMARKS} landmarks, "
                  f"seq={SEQ_LEN}f, stride={STRIDE}f, "
                  f"threshold={confidence_threshold:.0%}")

        # Buffers
        self._buf = collections.deque(maxlen=SEQ_LEN)
        self._call_n = 0
        self._hist = collections.deque(maxlen=5)
        self._raw = (None, 0.0)
        self._no_hand_frames = 0

        # Async state
        self._async_lock = threading.Lock()
        self._async_running = False
        self._async_result = (None, 0.0)
        self._async_top5 = []

    @staticmethod
    def _best_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_labels(self, path):
        if not os.path.exists(path):
            print(f"[LandmarkClassifier] labels not found: {path}")
            return [f"sign_{i}" for i in range(250)]
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data[str(i)] for i in range(len(data))]
        return data

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"[LandmarkClassifier] model not found: {path}")
            return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            num_classes = ckpt.get("num_classes", 250)
            input_dim = ckpt.get("input_dim", INPUT_DIM)
            model = SignLSTM(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            return model
        except Exception as exc:
            print(f"[LandmarkClassifier] load failed: {exc}")
            return None

    def _ensure_tracker(self):
        if self._tracker is None:
            self._tracker = HolisticTracker()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr):
        """
        Process one webcam frame. Runs holistic tracking + buffers landmarks.

        Returns:
            annotated_frame, sign_or_None, confidence
        """
        self._ensure_tracker()
        annotated, lm543, lhand_px, rhand_px, norm_left, norm_right = \
            self._tracker.process_frame(frame_bgr)

        hands_visible = (lhand_px is not None) or (rhand_px is not None)
        self.add_landmarks(lm543, hands_visible)

        return annotated, hands_visible

    def add_landmarks(self, lm543, hands_visible=True):
        """Buffer one frame of landmarks (543, 3)."""
        if not hands_visible:
            self._no_hand_frames += 1
            if self._no_hand_frames > 10:
                self._buf.clear()
                self._hist.clear()
                self._raw = (None, 0.0)
                self._no_hand_frames = 0
                with self._async_lock:
                    self._async_result = (None, 0.0)
                    self._async_top5 = []
            return

        self._no_hand_frames = 0
        selected = extract_selected_landmarks(lm543)  # (N_LANDMARKS, 3)
        self._buf.append(selected)
        self._call_n += 1

    # ------------------------------------------------------------------
    # Sync predict
    # ------------------------------------------------------------------

    def predict(self):
        if not self.ready or len(self._buf) < SEQ_LEN:
            return (None, 0.0)
        if self._call_n % STRIDE != 0:
            with self._async_lock:
                return self._async_result
        return self._run_inference(list(self._buf))

    def predict_top5(self):
        with self._async_lock:
            return list(self._async_top5)

    def _run_inference(self, frames):
        seq = np.stack(frames, axis=0)  # (T, N_LANDMARKS, 3)
        flat = normalize_sequence(seq)   # (T, N*3)
        flat = pad_or_truncate(flat, SEQ_LEN)  # (SEQ_LEN, N*3)
        tensor = torch.from_numpy(flat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            conf = conf.item()
            idx = idx.item()

            top5_conf, top5_idx = probs.topk(5, dim=1)
            top5 = [(self._labels[i.item()], c.item())
                    for i, c in zip(top5_idx[0], top5_conf[0])]

        gloss = self._labels[idx] if idx < len(self._labels) else f"sign_{idx}"
        self._raw = (gloss, conf)
        self._hist.append((gloss, conf))

        # Require 2 consecutive agreeing predictions
        result = (None, conf)
        if conf >= self.confidence_threshold and len(self._hist) >= 2:
            last_two = [g for g, c in list(self._hist)[-2:]
                        if c >= self.confidence_threshold]
            if len(last_two) >= 2 and last_two[-1] == last_two[-2]:
                result = (gloss, conf)

        with self._async_lock:
            self._async_result = result
            self._async_top5 = top5

        return result

    # ------------------------------------------------------------------
    # Async inference
    # ------------------------------------------------------------------

    def maybe_run_async(self):
        if not self.ready or len(self._buf) < SEQ_LEN:
            return False
        if self._call_n % STRIDE != 0:
            return False
        if self._async_running:
            return False

        frames = list(self._buf)
        self._async_running = True
        threading.Thread(target=self._infer_bg, args=(frames,), daemon=True).start()
        return True

    def _infer_bg(self, frames):
        try:
            self._run_inference(frames)
        finally:
            self._async_running = False

    def get_async_result(self):
        with self._async_lock:
            return self._async_result, list(self._async_top5)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def on_sign_boundary(self):
        self._buf.clear()
        self._hist.clear()
        self._call_n = 0
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []

    @property
    def raw(self):
        return self._raw

    @property
    def buf_fill(self):
        return len(self._buf)

    def reset(self):
        self._buf.clear()
        self._hist.clear()
        self._call_n = 0
        self._no_hand_frames = 0
        self._raw = (None, 0.0)
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
