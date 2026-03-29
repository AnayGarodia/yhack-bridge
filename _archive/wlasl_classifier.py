"""
WLASL I3D word-level classifier.

Takes raw BGR webcam frames, buffers a 16-frame clip at 224x224,
normalizes to [-1, 1], and runs the pretrained I3D model.

Supports async inference via predict_async() for non-blocking usage.

Usage:
    clf = WLASLClassifier()
    clf.add_frame(bgr_frame)
    gloss, conf = clf.predict()
"""

import collections
import json
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .i3d_model import InceptionI3d
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.recognition.i3d_model import InceptionI3d

_CKPT_PATH   = os.path.join(os.path.dirname(__file__),
                             "..", "..", "models", "archived", "asl1000",
                             "FINAL_nslt_1000_iters=5104_top1=47.33_top5=76.44_top10=84.33.pt")
_LABELS_PATH = os.path.join(os.path.dirname(__file__),
                             "wlasl_weights", "class_list_1000.json")

CLIP_LEN    = 16      # frames per inference clip
FRAME_SIZE  = 224     # spatial resolution
STRIDE      = 8       # run inference every N frames
NUM_CLASSES = 1000


def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS does not support MaxPool3d / F.pad on 5D tensors used by I3D.
    return torch.device("cpu")


class WLASLClassifier:
    """
    Real-time WLASL sign classifier using pretrained I3D.

    Args:
        confidence_threshold: suppress results below this softmax probability.
        stride: run inference every N frames.
    """

    def __init__(self, ckpt_path=None, labels_path=None,
                 confidence_threshold=0.15, stride=STRIDE):
        self.confidence_threshold = confidence_threshold
        self.stride  = stride
        self.ready   = False
        self.device  = _best_device()

        ckpt_path   = ckpt_path   or _CKPT_PATH
        labels_path = labels_path or _LABELS_PATH

        self._labels   = self._load_labels(labels_path)
        self._model    = self._load_model(ckpt_path)

        if self._model is not None:
            self.ready = True
            print(f"[WLASLClassifier] I3D {NUM_CLASSES}-class loaded on {self.device}")
            print(f"[WLASLClassifier] clip={CLIP_LEN}f  stride={stride}f  "
                  f"threshold={confidence_threshold:.0%}")

        # Frame buffer: preprocessed tensors (C, H, W) float32
        self._buf       = collections.deque(maxlen=CLIP_LEN)
        self._call_n    = 0
        self._hist      = collections.deque(maxlen=5)
        self._raw       = (None, 0.0)
        self._last      = (None, 0.0)

        # Async inference state
        self._async_lock    = threading.Lock()
        self._async_running = False
        self._async_result  = (None, 0.0)
        self._async_top5    = []

    # ------------------------------------------------------------------

    def _load_labels(self, path):
        if not os.path.exists(path):
            print(f"[WLASLClassifier] labels not found: {path}")
            return [f"sign_{i}" for i in range(NUM_CLASSES)]
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data.get(str(i), f"sign_{i}") for i in range(NUM_CLASSES)]

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"[WLASLClassifier] checkpoint not found: {path}")
            return None
        try:
            model = InceptionI3d(num_classes=NUM_CLASSES)
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
            return model
        except Exception as exc:
            print(f"[WLASLClassifier] load failed: {exc}")
            return None

    def _preprocess_frame(self, bgr_frame):
        """BGR frame -> float32 tensor (3, 224, 224) normalized to [-1, 1]."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        t = rgb.astype(np.float32) / 128.0 - 1.0
        return t.transpose(2, 0, 1)

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def add_frame(self, bgr_frame):
        """Buffer one BGR webcam frame."""
        self._buf.append(self._preprocess_frame(bgr_frame))
        self._call_n += 1

    def predict(self):
        """
        Run inference synchronously (blocks).
        Returns (gloss: str|None, confidence: float).
        """
        if not self.ready or len(self._buf) < CLIP_LEN:
            return self._last

        if self._call_n % self.stride != 0:
            return self._last

        clip = np.stack(list(self._buf), axis=1)
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            conf = conf.item()
            idx  = idx.item()

        gloss = self._labels[idx] if idx < len(self._labels) else f"sign_{idx}"
        self._raw = (gloss, conf)
        self._hist.append((gloss, conf))

        if conf < self.confidence_threshold:
            self._last = (None, conf)
            return self._last

        # Confidence-weighted majority-vote smoothing
        if len(self._hist) >= 3:
            votes = collections.Counter()
            for g, c in self._hist:
                votes[g] += c
            winner = votes.most_common(1)[0][0]
            total_conf = sum(c for g, c in self._hist if g == winner)
            count = sum(1 for g, _ in self._hist if g == winner)
            self._last = (winner, total_conf / count)
        else:
            self._last = (gloss, conf)

        return self._last

    def predict_top5(self):
        """Return top-5 (gloss, confidence) pairs from last buffer state."""
        if not self.ready or len(self._buf) < CLIP_LEN:
            return []

        clip = np.stack(list(self._buf), axis=1)
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)
            top5_conf, top5_idx = probs.topk(5, dim=1)

        results = []
        for i in range(5):
            idx = top5_idx[0, i].item()
            conf = top5_conf[0, i].item()
            gloss = self._labels[idx] if idx < len(self._labels) else f"sign_{idx}"
            results.append((gloss, conf))
        return results

    # ------------------------------------------------------------------
    # Async API (non-blocking inference in background thread)
    # ------------------------------------------------------------------

    def maybe_run_async(self):
        """
        Trigger background inference if conditions are met.
        Call this every frame; it only launches a thread on stride boundaries.
        Returns True if inference was launched.
        """
        if not self.ready or len(self._buf) < CLIP_LEN:
            return False
        if self._call_n % self.stride != 0:
            return False
        if self._async_running:
            return False

        # Snapshot buffer (copy to avoid race)
        clip = np.stack(list(self._buf), axis=1).copy()
        self._async_running = True
        threading.Thread(target=self._infer_bg, args=(clip,), daemon=True).start()
        return True

    def _infer_bg(self, clip):
        """Run inference in background thread."""
        try:
            tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=1)

                # Top-1
                conf, idx = probs.max(dim=1)
                conf = conf.item()
                idx = idx.item()
                gloss = self._labels[idx] if idx < len(self._labels) else f"sign_{idx}"

                # Top-5
                top5_conf, top5_idx = probs.topk(5, dim=1)
                top5 = []
                for i in range(5):
                    t_idx = top5_idx[0, i].item()
                    t_conf = top5_conf[0, i].item()
                    t_gloss = self._labels[t_idx] if t_idx < len(self._labels) else f"sign_{t_idx}"
                    top5.append((t_gloss, t_conf))

            self._raw = (gloss, conf)
            self._hist.append((gloss, conf))

            # Compute smoothed result
            if conf < self.confidence_threshold:
                result = (None, conf)
            elif len(self._hist) >= 3:
                votes = collections.Counter()
                for g, c in self._hist:
                    votes[g] += c
                winner = votes.most_common(1)[0][0]
                total_c = sum(c for g, c in self._hist if g == winner)
                count = sum(1 for g, _ in self._hist if g == winner)
                result = (winner, total_c / count)
            else:
                result = (gloss, conf)

            with self._async_lock:
                self._async_result = result
                self._async_top5 = top5

        finally:
            self._async_running = False

    def get_async_result(self):
        """Get the latest async inference result (non-blocking)."""
        with self._async_lock:
            return self._async_result, self._async_top5

    # ------------------------------------------------------------------

    def on_sign_boundary(self):
        """Call after emitting a sign to start fresh for the next one."""
        self._buf.clear()
        self._hist.clear()
        self._call_n = 0
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []

    @property
    def raw(self):
        return self._raw

    def reset(self):
        self._buf.clear()
        self._hist.clear()
        self._call_n = 0
        self._raw  = (None, 0.0)
        self._last = (None, 0.0)
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []
