"""
WLASL I3D word-level classifier.

Takes raw BGR webcam frames, buffers a 16-frame clip at 224x224,
normalizes to [-1, 1], and runs the pretrained I3D model.

Usage:
    clf = WLASLClassifier()
    clf.add_frame(bgr_frame)           # call every webcam frame
    gloss, conf = clf.predict()        # returns on stride boundary
"""

import collections
import json
import os

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
                             "..", "..", "models", "archived", "asl100",
                             "FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt")
_LABELS_PATH = os.path.join(os.path.dirname(__file__),
                             "wlasl_weights", "class_list_100.json")

CLIP_LEN    = 16      # frames per inference clip
FRAME_SIZE  = 224     # spatial resolution
STRIDE      = 8       # run inference every N frames
NUM_CLASSES = 100


def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WLASLClassifier:
    """
    Real-time WLASL sign classifier using pretrained I3D.

    Args:
        confidence_threshold: suppress results below this softmax probability.
        stride: run inference every N frames.
    """

    def __init__(self, ckpt_path=None, labels_path=None,
                 confidence_threshold=0.20, stride=STRIDE):
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
            # Handle wrapped checkpoints
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
        """BGR frame → float32 tensor (3, 224, 224) normalized to [-1, 1]."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        t = rgb.astype(np.float32) / 128.0 - 1.0   # [0,255] → [-1, 1]
        return t.transpose(2, 0, 1)                  # (3, H, W)

    # ------------------------------------------------------------------

    def add_frame(self, bgr_frame):
        """Buffer one BGR webcam frame."""
        self._buf.append(self._preprocess_frame(bgr_frame))
        self._call_n += 1

    def predict(self):
        """
        Run inference if buffer is full and we're on a stride boundary.
        Returns (gloss: str|None, confidence: float).
        """
        if not self.ready:
            return (None, 0.0)

        if len(self._buf) < CLIP_LEN:
            return self._last

        if self._call_n % self.stride != 0:
            return self._last

        # Stack to (1, 3, T, H, W)
        clip = np.stack(list(self._buf), axis=1)       # (3, T, H, W)
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)               # (1, 100)
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

        # Majority-vote smoothing
        if len(self._hist) >= 3:
            glosses = [g for g, _ in self._hist]
            winner  = max(set(glosses), key=glosses.count)
            avg_c   = sum(c for g, c in self._hist if g == winner) / glosses.count(winner)
            self._last = (winner, avg_c)
        else:
            self._last = (gloss, conf)

        return self._last

    def process_frame(self, bgr_frame):
        """Convenience: add_frame + predict."""
        self.add_frame(bgr_frame)
        return self.predict()

    @property
    def raw(self):
        return self._raw

    def reset(self):
        self._buf.clear()
        self._hist.clear()
        self._call_n = 0
        self._raw  = (None, 0.0)
        self._last = (None, 0.0)


if __name__ == "__main__":
    import sys
    import time

    print("=== WLASLClassifier live test ===")
    clf = WLASLClassifier(confidence_threshold=0.10)
    if not clf.ready:
        print("Model not loaded.")
        sys.exit(1)

    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open webcam {cam_idx}")
        sys.exit(1)

    print(f"Webcam {cam_idx}. Press 'q' quit, 'r' reset.\n")
    fps_times = collections.deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        clf.add_frame(frame)
        gloss, conf = clf.predict()
        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

        raw_g, raw_c = clf.raw
        print(f"\r  buf={len(clf._buf):2d}/16  raw={raw_g or '—':15s} {raw_c:.0%}  "
              f"out={gloss or '—':15s} {conf:.0%}  {fps:.0f}fps",
              end="", flush=True)

        h, w = frame.shape[:2]
        label = f"{raw_g} ({raw_c:.0%})" if raw_g else f"buf {len(clf._buf)}/16"
        color = (0, 220, 0) if raw_c > 0.4 else (0, 200, 255) if raw_c > 0.2 else (100, 100, 200)
        cv2.putText(frame, label, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"{fps:.0f} fps  buf:{len(clf._buf)}/16",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("WLASLClassifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clf.reset()
            print("\n  [reset]")

    print()
    cap.release()
    cv2.destroyAllWindows()
