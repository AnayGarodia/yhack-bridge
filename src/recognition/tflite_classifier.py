"""
TFLite-based ASL word classifier using hoyso48's 1st-place Kaggle model.

The TFLite model includes all preprocessing internally:
  - Selects 118 key landmarks (lips + hands + nose + eyes)
  - Centers on landmark 17 (nose), normalizes by NaN-aware std
  - Computes position (x,y), velocity, acceleration → 708 features
  - Runs Conv1D + Transformer blocks → 250-class output

This wrapper just feeds raw (T, 543, 3) MediaPipe landmarks.

Usage:
    clf = TFLiteClassifier()
    annotated, hands_visible = clf.process_frame(bgr_frame)
    sign, conf = clf.predict()
    top5 = clf.predict_top5()
"""

import collections
import json
import os
import threading

import cv2
import numpy as np

try:
    from .holistic_tracker import HolisticTracker
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.recognition.holistic_tracker import HolisticTracker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.tflite")
_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "sign_to_prediction_index_map.json")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MIN_FRAMES = 10       # minimum frames before attempting inference
MAX_FRAMES = 384      # model's max sequence length
STRIDE = 15           # run inference every N buffered frames
NUM_CLASSES = 250


def _load_tflite_interpreter(model_path):
    """Load TFLite interpreter, trying tflite-runtime first, then tensorflow."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path)
    except ImportError:
        pass
    raise ImportError(
        "Neither tflite-runtime nor tensorflow is installed. "
        "Install one: pip install tflite-runtime"
    )


class TFLiteClassifier:
    """
    Real-time ASL sign classifier wrapping hoyso48's 1st-place TFLite model.

    Same interface as WLASLClassifier for drop-in use in main.py.
    """

    def __init__(self, model_path=None, labels_path=None,
                 confidence_threshold=0.40):
        self.confidence_threshold = confidence_threshold
        self.ready = False
        self.device = "tflite"

        model_path = model_path or _MODEL_PATH
        labels_path = labels_path or _LABELS_PATH

        # Load labels: sign_to_prediction_index_map → index_to_sign
        self._idx_to_sign = {}
        self._load_labels(labels_path)

        # Load TFLite model
        self._prediction_fn = None
        self._load_model(model_path)

        # HolisticTracker (lazy-loaded on first process_frame call)
        self._tracker = None

        # Frame buffer: list of (543, 3) float32 arrays
        self._buf = collections.deque(maxlen=MAX_FRAMES)
        self._call_n = 0
        self._no_hand_frames = 0

        # Prediction state
        self._hist = collections.deque(maxlen=5)
        self._raw = (None, 0.0)
        self._last_result = (None, 0.0)
        self._last_top5 = []

        # Async inference state
        self._async_lock = threading.Lock()
        self._async_running = False
        self._async_result = (None, 0.0)
        self._async_top5 = []

    def _load_labels(self, path):
        if not os.path.exists(path):
            print(f"[TFLiteClassifier] labels not found: {path}")
            self._idx_to_sign = {i: f"sign_{i}" for i in range(NUM_CLASSES)}
            return
        with open(path) as f:
            sign_to_idx = json.load(f)
        self._idx_to_sign = {v: k for k, v in sign_to_idx.items()}
        print(f"[TFLiteClassifier] {len(self._idx_to_sign)} labels loaded")

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"[TFLiteClassifier] model not found: {path}")
            return
        try:
            interpreter = _load_tflite_interpreter(path)
            self._prediction_fn = interpreter.get_signature_runner("serving_default")
            self.ready = True
            print(f"[TFLiteClassifier] TFLite model loaded ({os.path.getsize(path) / 1e6:.1f} MB)")
            print(f"[TFLiteClassifier] threshold={self.confidence_threshold:.0%}, "
                  f"stride={STRIDE}f, min_frames={MIN_FRAMES}")
        except Exception as exc:
            print(f"[TFLiteClassifier] load failed: {exc}")

    def _ensure_tracker(self):
        if self._tracker is None:
            self._tracker = HolisticTracker()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr):
        """
        Run holistic tracking on one BGR frame and buffer landmarks.

        Returns:
            annotated_frame: frame with landmarks drawn
            hands_visible: bool
        """
        self._ensure_tracker()
        annotated, lm543, lhand_px, rhand_px, norm_left, norm_right = \
            self._tracker.process_frame(frame_bgr)

        hands_visible = (lhand_px is not None) or (rhand_px is not None)

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
        else:
            self._no_hand_frames = 0
            self._buf.append(lm543.copy())
            self._call_n += 1

        return annotated, hands_visible

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, frames):
        """
        Run TFLite inference on a sequence of landmark frames.

        Args:
            frames: list of (543, 3) float32 arrays

        Returns:
            (sign, confidence), [(sign, conf), ...] top5
        """
        # Stack to (T, 543, 3) — the exact format the TFLite model expects
        xyz = np.stack(frames, axis=0).astype(np.float32)

        # Run the model (preprocessing is built into the TFLite graph)
        output = self._prediction_fn(inputs=xyz)
        logits = output["outputs"].reshape(-1)  # (250,)

        # Softmax
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted)
        probs /= probs.sum()

        # Top-1
        idx = int(probs.argmax())
        conf = float(probs[idx])
        sign = self._idx_to_sign.get(idx, f"sign_{idx}")

        # Top-5
        top5_idx = probs.argsort()[-5:][::-1]
        top5 = [(self._idx_to_sign.get(int(i), f"sign_{i}"), float(probs[i]))
                for i in top5_idx]

        self._raw = (sign, conf)
        self._hist.append((sign, conf))

        # Require 2 consecutive agreeing predictions above threshold
        result = (None, conf)
        if conf >= self.confidence_threshold and len(self._hist) >= 2:
            recent = [(g, c) for g, c in list(self._hist)[-2:]
                      if c >= self.confidence_threshold]
            if len(recent) >= 2 and recent[-1][0] == recent[-2][0]:
                result = (sign, conf)

        return result, top5

    def predict(self):
        """Synchronous predict. Returns (sign_or_None, confidence)."""
        if not self.ready or len(self._buf) < MIN_FRAMES:
            return (None, 0.0)
        if self._call_n % STRIDE != 0:
            return self._last_result

        result, top5 = self._run_inference(list(self._buf))
        self._last_result = result
        self._last_top5 = top5
        with self._async_lock:
            self._async_result = result
            self._async_top5 = top5
        return result

    def predict_top5(self):
        """Return top-5 from the last inference."""
        with self._async_lock:
            return list(self._async_top5)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    def maybe_run_async(self):
        """Trigger background inference if conditions are met."""
        if not self.ready or len(self._buf) < MIN_FRAMES:
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
            result, top5 = self._run_inference(frames)
            self._last_result = result
            self._last_top5 = top5
            with self._async_lock:
                self._async_result = result
                self._async_top5 = top5
        except Exception as exc:
            print(f"[TFLiteClassifier] async inference error: {exc}")
        finally:
            self._async_running = False

    def get_async_result(self):
        """Non-blocking: get latest async result."""
        with self._async_lock:
            return self._async_result, list(self._async_top5)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def on_sign_boundary(self):
        """Call after emitting a sign to reset for the next one."""
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
        self._last_result = (None, 0.0)
        self._last_top5 = []
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
