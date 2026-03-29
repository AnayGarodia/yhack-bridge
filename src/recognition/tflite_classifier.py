import os as _os
_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

"""
TFLite-based ASL word classifier using hoyso48's 1st-place Kaggle model.

All heavy imports (tensorflow, mediapipe) are lazy-loaded on first use,
so importing this module is instant.
"""

import collections
import json
import os
import threading

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.tflite")
_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "sign_to_prediction_index_map.json")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MIN_FRAMES = 10
MAX_FRAMES = 384
STRIDE = 15
NUM_CLASSES = 250


class TFLiteClassifier:
    """
    Real-time ASL sign classifier wrapping hoyso48's 1st-place TFLite model.
    All heavy loading is deferred until first use (process_frame or predict).
    """

    def __init__(self, model_path=None, labels_path=None,
                 confidence_threshold=0.35):
        self.confidence_threshold = confidence_threshold
        self.ready = False
        self.device = "tflite"

        self._model_path = model_path or _MODEL_PATH
        self._labels_path = labels_path or _LABELS_PATH

        # These are lazy-loaded
        self._idx_to_sign = None
        self._prediction_fn = None
        self._tracker = None
        self._initialized = False

        # Frame buffer
        self._buf = collections.deque(maxlen=MAX_FRAMES)
        self._call_n = 0
        self._no_hand_frames = 0

        # Prediction state
        self._hist = collections.deque(maxlen=5)
        self._raw = (None, 0.0)
        self._last_result = (None, 0.0)
        self._last_top5 = []

        # Async inference
        self._async_lock = threading.Lock()
        self._async_running = False
        self._async_result = (None, 0.0)
        self._async_top5 = []

    def _lazy_init(self):
        """Load labels, TFLite model, and tracker on first use."""
        if self._initialized:
            return
        self._initialized = True

        # Load labels
        if os.path.exists(self._labels_path):
            with open(self._labels_path) as f:
                sign_to_idx = json.load(f)
            self._idx_to_sign = {v: k for k, v in sign_to_idx.items()}
            print(f"[TFLiteClassifier] {len(self._idx_to_sign)} labels loaded")
        else:
            self._idx_to_sign = {i: f"sign_{i}" for i in range(NUM_CLASSES)}
            print(f"[TFLiteClassifier] labels not found: {self._labels_path}")

        # Load TFLite model
        if os.path.exists(self._model_path):
            try:
                try:
                    import tflite_runtime.interpreter as tflite
                    interpreter = tflite.Interpreter(self._model_path)
                except ImportError:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(self._model_path)

                self._prediction_fn = interpreter.get_signature_runner("serving_default")
                self.ready = True
                print(f"[TFLiteClassifier] model loaded ({os.path.getsize(self._model_path) / 1e6:.1f} MB)")
            except Exception as exc:
                print(f"[TFLiteClassifier] load failed: {exc}")
        else:
            print(f"[TFLiteClassifier] model not found: {self._model_path}")

        # Load tracker
        from src.recognition.holistic_tracker import HolisticTracker
        self._tracker = HolisticTracker()
        print("[TFLiteClassifier] holistic tracker ready")

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr):
        self._lazy_init()
        import cv2
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
        xyz = np.stack(frames, axis=0).astype(np.float32)
        output = self._prediction_fn(inputs=xyz)
        logits = output["outputs"].reshape(-1)

        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted)
        probs /= probs.sum()

        idx = int(probs.argmax())
        conf = float(probs[idx])
        sign = self._idx_to_sign.get(idx, f"sign_{idx}")

        top5_idx = probs.argsort()[-5:][::-1]
        top5 = [(self._idx_to_sign.get(int(i), f"sign_{i}"), float(probs[i]))
                for i in top5_idx]

        self._raw = (sign, conf)
        self._hist.append((sign, conf))

        if conf >= self.confidence_threshold:
            result = (sign, conf)
        else:
            result = (None, conf)

        return result, top5

    def predict(self):
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
        with self._async_lock:
            return list(self._async_top5)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    def maybe_run_async(self):
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
            print(f"[TFLiteClassifier] async error: {exc}")
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
        self._last_result = (None, 0.0)
        self._last_top5 = []
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
