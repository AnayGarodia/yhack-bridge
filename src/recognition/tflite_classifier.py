import os as _os
_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

"""
TFLite-based ASL word classifier using hoyso48's 1st-place Kaggle model.

Sliding window architecture: classification runs continuously on overlapping
windows of frames. Deduplication logic ensures each sign is emitted once.

All heavy imports (tensorflow, mediapipe) are lazy-loaded on first use,
so importing this module is instant.
"""

import collections
import json
import os
import threading
import time

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
NUM_CLASSES = 250


class TFLiteClassifier:
    """
    Real-time ASL sign classifier wrapping hoyso48's 1st-place TFLite model.

    Uses a sliding window approach: landmarks are buffered continuously, and
    classification runs every STRIDE frames on overlapping windows.
    Deduplication (consecutive agreement + cooldown) ensures each sign is
    emitted only once.

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

        # Sliding window parameters (from .env with defaults)
        self.WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "30"))
        self.STRIDE = int(os.getenv("WINDOW_STRIDE", "5"))
        self.MAX_BUFFER = 60
        self.MIN_FRAMES = int(os.getenv("MIN_FRAMES", "8"))
        self.CONFIDENCE_THRESHOLD = float(
            os.getenv("CONFIDENCE_THRESHOLD", str(confidence_threshold))
        )
        self.SAME_WORD_COOLDOWN = float(os.getenv("SAME_WORD_COOLDOWN", "1.5"))
        self.DIFFERENT_WORD_COOLDOWN = 0.0
        self.CONSECUTIVE_AGREEMENT = int(os.getenv("CONSECUTIVE_AGREEMENT", "2"))

        # Rolling landmark buffer (list, trimmed to MAX_BUFFER)
        self._frame_buffer = []
        self._frames_since_last_classify = 0
        self._no_hand_frames = 0

        # Prediction / emission state
        self._prediction_history = []   # [(word, conf, timestamp)]
        self._last_emitted_word = None
        self._last_emitted_time = 0
        self._pending_result = None
        self._hist = collections.deque(maxlen=5)
        self._raw = (None, 0.0)
        self._last_result = (None, 0.0)
        self._last_top5 = []

        # Async inference
        self._async_lock = threading.Lock()
        self._async_running = False
        self._async_result = (None, 0.0)       # deduped (for main.py)
        self._async_top5 = []
        self._raw_async_result = (None, 0.0)   # raw (for sign_router.py)
        self._raw_async_top5 = []

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

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
        """
        Process a BGR frame: run MediaPipe, buffer landmarks.
        Returns (annotated_frame, hands_visible).
        """
        self._lazy_init()
        import cv2
        annotated, lm543, lhand_px, rhand_px, norm_left, norm_right = \
            self._tracker.process_frame(frame_bgr)

        hands_visible = (lhand_px is not None) or (rhand_px is not None)

        if not hands_visible:
            self._no_hand_frames += 1
            if self._no_hand_frames >= 10:
                # Hands gone for 10+ frames — full clear (not signing)
                self._frame_buffer = []
                self._prediction_history = []
                self._frames_since_last_classify = 0
                self._raw = (None, 0.0)
                self._no_hand_frames = 0
                with self._async_lock:
                    self._async_result = (None, 0.0)
                    self._async_top5 = []
                    self._raw_async_result = (None, 0.0)
                    self._raw_async_top5 = []
        else:
            self._no_hand_frames = 0
            # Add landmark frame to rolling buffer
            self._frame_buffer.append(lm543.copy())
            # Keep buffer bounded
            if len(self._frame_buffer) > self.MAX_BUFFER:
                self._frame_buffer = self._frame_buffer[-self.MAX_BUFFER:]
            self._frames_since_last_classify += 1

        return annotated, hands_visible

    # ------------------------------------------------------------------
    # Inference (UNCHANGED — do not modify)
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

    def _classify_window(self, frames):
        """
        Run TFLite inference on a window of landmark frames.
        Thin wrapper around _run_inference — does not change preprocessing.
        Returns: (top_word, top_confidence) or (None, 0.0).
        """
        try:
            result, top5 = self._run_inference(frames)
            self._last_top5 = top5
            with self._async_lock:
                self._async_top5 = list(top5)
            return result[0], result[1]
        except Exception as e:
            print(f"Inference error: {e}")
            return None, 0.0

    # ------------------------------------------------------------------
    # Sliding window emission
    # ------------------------------------------------------------------

    def _check_and_emit(self):
        """
        Emit a word only when:
        1. Last CONSECUTIVE_AGREEMENT windows all agree on the same word
        2. Sufficient cooldown has passed since last emission of this word
        """
        if len(self._prediction_history) < self.CONSECUTIVE_AGREEMENT:
            return

        # Get the last N predictions
        recent = self._prediction_history[-self.CONSECUTIVE_AGREEMENT:]
        recent_words = [w for w, c, t in recent]
        recent_confs = [c for w, c, t in recent]

        # All recent predictions must agree
        if len(set(recent_words)) != 1:
            return

        agreed_word = recent_words[0]
        avg_confidence = sum(recent_confs) / len(recent_confs)

        # Check cooldown
        now = time.time()
        if agreed_word == self._last_emitted_word:
            if now - self._last_emitted_time < self.SAME_WORD_COOLDOWN:
                return

        # All checks passed — emit the word
        self._last_emitted_word = agreed_word
        self._last_emitted_time = now
        self._prediction_history = []
        self._pending_result = (agreed_word, avg_confidence)

        print(f"[SlidingWindow] WORD: '{agreed_word}' (conf={avg_confidence:.2f})")

        with self._async_lock:
            self._async_result = (agreed_word, avg_confidence)

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def predict(self):
        """
        Returns (word, confidence) if a new word was recognized
        since the last call, otherwise (None, 0.0).
        Non-blocking — returns immediately.
        """
        if self._pending_result is not None:
            result = self._pending_result
            self._pending_result = None
            return result
        return (None, 0.0)

    def predict_top5(self):
        with self._async_lock:
            return list(self._async_top5)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    def maybe_run_async(self):
        """
        Launch background inference if enough frames have accumulated
        since the last classification.  Returns True if launched.
        """
        if not self.ready or len(self._frame_buffer) < self.MIN_FRAMES:
            return False
        if self._frames_since_last_classify < self.STRIDE:
            return False
        if self._async_running:
            return False

        self._frames_since_last_classify = 0
        frames = list(self._frame_buffer[-self.WINDOW_SIZE:])
        self._async_running = True
        threading.Thread(target=self._infer_bg, args=(frames,), daemon=True).start()
        return True

    def _infer_bg(self, frames):
        """Background inference thread with sliding window dedup."""
        try:
            result, top5 = self._run_inference(frames)
            self._last_result = result
            self._last_top5 = top5

            # Always update raw results (for sign_router.py)
            with self._async_lock:
                self._raw_async_result = result
                self._raw_async_top5 = list(top5)
                self._async_top5 = list(top5)

            top_word, top_conf = result

            if top_word is None or top_conf < self.CONFIDENCE_THRESHOLD:
                return

            # Add to prediction history with timestamp
            now = time.time()
            self._prediction_history.append((top_word, top_conf, now))

            # Keep only recent predictions (last 1 second)
            self._prediction_history = [
                (w, c, t) for w, c, t in self._prediction_history
                if now - t < 1.0
            ]

            # Check for consecutive agreement
            self._check_and_emit()

        except Exception as exc:
            print(f"[TFLiteClassifier] async error: {exc}")
        finally:
            self._async_running = False

    def get_async_result(self):
        """Deduped result — returns a word only when sliding window confirms it."""
        with self._async_lock:
            return self._async_result, list(self._async_top5)

    def get_raw_async_result(self):
        """Raw (unfiltered) result for callers with their own stability logic."""
        with self._async_lock:
            return self._raw_async_result, list(self._raw_async_top5)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def on_sign_boundary(self):
        """
        Called after a word is emitted to prepare for next sign.
        Keep last 8 frames as transition context rather than clearing.
        """
        self._frame_buffer = self._frame_buffer[-8:]
        self._prediction_history = []
        self._frames_since_last_classify = 0
        self._hist.clear()
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []

    @property
    def raw(self):
        return self._raw

    @property
    def buf_fill(self):
        return len(self._frame_buffer)

    def reset(self):
        """Full reset — clears everything."""
        self._frame_buffer = []
        self._prediction_history = []
        self._pending_result = None
        self._last_emitted_word = None
        self._last_emitted_time = 0
        self._frames_since_last_classify = 0
        self._no_hand_frames = 0
        self._hist.clear()
        self._raw = (None, 0.0)
        self._last_result = (None, 0.0)
        self._last_top5 = []
        with self._async_lock:
            self._async_result = (None, 0.0)
            self._async_top5 = []
            self._raw_async_result = (None, 0.0)
            self._raw_async_top5 = []

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
