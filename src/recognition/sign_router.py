"""
SignRouter — unified interface for ASL recognition.

Works like a human interpreter:
  1. While signer is signing (hands visible) → show live prediction on UI
  2. When signer finishes (hands drop) → commit the best word → feed to pipeline
  3. Wait for next sign

This prevents the model from spamming partial/wrong predictions mid-gesture.
"""

import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Lazy imports — these pull in tensorflow/mediapipe which are slow
TFLiteClassifier = None
ASLClassifier = None

def _ensure_imports():
    global TFLiteClassifier, ASLClassifier
    if TFLiteClassifier is None:
        from src.recognition.tflite_classifier import TFLiteClassifier as _T
        from src.recognition.asl_classifier import ASLClassifier as _A
        TFLiteClassifier = _T
        ASLClassifier = _A

_MODE_WORD = "word"
_MODE_LETTER = "letter"
_MODE_IDLE = "idle"

# How many no-hand frames before we consider the sign "done"
HAND_DROP_FRAMES = 8    # ~0.27s at 30fps — pause between signs
MIN_SIGN_FRAMES = 15    # minimum frames of hands visible to count as a sign


class SignRouter:
    def __init__(self, word_threshold=0.25, letter_threshold=0.55,
                 word_cooldown_s=1.5):
        self.word_threshold = word_threshold
        self.letter_threshold = letter_threshold
        self.word_cooldown_s = word_cooldown_s

        self._word_clf = None
        self._letter_clf = None
        self._text_buffer = []

        # State tracking
        self._hands_visible_count = 0    # consecutive frames with hands
        self._no_hand_count = 0          # consecutive frames without hands
        self._signing = False            # currently in a sign gesture
        self._sign_predictions = []      # predictions accumulated during this sign

        # Live display state (updated every inference, shown on UI)
        self._live_sign = None
        self._live_conf = 0.0
        self._live_top5 = []

        # Committed state (only set when a sign is finalized)
        self._last_committed = None
        self._last_commit_time = 0.0

    def open(self):
        _ensure_imports()
        if self._word_clf is None:
            self._word_clf = TFLiteClassifier(confidence_threshold=self.word_threshold)
            self._letter_clf = ASLClassifier()
            print(f"[SignRouter] ready={self._word_clf.ready}")

    def close(self):
        if self._word_clf is not None:
            self._word_clf.close()
            self._word_clf = None

    def process_frame(self, frame_bgr):
        """
        Returns:
            annotated_frame, sign_or_None, confidence, mode

        sign is only non-None when a completed sign is committed
        (hands dropped after signing). The UI should also use
        get_live_display() for real-time feedback.
        """
        if self._word_clf is None:
            raise RuntimeError("Call SignRouter.open() first")

        # 1. Run holistic tracker + buffer landmarks
        annotated, hands_visible = self._word_clf.process_frame(frame_bgr)

        # 2. Track hand presence state
        if hands_visible:
            self._hands_visible_count += 1
            self._no_hand_count = 0

            # Start signing if hands have been up long enough
            if self._hands_visible_count >= 5 and not self._signing:
                self._signing = True
                self._sign_predictions.clear()

        else:
            self._no_hand_count += 1
            self._hands_visible_count = 0

        # 3. Run inference for live display (async, non-blocking)
        self._word_clf.maybe_run_async()
        (raw_sign, raw_conf), top5 = self._word_clf.get_async_result()

        # Update live display
        if raw_sign is not None:
            self._live_sign = raw_sign
            self._live_conf = raw_conf
            self._live_top5 = top5

            # Accumulate predictions during signing
            if self._signing:
                self._sign_predictions.append((raw_sign, raw_conf))

        # 4. Check for sign completion (hands dropped after signing)
        committed_sign, committed_conf, committed_mode = None, 0.0, _MODE_IDLE

        if (self._signing and
                self._no_hand_count >= HAND_DROP_FRAMES and
                len(self._sign_predictions) >= 1):

            # Sign is done — pick the best prediction
            best = self._pick_best_prediction()
            now = time.monotonic()

            if best is not None:
                sign_name, sign_conf = best
                # Apply cooldown
                if (sign_name != self._last_committed or
                        (now - self._last_commit_time) > self.word_cooldown_s):
                    committed_sign = sign_name
                    committed_conf = sign_conf
                    committed_mode = _MODE_WORD
                    self._last_committed = sign_name
                    self._last_commit_time = now
                    self._text_buffer.append(sign_name)
                    print(f"[SignRouter] COMMIT: {sign_name} ({sign_conf:.0%})")

            # Reset for next sign
            self._signing = False
            self._sign_predictions.clear()
            self._word_clf.on_sign_boundary()
            self._live_sign = None
            self._live_conf = 0.0

        return annotated, committed_sign, committed_conf, committed_mode

    def _pick_best_prediction(self):
        """Pick the most confident prediction from the signing period."""
        if not self._sign_predictions:
            return None

        # Weighted vote: accumulate confidence per sign
        votes = collections.Counter()
        for sign, conf in self._sign_predictions:
            if conf >= self.word_threshold:
                votes[sign] += conf

        if not votes:
            return None

        best_sign = votes.most_common(1)[0][0]
        # Get max confidence for the winning sign
        best_conf = max(c for s, c in self._sign_predictions if s == best_sign)
        return best_sign, best_conf

    def get_live_display(self):
        """
        Get the current live prediction for UI display.
        Returns (sign, confidence, top5) — updated continuously during signing.
        """
        return self._live_sign, self._live_conf, self._live_top5

    @property
    def text_so_far(self):
        return " ".join(self._text_buffer)

    def reset_text(self):
        self._text_buffer.clear()
        if self._word_clf:
            self._word_clf.reset()
        self._last_committed = None
        self._signing = False
        self._sign_predictions.clear()
        self._live_sign = None
        self._live_conf = 0.0
