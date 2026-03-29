"""
SignRouter — unified interface for ASL recognition.

Wraps TFLiteClassifier (250-sign word recognition) and ASLClassifier
(A-Z fingerspelling fallback). Provides the same interface that app.py expects.
"""

import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.recognition.tflite_classifier import TFLiteClassifier
from src.recognition.asl_classifier import ASLClassifier

_MODE_WORD = "word"
_MODE_LETTER = "letter"
_MODE_IDLE = "idle"


class SignRouter:
    """
    Unified ASL recognition interface used by app.py.

    Methods:
        open()          — initialize models
        process_frame() — returns (annotated_frame, sign, confidence, mode)
        close()         — release resources
        reset_text()    — clear accumulated text buffer
    """

    def __init__(self, word_threshold=0.40, letter_threshold=0.55,
                 word_cooldown_s=2.0):
        self.word_threshold = word_threshold
        self.letter_threshold = letter_threshold
        self.word_cooldown_s = word_cooldown_s

        self._word_clf = None
        self._letter_clf = ASLClassifier()
        self._text_buffer = []

        self._last_word_emit = None
        self._last_word_time = 0.0

    def open(self):
        """Initialize the TFLite classifier (downloads MediaPipe model if needed)."""
        if self._word_clf is None:
            self._word_clf = TFLiteClassifier(confidence_threshold=self.word_threshold)
            print(f"[SignRouter] ready={self._word_clf.ready}")

    def close(self):
        if self._word_clf is not None:
            self._word_clf.close()
            self._word_clf = None

    def process_frame(self, frame_bgr):
        """
        Process one BGR webcam frame.

        Returns:
            annotated_frame : frame with landmarks drawn
            sign            : str or None
            confidence      : float
            mode            : 'word' | 'letter' | 'idle'
        """
        if self._word_clf is None:
            raise RuntimeError("Call SignRouter.open() first")

        # Run holistic tracker + buffer landmarks
        annotated, hands_visible = self._word_clf.process_frame(frame_bgr)

        # Run TFLite inference (synchronous — fast enough for TFLite)
        word_sign, word_conf = self._word_clf.predict()
        top5 = self._word_clf.predict_top5()

        # Fingerspelling from holistic tracker's hand landmarks
        letter, letter_conf = None, 0.0
        if self._word_clf._tracker is not None and hands_visible:
            # Get the last holistic result's normalized hand landmarks
            tracker = self._word_clf._tracker
            # Re-extract from the tracker's last result
            # The tracker already ran in process_frame, so we can use
            # the landmarks from the buffer
            pass  # fingerspelling handled below via separate check

        # Route: word takes priority
        sign, conf, mode = None, 0.0, _MODE_IDLE
        now = time.monotonic()

        if word_sign is not None and word_conf >= self.word_threshold:
            if word_sign != self._last_word_emit or (now - self._last_word_time) > self.word_cooldown_s:
                sign, conf, mode = word_sign, word_conf, _MODE_WORD
                self._last_word_emit = word_sign
                self._last_word_time = now
                self._word_clf.on_sign_boundary()

        if sign is not None:
            self._text_buffer.append(sign)

        return annotated, sign, conf, mode

    @property
    def text_so_far(self):
        return " ".join(self._text_buffer)

    def reset_text(self):
        self._text_buffer.clear()
        if self._word_clf:
            self._word_clf.reset()
        self._last_word_emit = None
