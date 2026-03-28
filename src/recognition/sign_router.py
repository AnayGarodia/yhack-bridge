"""
SignRouter — unified state machine for real-time ASL recognition.

Combines:
  • WordClassifier  (TFLite, 250 ASL signs from Kaggle model)  — PRIMARY
  • ASLClassifier   (rule-based fingerspelling A-Z)             — FALLBACK

Routing logic:
  - If WordClassifier confidence ≥ word_threshold  → use word result
  - Else if ASLClassifier confidence ≥ letter_threshold → use letter result
  - Else → return None

Usage:
    router = SignRouter()
    router.open()
    frame, sign, conf, mode = router.process_frame(bgr_frame)
    router.close()
"""

import collections
import os
import sys
import time

import cv2
import numpy as np

# Allow running as __main__ directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.recognition.holistic_tracker  import HolisticTracker
from src.recognition.wlasl_classifier  import WLASLClassifier
from src.recognition.asl_classifier    import ASLClassifier

# Mode labels for overlay
_MODE_WORD   = "word"
_MODE_LETTER = "letter"
_MODE_IDLE   = "idle"


class SignRouter:
    """
    Args:
        word_threshold:   minimum WordClassifier confidence to accept a word.
        letter_threshold: minimum ASLClassifier confidence to accept a letter.
        word_cooldown_s:  seconds to wait before emitting the same word again.
    """

    def __init__(self, word_threshold=0.10, letter_threshold=0.55,
                 word_cooldown_s=1.5):
        self.word_threshold   = word_threshold
        self.letter_threshold = letter_threshold
        self.word_cooldown_s  = word_cooldown_s

        self._tracker  = None  # HolisticTracker — opened in open()
        self._word_clf = WLASLClassifier(confidence_threshold=word_threshold)
        self._asl_clf  = ASLClassifier()

        self._last_word_time  = 0.0
        self._last_word_emit  = None
        self._text_buffer     = []          # accumulated recognized tokens
        self._last_sign_hist  = collections.deque(maxlen=8)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def open(self):
        """Initialize the HolisticTracker (downloads model if needed)."""
        if self._tracker is None:
            self._tracker = HolisticTracker()

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    # -----------------------------------------------------------------------
    # Per-frame processing
    # -----------------------------------------------------------------------

    def process_frame(self, frame_bgr):
        """
        Process one BGR webcam frame.

        Returns:
            annotated_frame : frame with landmarks + overlay drawn
            sign            : str or None — the recognized sign/letter
            confidence      : float
            mode            : 'word' | 'letter' | 'idle'
        """
        if self._tracker is None:
            raise RuntimeError("Call SignRouter.open() before process_frame()")

        annotated, lm543, lhand_px, rhand_px, norm_left, norm_right = \
            self._tracker.process_frame(frame_bgr)

        # ---- Word-level prediction (I3D takes raw video frames) ------------
        self._word_clf.add_frame(frame_bgr)
        word, word_conf = self._word_clf.predict()

        # ---- Fingerspelling prediction --------------------------------------
        # Prefer right hand; fall back to left
        letter, letter_conf = None, 0.0
        active_hand_norm = norm_right or norm_left
        if active_hand_norm is not None:
            letter, letter_conf = self._asl_clf.classify(active_hand_norm)

        # ---- Routing -------------------------------------------------------
        sign, conf, mode = None, 0.0, _MODE_IDLE

        if word is not None and word_conf >= self.word_threshold:
            now = time.monotonic()
            # Debounce: don't re-emit the same word within cooldown period
            if not (word == self._last_word_emit and
                    now - self._last_word_time < self.word_cooldown_s):
                sign, conf, mode = word, word_conf, _MODE_WORD
                self._last_word_emit = word
                self._last_word_time = now

        elif letter is not None and letter_conf >= self.letter_threshold:
            sign, conf, mode = letter, letter_conf, _MODE_LETTER

        if sign is not None:
            self._last_sign_hist.append(sign)
            self._text_buffer.append(sign)

        # ---- Overlay -------------------------------------------------------
        self._draw_overlay(annotated, sign, conf, mode, word, word_conf,
                           letter, letter_conf)

        return annotated, sign, conf, mode

    @property
    def text_so_far(self):
        """All recognized signs joined as a sentence."""
        return " ".join(self._text_buffer)

    def reset_text(self):
        self._text_buffer.clear()

    # -----------------------------------------------------------------------
    # Drawing helpers
    # -----------------------------------------------------------------------

    def _draw_overlay(self, frame, sign, conf, mode,
                      word, word_conf, letter, letter_conf):
        h, w = frame.shape[:2]

        # Primary result (large)
        if sign:
            color = (0, 220, 0) if mode == _MODE_WORD else (0, 200, 255)
            tag   = f"[{mode}]  {sign}  ({conf:.0%})"
            cv2.putText(frame, tag, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2, cv2.LINE_AA)

        # Secondary info (small)
        def small(text, y, color=(180, 180, 180)):
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)

        w_label = f"word:   {word or '—'}  ({word_conf:.0%})"
        l_label = f"letter: {letter or '—'}  ({letter_conf:.0%})"
        small(w_label, h - 50)
        small(l_label, h - 28)

        # Text buffer strip at top
        buf_text = " ".join(self._text_buffer[-10:])  # last 10 tokens
        if buf_text:
            cv2.putText(frame, buf_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 100), 1, cv2.LINE_AA)


if __name__ == "__main__":
    # Keys: w=word mode  f=fingerspell mode  a=auto  c=clear  q=quit
    FORCE_WORD   = "word"
    FORCE_LETTER = "letter"
    FORCE_AUTO   = "auto"

    router = SignRouter(word_threshold=0.10, letter_threshold=0.45)
    router.open()

    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Error: could not open webcam (index {cam_idx})")
        raise SystemExit(1)

    force_mode = FORCE_WORD   # default: word mode
    fps_times  = collections.deque(maxlen=30)
    frame_n    = 0

    print("SignRouter running")
    print("  w = word mode (default)   f = fingerspell   a = auto   c = clear   q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        # Downscale for holistic inference to improve FPS
        small = cv2.resize(frame, (640, 360)) if frame.shape[1] > 640 else frame

        t0 = time.perf_counter()
        annotated, sign, conf, mode = router.process_frame(small)
        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

        # Raw predictions (always available for display)
        word_raw, word_raw_conf = router._word_clf.raw

        # Determine what to display based on forced mode
        # Only show word if confidence is above a minimum display threshold
        _WORD_DISPLAY_MIN = 0.08
        if force_mode == FORCE_WORD:
            display_sign = word_raw if word_raw_conf >= _WORD_DISPLAY_MIN else None
            display_conf = word_raw_conf
            display_mode = "WORD"
        elif force_mode == FORCE_LETTER:
            display_sign = sign if mode == _MODE_LETTER else None
            display_conf = conf if mode == _MODE_LETTER else 0.0
            display_mode = "LETTER"
        else:  # auto
            display_sign, display_conf, display_mode = sign, conf, mode.upper()

        # ---- Terminal debug (every frame) -----------------------------------
        w_str = f"{word_raw}({word_raw_conf:.0%})" if word_raw else f"—({word_raw_conf:.0%})"
        l_str = f"{sign}({conf:.0%})" if mode == _MODE_LETTER and sign else "—"
        print(f"\r  word:{w_str:<22} letter:{l_str:<8} mode:{force_mode:<8} fps:{fps:.0f}  ",
              end="", flush=True)

        # ---- Video overlay --------------------------------------------------
        h, w_px = annotated.shape[:2]

        def put(text, y, scale=0.6, color=(200, 200, 200), thick=1):
            cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)

        # Mode indicator top-right
        mode_color = {"WORD": (0, 220, 0), "LETTER": (0, 200, 255), "auto": (180, 180, 0)}
        cv2.putText(annotated, f"[{force_mode.upper()}]",
                    (w_px - 130, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, mode_color.get(force_mode.upper(), (180, 180, 0)), 2, cv2.LINE_AA)

        # Always show raw word prediction
        w_color = (0, 220, 0) if word_raw_conf > 0.5 else \
                  (0, 200, 255) if word_raw_conf > 0.25 else (100, 100, 200)
        put(f"WORD:   {word_raw or '—'}  {word_raw_conf:.0%}", h - 55,
            scale=0.7, color=w_color, thick=2)

        # Always show raw letter prediction
        letter_str = f"{sign}  {conf:.0%}" if mode == _MODE_LETTER and sign else "—"
        put(f"LETTER: {letter_str}", h - 28, scale=0.65)

        # Primary display
        if display_sign:
            color = (0, 255, 80) if "WORD" in display_mode else (0, 200, 255)
            put(f"{display_sign}  ({display_conf:.0%})", 55,
                scale=1.4, color=color, thick=3)

        # Text buffer
        if router._text_buffer:
            put(" ".join(router._text_buffer[-8:]), 90, scale=0.7, color=(255, 255, 80))

        # FPS
        put(f"{fps:.0f} fps", 25, scale=0.55, color=(100, 100, 100))

        # Only log EMIT when sign changes and confidence is meaningful
        if display_sign and display_conf >= 0.15:
            print(f"\n  >>> EMIT [{display_mode}] {display_sign} ({display_conf:.0%})")

        cv2.imshow("Bridge — ASL Router", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            force_mode = FORCE_WORD
            print(f"\n  [mode → word]")
        elif key == ord("f"):
            force_mode = FORCE_LETTER
            print(f"\n  [mode → fingerspell]")
        elif key == ord("a"):
            force_mode = FORCE_AUTO
            print(f"\n  [mode → auto]")
        elif key == ord("c"):
            router.reset_text()
            print("\n  [buffer cleared]")

    print()
    cap.release()
    cv2.destroyAllWindows()
    router.close()
