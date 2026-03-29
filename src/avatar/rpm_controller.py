"""
RPM Avatar Controller — state machine driving the 3D avatar.

States:
  IDLE → word arrives → SIGNING
  SIGNING (0.6s = 18 frames) → queue empty → IDLE
  SIGNING → queue has word → TRANSITIONING
  TRANSITIONING (0.2s = 6 frames) → SIGNING (next word)
"""

import collections
import logging
import threading
import time

import cv2
import numpy as np

from .sign_library import SignLibrary
from .rpm_renderer import RPMRenderer
from .animation_engine import AnimationEngine

logger = logging.getLogger(__name__)

_MAX_QUEUE = 3
_SIGN_DURATION_S = 0.6       # 18 frames at 30fps
_TRANSITION_DURATION_S = 0.2  # 6 frames at 30fps


class RPMAvatarController:
    """Drives the RPM avatar with queued ASL signs."""

    def __init__(self, renderer: RPMRenderer, library: SignLibrary,
                 engine: AnimationEngine):
        self._renderer = renderer
        self._library = library
        self._engine = engine

        self._lock = threading.Lock()
        self._queue: collections.deque[tuple[str, np.ndarray]] = collections.deque()

        # State
        self._state = "idle"  # idle | signing | transitioning
        self._current_word = ""
        self._current_frames: np.ndarray | None = None   # (30, 2, 21, 3)
        self._next_frames: np.ndarray | None = None
        self._next_word = ""
        self._state_start = time.monotonic()
        self._last_frame: np.ndarray | None = None

    def queue_word(self, word: str):
        """Add word to sign queue. Max 3, drops oldest if over."""
        with self._lock:
            frames = self._library.get(word)
            if frames is not None:
                # Dedup
                if self._queue and self._queue[-1][0].lower() == word.lower():
                    return
                self._queue.append((word, frames))
                while len(self._queue) > _MAX_QUEUE:
                    dropped = self._queue.popleft()
                    logger.debug("Dropped: %s", dropped[0])
            else:
                # Fingerspell short words
                letters = self._library.fingerspell(word)
                for letter in letters:
                    lf = self._library.get(letter)
                    if lf is not None:
                        self._queue.append((letter, lf))

    def get_frame(self) -> np.ndarray:
        """Return current BGR frame. Called at 30fps. Never blocks >33ms."""
        with self._lock:
            try:
                frame = self._advance()
                self._last_frame = frame
                return frame
            except Exception as e:
                logger.error("Render error: %s", e)
                if self._last_frame is not None:
                    return self._last_frame
                return np.zeros((720, 1280, 3), dtype=np.uint8)

    def reset(self):
        with self._lock:
            self._queue.clear()
            self._state = "idle"
            self._current_frames = None

    @property
    def current_word(self) -> str:
        with self._lock:
            return self._current_word if self._state != "idle" else ""

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _advance(self) -> np.ndarray:
        now = time.monotonic()
        elapsed = now - self._state_start

        if self._state == "idle":
            return self._handle_idle(now)
        elif self._state == "signing":
            return self._handle_signing(elapsed, now)
        elif self._state == "transitioning":
            return self._handle_transitioning(elapsed, now)
        return self._render_idle()

    def _handle_idle(self, now: float) -> np.ndarray:
        if not self._queue:
            return self._render_idle()

        self._current_word, self._current_frames = self._queue.popleft()
        self._state = "signing"
        self._state_start = now
        logger.debug("SIGNING: %s", self._current_word)
        return self._render_signing(0.0)

    def _handle_signing(self, elapsed: float, now: float) -> np.ndarray:
        t = min(1.0, elapsed / _SIGN_DURATION_S)
        frame = self._render_signing(t)

        if t >= 1.0:
            if self._queue:
                # Transition to next sign
                self._next_word, self._next_frames = self._queue.popleft()
                self._state = "transitioning"
                self._state_start = now
                logger.debug("TRANSITIONING: %s → %s", self._current_word, self._next_word)
            else:
                self._state = "idle"

        return frame

    def _handle_transitioning(self, elapsed: float, now: float) -> np.ndarray:
        t = min(1.0, elapsed / _TRANSITION_DURATION_S)
        frame = self._render_transition(t)

        if t >= 1.0:
            self._current_word = self._next_word
            self._current_frames = self._next_frames
            self._next_frames = None
            self._state = "signing"
            self._state_start = now
            logger.debug("SIGNING: %s", self._current_word)

        return frame

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_idle(self) -> np.ndarray:
        t = time.time()
        right_hand, left_hand = self._engine.idle_pose(t)
        self._renderer.set_pose(right_hand, left_hand)
        frame = self._renderer.render()
        self._add_overlay(frame, "Listening...", "")
        return frame

    def _render_signing(self, t: float) -> np.ndarray:
        if self._current_frames is None:
            return self._render_idle()

        right_hand, left_hand = self._engine.interpolate_sign(self._current_frames, t)
        self._renderer.set_pose(right_hand, left_hand)
        frame = self._renderer.render()

        next_word = self._queue[0][0] if self._queue else ""
        self._add_overlay(frame, self._current_word, next_word)
        return frame

    def _render_transition(self, t: float) -> np.ndarray:
        if self._current_frames is None or self._next_frames is None:
            return self._render_idle()

        right_hand, left_hand = self._engine.blend_signs(
            self._current_frames, self._next_frames, t)
        self._renderer.set_pose(right_hand, left_hand)
        frame = self._renderer.render()

        self._add_overlay(frame, f"{self._current_word} → {self._next_word}", "")
        return frame

    def _add_overlay(self, frame: np.ndarray, current: str, next_word: str):
        """Add text overlay to rendered frame."""
        h, w = frame.shape[:2]

        if current:
            # Semi-transparent bar
            bar_h = 50
            overlay = frame[h - bar_h:h, :].copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame[h - bar_h:h, :], 0.4, 0,
                            frame[h - bar_h:h, :])

            # Current word
            (tw, th), _ = cv2.getTextSize(current, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, current, ((w - tw) // 2, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Next word hint
        if next_word:
            cv2.putText(frame, f"next: {next_word}", (w - 180, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # Queue depth
        qlen = len(self._queue)
        if qlen > 0:
            color = (0, 200, 255) if qlen > 3 else (120, 120, 120)
            label = f"queue: {qlen}"
            if qlen > 3:
                label += " (fast)"
            cv2.putText(frame, label, (w - 120, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
