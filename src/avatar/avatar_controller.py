"""
AvatarController — queues ASL glosses and plays back landmark animations.

State machine:
  idle → signing → pause → idle (checks queue again)
  idle → fingerspelling → pause → idle

Thread-safe: enqueue() from SocketIO thread, get_frame() from MJPEG generator.
"""

import collections
import logging
import threading
import time

import numpy as np

from .sign_database import SignDatabase
from .avatar_renderer import AvatarRenderer

logger = logging.getLogger(__name__)

_MAX_QUEUE = 2  # Keep queue tiny — real-time means current, not queued
_MAX_SIGN_FRAMES = 18  # 18 frames at 30fps = 0.6 seconds max per sign


class AvatarController:
    """Queues ASL glosses and returns animated avatar frames."""

    def __init__(self, database: SignDatabase, renderer: AvatarRenderer,
                 sign_fps: int = 30, pause_ms: int = 100,
                 letter_hold_ms: int = 200):
        self._db = database
        self._renderer = renderer
        self._sign_fps = sign_fps
        self._pause_s = pause_ms / 1000.0
        self._letter_hold_s = letter_hold_ms / 1000.0
        self._frame_interval = 1.0 / sign_fps

        self._lock = threading.Lock()
        self._queue: collections.deque[tuple[str, np.ndarray | None]] = collections.deque()

        # Current animation state
        self._state = "idle"  # idle | signing | pause
        self._current_label = ""
        self._current_frames: np.ndarray | None = None  # (T, 543, 3)
        self._current_idx = 0
        self._last_frame_time = 0.0
        self._pause_start = 0.0

        # Cached last rendered frame (for pause state)
        self._last_rendered: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, glosses: list[str]) -> None:
        """Add glosses to the playback queue. Drops oldest if queue overflows."""
        with self._lock:
            for gloss in glosses:
                self._enqueue_one(gloss)
            # Drop oldest if queue exceeds max — keep it real-time
            while len(self._queue) > _MAX_QUEUE:
                dropped = self._queue.popleft()
                logger.debug("Dropped old sign from queue: %s", dropped[0])

    def get_frame(self) -> np.ndarray:
        """Return the current avatar BGR frame. Called at ~30fps. Thread-safe."""
        with self._lock:
            return self._advance()

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state != "idle" or len(self._queue) > 0

    @property
    def current_sign(self) -> str:
        with self._lock:
            return self._current_label if self._state != "idle" else ""

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Clear queue and reset to idle."""
        with self._lock:
            self._queue.clear()
            self._state = "idle"
            self._current_frames = None
            self._current_label = ""
            self._last_rendered = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue_one(self, gloss: str) -> None:
        """Resolve a single gloss into playable (label, frames) entries."""
        # Dedup: skip if same as last queued sign
        if self._queue and self._queue[-1][0].lower() == gloss.strip().lower():
            return

        # Fingerspelled word: "J-O-H-N" → individual letters
        # Only fingerspell short words (≤4 letters) during fast speech
        if "-" in gloss and len(gloss) > 1:
            letters = [l.strip().upper() for l in gloss.split("-") if l.strip()]
            if len(letters) > 4:
                # Too long to fingerspell in fast speech — show as text card
                self._queue.append((gloss, None))
                return
            for letter in letters:
                frames = self._db.get(letter)
                self._queue.append((letter, frames))
            return

        # Known sign in database
        frames = self._db.get(gloss)
        if frames is not None:
            # Cap frames to MAX_SIGN_FRAMES (0.6s at 30fps)
            if len(frames) > _MAX_SIGN_FRAMES:
                indices = np.linspace(0, len(frames) - 1, _MAX_SIGN_FRAMES, dtype=int)
                frames = frames[indices]
            self._queue.append((gloss, frames))
            return

        # Unknown sign — fingerspell only if short (≤4 letters)
        word = gloss.strip().upper()
        if len(word) <= 4:
            for ch in word:
                if ch.isalpha():
                    frames = self._db.get(ch)
                    self._queue.append((ch, frames))
        else:
            # Long unknown word: show as text card, skip fingerspelling
            self._queue.append((word, None))

    def _advance(self) -> np.ndarray:
        """Advance the state machine and return the current frame."""
        now = time.monotonic()

        if self._state == "idle":
            return self._handle_idle()

        elif self._state == "signing":
            return self._handle_signing(now)

        elif self._state == "pause":
            return self._handle_pause(now)

        return self._renderer.render_idle()

    def _handle_idle(self) -> np.ndarray:
        """Check queue; if non-empty, start next sign."""
        if not self._queue:
            return self._renderer.render_idle()

        label, frames = self._queue.popleft()
        self._current_label = label
        self._current_frames = frames
        self._current_idx = 0
        self._last_frame_time = time.monotonic()

        if frames is not None and len(frames) > 0:
            self._state = "signing"
            logger.debug("Playing sign: %s (%d frames)", label, len(frames))
            return self._render_current()
        else:
            # No landmark data — show text card briefly then pause
            self._state = "pause"
            self._pause_start = time.monotonic()
            self._last_rendered = self._renderer.render_text_card(label)
            return self._last_rendered

    def _handle_signing(self, now: float) -> np.ndarray:
        """Advance through frames of the current sign. No hold/loop at end."""
        elapsed = now - self._last_frame_time

        if self._current_frames is not None:
            n_frames = len(self._current_frames)
            if n_frames == 1:
                # Single-frame (letter): hold briefly then move on
                if elapsed >= self._letter_hold_s:
                    self._state = "idle"  # skip pause, go straight to next
                    return self._handle_idle()
                return self._render_current()
            else:
                # Multi-frame: advance at sign_fps
                if elapsed >= self._frame_interval:
                    self._current_idx += 1
                    self._last_frame_time = now

                    if self._current_idx >= n_frames:
                        # Done — immediately move to next sign (no hold/loop)
                        self._state = "idle"
                        return self._handle_idle()

        return self._render_current()

    def _handle_pause(self, now: float) -> np.ndarray:
        """Minimal pause between signs — skip entirely if queue has items."""
        # If queue has pending signs, skip the pause entirely
        if self._queue:
            self._state = "idle"
            return self._handle_idle()

        if now - self._pause_start >= self._pause_s:
            self._state = "idle"
            return self._handle_idle()

        return self._last_rendered if self._last_rendered is not None else self._renderer.render_idle()

    def _enter_pause(self, now: float) -> None:
        self._state = "pause"
        self._pause_start = now

    def _render_current(self) -> np.ndarray:
        """Render the current frame of the current sign."""
        if self._current_frames is None or len(self._current_frames) == 0:
            return self._renderer.render_idle()

        idx = min(self._current_idx, len(self._current_frames) - 1)
        n_frames = len(self._current_frames)
        progress = (idx + 1) / n_frames if n_frames > 0 else 0.0

        frame = self._renderer.render_frame(
            self._current_frames[idx],
            label=self._current_label,
            progress=progress,
        )
        self._last_rendered = frame
        return frame
