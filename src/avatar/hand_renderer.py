"""
Hand renderer — renders ASL sign names as animated frames.

Each sign is rendered as exactly 18 frames (0.6s at 30fps):
  - Frames 1-4: fade in
  - Frames 5-14: hold
  - Frames 15-18: fade out
No hold/loop at end — immediately ready for next word.
"""

import cv2
import numpy as np

FRAMES_PER_SIGN = 18   # 18 frames at 30fps = 0.6 seconds
WIDTH  = 640
HEIGHT = 480
BG_COLOR = (20, 20, 20)


class HandRenderer:
    """Renders ASL sign words as animated frame sequences."""

    def __init__(self, width=WIDTH, height=HEIGHT, frames_per_sign=FRAMES_PER_SIGN):
        self._width = width
        self._height = height
        self._frames_per_sign = frames_per_sign
        self._current_word = None
        self._frame_idx = 0
        self._frames_cache = []

    def set_word(self, word: str):
        """Set the current word to render. Generates all frames immediately."""
        self._current_word = word
        self._frame_idx = 0
        self._frames_cache = self._generate_frames(word)

    def next_frame(self):
        """
        Get the next animation frame for the current word.
        Returns (frame_bgr, done) where done=True when all frames shown.
        """
        if not self._frames_cache:
            return self._blank_frame(), True

        if self._frame_idx >= len(self._frames_cache):
            return self._blank_frame(), True

        frame = self._frames_cache[self._frame_idx]
        self._frame_idx += 1
        done = self._frame_idx >= len(self._frames_cache)
        return frame, done

    @property
    def is_idle(self):
        return self._current_word is None or self._frame_idx >= len(self._frames_cache)

    def _blank_frame(self):
        frame = np.full((self._height, self._width, 3), BG_COLOR, dtype=np.uint8)
        cv2.putText(frame, "Listening...", (self._width // 2 - 100, self._height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        return frame

    def _generate_frames(self, word: str):
        """Generate exactly frames_per_sign frames for a word."""
        n = self._frames_per_sign
        frames = []

        for i in range(n):
            frame = np.full((self._height, self._width, 3), BG_COLOR, dtype=np.uint8)

            # Fade: frames 0-3 fade in, 4-13 full, 14-17 fade out
            if i < 4:
                alpha = (i + 1) / 4.0
            elif i >= n - 4:
                alpha = (n - i) / 4.0
            else:
                alpha = 1.0

            # Word text (large, centered)
            display = word.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Scale font based on word length
            if len(display) <= 4:
                scale, thick = 2.5, 4
            elif len(display) <= 8:
                scale, thick = 1.8, 3
            else:
                scale, thick = 1.2, 2

            (tw, th), _ = cv2.getTextSize(display, font, scale, thick)
            x = (self._width - tw) // 2
            y = (self._height + th) // 2

            # Apply alpha to text color
            color = (int(80 * alpha + 20), int(255 * alpha), int(80 * alpha + 20))
            cv2.putText(frame, display, (x, y), font, scale, color, thick, cv2.LINE_AA)

            # Small label "ASL SIGN" above
            label_color = (int(100 * alpha), int(100 * alpha), int(100 * alpha))
            cv2.putText(frame, "ASL SIGN", (self._width // 2 - 45, y - th - 20),
                        font, 0.5, label_color, 1, cv2.LINE_AA)

            # Progress bar at bottom
            bar_y = self._height - 8
            bar_w = int((i + 1) / n * self._width)
            cv2.rectangle(frame, (0, bar_y), (bar_w, self._height), (0, int(180 * alpha), 0), -1)

            frames.append(frame)

        return frames
