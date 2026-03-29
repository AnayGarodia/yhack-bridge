"""
Frame composer — overlays transcription text, status indicators, and FPS
counter onto a webcam frame before it gets sent to the virtual camera.
"""

import time

import cv2
import numpy as np


class FrameComposer:
    """Composes an output frame with overlays for the virtual camera feed."""

    def __init__(self, width: int = 1280, height: int = 720):
        self._width = width
        self._height = height
        self._fps_times: list[float] = []
        self._fps_max_samples = 60

    def compose(
        self,
        bgr_frame: np.ndarray,
        transcription_text: str = "",
        is_active: bool = True,
        is_muted: bool = False,
    ) -> np.ndarray:
        """Overlay transcription, status indicators, and FPS onto the frame.

        Args:
            bgr_frame: Raw webcam frame (BGR, any resolution).
            transcription_text: Current recognized text to display.
            is_active: True if recognition is actively processing.
            is_muted: True if transcription is paused/muted.

        Returns:
            Composited BGR frame at target resolution.
        """
        t0 = time.perf_counter()

        # Resize to target resolution
        h, w = bgr_frame.shape[:2]
        if w != self._width or h != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))
        else:
            bgr_frame = bgr_frame.copy()

        h, w = bgr_frame.shape[:2]

        # -- Transcription text at bottom --
        self._draw_transcription(bgr_frame, w, h, transcription_text, is_active)

        # -- Mute indicator --
        if is_muted:
            self._draw_mute_indicator(bgr_frame, w)

        # -- FPS counter top-right --
        self._fps_times.append(time.perf_counter() - t0)
        if len(self._fps_times) > self._fps_max_samples:
            self._fps_times = self._fps_times[-self._fps_max_samples:]
        fps = 1.0 / (sum(self._fps_times) / len(self._fps_times)) if self._fps_times else 0
        self._draw_fps(bgr_frame, w, fps)

        return bgr_frame

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_transcription(
        self, frame: np.ndarray, w: int, h: int,
        text: str, is_active: bool,
    ) -> None:
        bar_height = 52
        y_bar_top = h - bar_height

        # Semi-transparent black bar
        overlay = frame[y_bar_top:h, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame[y_bar_top:h, 0:w], 0.4, 0, frame[y_bar_top:h, 0:w])

        if text:
            # Truncate to fit
            display = text if len(text) <= 80 else "..." + text[-77:]
            cv2.putText(
                frame, display, (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
        elif is_active:
            cv2.putText(
                frame, "TRANSCRIBING...", (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA,
            )

    def _draw_mute_indicator(self, frame: np.ndarray, w: int) -> None:
        label = "MUTED"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - tw - 14
        y = 60
        cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 180), -1)
        cv2.putText(
            frame, label, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _draw_fps(self, frame: np.ndarray, w: int, fps: float) -> None:
        label = f"{fps:.0f} fps"
        cv2.putText(
            frame, label, (w - 90, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )
