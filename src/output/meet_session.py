"""
Meet session — ties virtual camera and frame composer together.

Usage:
    session = MeetSession()
    session.start()
    while running:
        session.push_frame(bgr_frame, transcription_text)
    session.stop()
"""

import logging

import numpy as np

from .frame_composer import FrameComposer
from .virtual_camera import VirtualCamera

logger = logging.getLogger(__name__)


class MeetSession:
    """High-level interface: compose overlays and pipe to virtual camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._vcam = VirtualCamera(width=width, height=height, fps=fps)
        self._composer = FrameComposer(width=width, height=height)
        self._is_muted = False

    @property
    def is_available(self) -> bool:
        return VirtualCamera.is_available()

    @property
    def status(self) -> str:
        return self._vcam.status

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @is_muted.setter
    def is_muted(self, value: bool) -> None:
        self._is_muted = value

    def start(self) -> bool:
        """Start the virtual camera. Returns False if unavailable."""
        ok = self._vcam.start()
        if ok:
            logger.info("MeetSession started — virtual camera is live.")
        else:
            logger.warning("MeetSession: virtual camera unavailable, frames will be dropped.")
        return ok

    def push_frame(
        self,
        bgr_frame: np.ndarray,
        transcription_text: str = "",
        is_active: bool = True,
    ) -> None:
        """Compose overlays onto the frame and send to the virtual camera."""
        composed = self._composer.compose(
            bgr_frame,
            transcription_text=transcription_text,
            is_active=is_active,
            is_muted=self._is_muted,
        )
        self._vcam.send_frame(composed)

    def stop(self) -> None:
        """Stop the virtual camera."""
        self._vcam.stop()
        logger.info("MeetSession stopped.")
