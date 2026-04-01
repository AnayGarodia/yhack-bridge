"""
MeetSession — connects avatar frame source to virtual camera for Google Meet.

Usage:
    session = MeetSession()
    if session.start():
        # Push frames from your main loop:
        session.push_frame(bgr_frame, transcription_text="Hello world")
        # ... later ...
        session.stop()
"""

import logging
import time
import threading

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
    def is_running(self) -> bool:
        return self._vcam.is_running

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @is_muted.setter
    def is_muted(self, value: bool) -> None:
        self._is_muted = value

    @property
    def device(self) -> str:
        return self._vcam.device

    def start(self) -> bool:
        """Start the virtual camera. Returns False if unavailable."""
        ok = self._vcam.start()
        if ok:
            print("[meet] MeetSession started -- avatar is live in virtual camera")
        else:
            print("[meet] MeetSession: virtual camera unavailable")
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

    def push_raw_frame(self, bgr_frame: np.ndarray) -> None:
        """Send a frame directly without overlays."""
        self._vcam.send_frame(bgr_frame)

    def stop(self) -> None:
        self._vcam.stop()
        print("[meet] MeetSession stopped")
