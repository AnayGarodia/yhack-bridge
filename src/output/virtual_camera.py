"""
Virtual camera output — pipes processed frames into a virtual webcam device
that Google Meet (or any video call app) sees as a real camera.

Platform backends:
  macOS:   pyvirtualcam + OBS Virtual Camera
  Linux:   pyvirtualcam + v4l2loopback kernel module
  Windows: pyvirtualcam + OBS Virtual Camera
"""

import logging
import platform
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PYVIRTUALCAM_AVAILABLE = False
_IMPORT_ERROR = None

try:
    import pyvirtualcam
    _PYVIRTUALCAM_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = e


def _platform_setup_hint() -> str:
    """Return OS-specific install instructions for the virtual camera backend."""
    system = platform.system()
    if system == "Darwin":
        return (
            "macOS virtual camera setup:\n"
            "  1. Install pyvirtualcam:  pip install pyvirtualcam\n"
            "  2. Install OBS Studio:    brew install --cask obs\n"
            "  3. Open OBS → Tools → Start Virtual Camera (at least once)\n"
            "  OBS must have been launched at least once to register the virtual camera plugin."
        )
    elif system == "Linux":
        return (
            "Linux virtual camera setup:\n"
            "  1. Install pyvirtualcam:  pip install pyvirtualcam\n"
            "  2. Install v4l2loopback:  sudo apt install v4l2loopback-dkms\n"
            "  3. Load the module:       sudo modprobe v4l2loopback devices=1 video_nr=10 "
            'card_label="Bridge Virtual Cam" exclusive_caps=1\n'
            "  4. Verify:                ls /dev/video*"
        )
    elif system == "Windows":
        return (
            "Windows virtual camera setup:\n"
            "  1. Install pyvirtualcam:  pip install pyvirtualcam\n"
            "  2. Install OBS Studio from https://obsproject.com\n"
            "  3. Open OBS → Tools → Start Virtual Camera (at least once)\n"
            "  OBS must have been launched at least once to register the virtual camera plugin."
        )
    return "Unsupported platform. pyvirtualcam supports macOS, Linux, and Windows."


class VirtualCamera:
    """Sends BGR (OpenCV) frames to a virtual webcam device."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._cam = None
        self._lock = threading.Lock()
        self._status = "stopped"
        self._frame_drop_timeout = 1.0 / fps  # drop frame if send takes longer than one frame period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Check whether the pyvirtualcam backend is importable."""
        return _PYVIRTUALCAM_AVAILABLE

    @property
    def status(self) -> str:
        """Return 'running', 'stopped', or 'unavailable'."""
        return self._status

    def start(self) -> bool:
        """Start the virtual camera. Returns False if backend unavailable."""
        if not _PYVIRTUALCAM_AVAILABLE:
            self._status = "unavailable"
            logger.warning(
                "pyvirtualcam is not installed.\n%s", _platform_setup_hint()
            )
            return False

        try:
            self._cam = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                print_fps=False,
            )
            self._status = "running"
            logger.info(
                "Virtual camera started: %s (%dx%d @ %d fps)",
                self._cam.device, self._width, self._height, self._fps,
            )
            return True
        except Exception as exc:
            self._status = "unavailable"
            logger.error(
                "Failed to start virtual camera: %s\n%s",
                exc, _platform_setup_hint(),
            )
            return False

    def send_frame(self, bgr_frame: np.ndarray) -> None:
        """Send a BGR frame to the virtual camera (thread-safe, non-blocking).

        - Converts BGR → RGB (pyvirtualcam expects RGB).
        - Resizes if dimensions don't match target resolution.
        - Drops the frame if the lock can't be acquired immediately.
        """
        if self._status != "running" or self._cam is None:
            return

        acquired = self._lock.acquire(timeout=self._frame_drop_timeout)
        if not acquired:
            return  # drop frame rather than block

        try:
            # Resize if needed
            h, w = bgr_frame.shape[:2]
            if w != self._width or h != self._height:
                bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))

            # BGR → RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            self._cam.send(rgb_frame)
        except Exception as exc:
            logger.debug("Frame send failed: %s", exc)
        finally:
            self._lock.release()

    def stop(self) -> None:
        """Stop the virtual camera and release resources."""
        with self._lock:
            if self._cam is not None:
                try:
                    self._cam.close()
                except Exception:
                    pass
                self._cam = None
            self._status = "stopped"
            logger.info("Virtual camera stopped.")
