"""
Virtual Camera — pushes frames into OBS Virtual Camera so Google Meet
sees the avatar as a real webcam.

Requirements:
1. OBS must be installed: obsproject.com
2. OBS Virtual Camera must be STARTED inside OBS before running this
3. In Google Meet: Settings -> Video -> select "OBS Virtual Camera"

Architecture:
  A dedicated push thread runs at target FPS, continuously sending the
  latest frame to the virtual camera.  The main thread calls send_frame()
  to update the current frame — this is non-blocking and thread-safe.
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
    system = platform.system()
    if system == "Darwin":
        return (
            "macOS setup:\n"
            "  1. pip install pyvirtualcam\n"
            "  2. brew install --cask obs  OR  download from obsproject.com\n"
            "  3. Open OBS -> Controls panel -> 'Start Virtual Camera'\n"
            "  4. In Google Meet: Settings -> Video -> select 'OBS Virtual Camera'"
        )
    elif system == "Linux":
        return (
            "Linux setup:\n"
            "  1. pip install pyvirtualcam\n"
            "  2. sudo apt install v4l2loopback-dkms\n"
            "  3. sudo modprobe v4l2loopback devices=1 video_nr=10 "
            'card_label="Bridge Virtual Cam" exclusive_caps=1'
        )
    elif system == "Windows":
        return (
            "Windows setup:\n"
            "  1. pip install pyvirtualcam\n"
            "  2. Install OBS from obsproject.com\n"
            "  3. Open OBS -> Tools -> Start Virtual Camera"
        )
    return "Unsupported platform."


class VirtualCamera:
    """Sends BGR (OpenCV) frames to a virtual webcam device via a push loop."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._cam = None
        self._running = False
        self._lock = threading.Lock()
        self._current_frame = None
        self._push_thread = None
        self._frames_pushed = 0
        self._last_fps_check = time.time()
        self._actual_fps = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        return _PYVIRTUALCAM_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self._running and self._cam is not None

    @property
    def status(self) -> str:
        if self._running and self._cam is not None:
            return "running"
        if not _PYVIRTUALCAM_AVAILABLE:
            return "unavailable"
        return "stopped"

    @property
    def device(self) -> str:
        return self._cam.device if self._cam else ""

    def start(self) -> bool:
        """Start the virtual camera and push loop. Returns False on failure."""
        if not _PYVIRTUALCAM_AVAILABLE:
            print(f"ERROR: pyvirtualcam not installed\n{_platform_setup_hint()}")
            return False

        try:
            # Try OBS backend first (macOS/Windows)
            try:
                self._cam = pyvirtualcam.Camera(
                    width=self.width, height=self.height, fps=self.fps,
                    backend='obs', print_fps=False,
                )
            except Exception:
                self._cam = pyvirtualcam.Camera(
                    width=self.width, height=self.height, fps=self.fps,
                    print_fps=False,
                )

            print(f"[vcam] Virtual camera started: {self._cam.device}")
            print(f"[vcam] Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print(f"[vcam] In Google Meet: Settings -> Video -> '{self._cam.device}'")

            # Start with a placeholder so Meet never sees black
            self._current_frame = self._make_placeholder_frame()

            self._running = True
            self._push_thread = threading.Thread(
                target=self._push_loop, name="vcam-push", daemon=True)
            self._push_thread.start()
            return True

        except Exception as e:
            print(f"ERROR: Failed to start virtual camera: {e}")
            print(f"\n{_platform_setup_hint()}")
            return False

    def send_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Update the current frame (BGR, any resolution).
        Thread-safe and non-blocking — the push loop will send it.
        """
        if bgr_frame is None:
            return

        # Resize if needed
        if bgr_frame.shape[:2] != (self.height, self.width):
            bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

        # Convert BGR -> RGB for pyvirtualcam
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        with self._lock:
            self._current_frame = rgb_frame

    def stop(self) -> None:
        self._running = False
        if self._push_thread and self._push_thread.is_alive():
            self._push_thread.join(timeout=2.0)
        if self._cam:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        print("[vcam] Virtual camera stopped")

    # ------------------------------------------------------------------
    # Push loop — dedicated thread for steady frame delivery
    # ------------------------------------------------------------------

    def _push_loop(self):
        """Push the latest frame to the virtual camera at target FPS."""
        logger.info("Virtual camera push loop started")

        while self._running and self._cam:
            try:
                with self._lock:
                    frame = self._current_frame

                if frame is not None:
                    self._cam.send(frame)
                    self._frames_pushed += 1

                self._cam.sleep_until_next_frame()

                # Log actual FPS every 10 seconds
                now = time.time()
                if now - self._last_fps_check > 10.0:
                    elapsed = now - self._last_fps_check
                    if elapsed > 0:
                        self._actual_fps = self._frames_pushed / elapsed
                    self._frames_pushed = 0
                    self._last_fps_check = now

            except Exception as e:
                if self._running:
                    logger.debug("Push error: %s", e)
                    time.sleep(0.033)

    def _make_placeholder_frame(self) -> np.ndarray:
        """Dark frame with text — shown before avatar loads."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Dark blue-grey background (RGB for pyvirtualcam)
        frame[:] = [30, 33, 48]
        # Add loading text (use BGR temporarily for cv2, then convert)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, "ASL Avatar Loading...",
                    (self.width // 2 - 220, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, "Bridge — Real-time ASL Translator",
                    (self.width // 2 - 260, self.height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
