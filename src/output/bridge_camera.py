"""
BridgeCamera — composites webcam + live ASL translation into a virtual camera
feed that appears as a real webcam in Google Meet, Zoom, or any video call app.

What the other person sees in the call:
┌─────────────────────────────────────────────┐
│  ┌─ Bridge — ASL Translator ─────── LIVE ─┐ │
│  │                                         │ │
│  │         [Your webcam feed               │ │
│  │          with hand landmarks]           │ │
│  │                                         │ │
│  ├─────────────────────────────────────────┤ │
│  │ 🤟 Signing: HELLO THANK-YOU             │ │
│  │ 💬 "Hello, thank you!"                  │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘

Usage:
    bridge_cam = BridgeCamera()
    bridge_cam.start()
    bridge_cam.update_frame(bgr_frame)
    bridge_cam.set_sign("HELLO", 0.95)
    bridge_cam.set_translation("Hello!")
    bridge_cam.stop()
"""

import threading
import time

import cv2
import numpy as np

from .virtual_camera import VirtualCamera


class BridgeCamera:
    """Composites webcam + ASL translation into a virtual camera for video calls."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._w = width
        self._h = height
        self._fps = fps
        self._vcam = VirtualCamera(width=width, height=height, fps=fps)

        # State (thread-safe)
        self._lock = threading.Lock()
        self._current_sign = ""
        self._sign_conf = 0.0
        self._building_signs: list[str] = []
        self._translation = ""
        self._translation_time = 0.0
        self._speaker_text = ""
        self._speaker_time = 0.0
        self._mic_active = False

    # ── Public API ───────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the virtual camera. Returns False if unavailable."""
        ok = self._vcam.start()
        if ok:
            print(f"[bridge-cam] Virtual camera started — select it in Google Meet")
            print(f"[bridge-cam] Device: {self._vcam._cam.device if self._vcam._cam else 'unknown'}")
        else:
            print(f"[bridge-cam] Virtual camera unavailable (OBS not installed?)")
        return ok

    def stop(self):
        self._vcam.stop()

    @property
    def is_running(self) -> bool:
        return self._vcam.status == "running"

    def send_frame(self, bgr_frame: np.ndarray):
        """Composite overlays onto webcam frame and send to virtual camera."""
        if not self.is_running:
            return

        # Resize to target
        h, w = bgr_frame.shape[:2]
        if w != self._w or h != self._h:
            bgr_frame = cv2.resize(bgr_frame, (self._w, self._h))
        else:
            bgr_frame = bgr_frame.copy()

        with self._lock:
            sign = self._current_sign
            conf = self._sign_conf
            building = list(self._building_signs)
            translation = self._translation
            trans_age = time.time() - self._translation_time
            speaker = self._speaker_text
            speaker_age = time.time() - self._speaker_time
            mic = self._mic_active

        # Draw overlays
        self._draw_header(bgr_frame)
        self._draw_subtitle_bar(bgr_frame, sign, conf, building, translation,
                                trans_age, speaker, speaker_age, mic)

        self._vcam.send_frame(bgr_frame)

    def set_sign(self, sign: str, confidence: float):
        """Update the current live sign prediction."""
        with self._lock:
            self._current_sign = sign
            self._sign_conf = confidence

    def add_committed_sign(self, sign: str):
        """Add a committed sign to the building sentence."""
        with self._lock:
            self._building_signs.append(sign)
            if len(self._building_signs) > 8:
                self._building_signs = self._building_signs[-8:]

    def set_translation(self, english: str):
        """Set the completed English translation (from ASL→English pipeline)."""
        with self._lock:
            self._translation = english
            self._translation_time = time.time()
            self._building_signs.clear()

    def set_speaker_text(self, text: str):
        """Set what the hearing person said (from STT)."""
        with self._lock:
            self._speaker_text = text
            self._speaker_time = time.time()

    def set_mic_active(self, active: bool):
        with self._lock:
            self._mic_active = active

    def clear(self):
        with self._lock:
            self._current_sign = ""
            self._building_signs.clear()
            self._translation = ""
            self._speaker_text = ""

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw_header(self, frame: np.ndarray):
        """Draw thin branded header bar at top."""
        h, w = frame.shape[:2]
        bar_h = 32

        # Semi-transparent dark bar
        overlay = frame[:bar_h, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame[:bar_h, :], 0.3, 0, frame[:bar_h, :])

        # Logo text
        cv2.putText(frame, "Bridge", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Translator", (85, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        # Live indicator
        cv2.circle(frame, (w - 55, 16), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "LIVE", (w - 45, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_subtitle_bar(self, frame: np.ndarray, sign: str, conf: float,
                           building: list, translation: str, trans_age: float,
                           speaker: str, speaker_age: float, mic: bool):
        """Draw subtitle bar at the bottom showing translations."""
        h, w = frame.shape[:2]

        # Determine what to show
        lines = []

        # Line 1: Current signing activity
        if sign and conf > 0.3:
            lines.append(("sign", f"Signing: {sign} ({conf:.0%})"))
        elif building:
            lines.append(("sign", "Signs: " + " ".join(building[-6:])))

        # Line 2: Translation (show for 8 seconds after completion)
        if translation and trans_age < 8.0:
            lines.append(("translation", translation))

        # Line 3: Speaker text (show for 8 seconds)
        if speaker and speaker_age < 8.0:
            lines.append(("speaker", f"Speaker: {speaker}"))

        if not lines:
            return

        # Bar height based on content
        line_h = 28
        bar_h = len(lines) * line_h + 16
        y_start = h - bar_h

        # Semi-transparent dark bar
        overlay = frame[y_start:h, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame[y_start:h, :], 0.25, 0, frame[y_start:h, :])

        # Draw each line
        for i, (line_type, text) in enumerate(lines):
            y = y_start + 8 + (i + 1) * line_h - 6

            if line_type == "sign":
                cv2.putText(frame, text, (16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
            elif line_type == "translation":
                cv2.putText(frame, text, (16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            elif line_type == "speaker":
                cv2.putText(frame, text, (16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1, cv2.LINE_AA)

        # Mic indicator
        if mic:
            cv2.circle(frame, (w - 30, y_start + bar_h // 2), 8, (0, 0, 200), -1, cv2.LINE_AA)
            cv2.putText(frame, "MIC", (w - 60, y_start + bar_h // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1, cv2.LINE_AA)
