"""
BridgeCamera — ASL avatar bot for Google Meet.

Shows the ASL avatar performing signs as the virtual camera feed.
When meeting participants speak, the avatar signs along in real-time.
Small picture-in-picture webcam in the corner + subtitles.

What other participants see:
┌─────────────────────────────────────────────┐
│  Bridge — ASL Interpreter              LIVE │
│                                    ┌──────┐ │
│         [ASL Avatar                │webcam│ │
│          signing along]            │ PiP  │ │
│                                    └──────┘ │
│                                             │
│  Currently signing: HELLO                   │
│  Speaker: "Hello, how are you?"             │
└─────────────────────────────────────────────┘
"""

import threading
import time

import math

import cv2
import numpy as np

from .virtual_camera import VirtualCamera


# Dark gradient background
def _make_bg(w, h):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        bg[y, :] = [int(30 + 15 * t), int(26 + 8 * t), int(26 + 4 * t)]  # dark blue gradient
    return bg


class BridgeCamera:
    """ASL avatar bot — shows signing avatar as virtual camera in video calls."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._w = width
        self._h = height
        self._fps = fps
        self._vcam = VirtualCamera(width=width, height=height, fps=fps)
        self._bg = _make_bg(width, height)

        # State (thread-safe)
        self._lock = threading.Lock()
        self._webcam_frame: np.ndarray | None = None
        self._avatar_frame: np.ndarray | None = None
        self._current_sign = ""
        self._sign_conf = 0.0
        self._building_signs: list[str] = []
        self._translation = ""
        self._translation_time = 0.0
        self._speaker_text = ""
        self._speaker_time = 0.0
        self._speaker_glosses = ""
        self._mic_active = False

    def start(self) -> bool:
        ok = self._vcam.start()
        if ok:
            print(f"[bridge-cam] Virtual camera started — select it in Google Meet")
            print(f"[bridge-cam] Device: {self._vcam._cam.device if self._vcam._cam else 'unknown'}")
        else:
            print(f"[bridge-cam] Virtual camera unavailable (install OBS first)")
        return ok

    def stop(self):
        self._vcam.stop()

    @property
    def is_running(self) -> bool:
        return self._vcam.status == "running"

    # ── Update state ─────────────────────────────────────────────────────

    def update_webcam(self, bgr_frame: np.ndarray):
        """Update the small PiP webcam frame."""
        with self._lock:
            self._webcam_frame = bgr_frame.copy()

    def update_avatar(self, bgr_frame: np.ndarray):
        """Update the avatar frame (from RPM render loop)."""
        with self._lock:
            self._avatar_frame = bgr_frame.copy()

    def set_sign(self, sign: str, confidence: float):
        with self._lock:
            self._current_sign = sign
            self._sign_conf = confidence

    def add_committed_sign(self, sign: str):
        with self._lock:
            self._building_signs.append(sign)
            if len(self._building_signs) > 8:
                self._building_signs = self._building_signs[-8:]

    def set_translation(self, english: str):
        with self._lock:
            self._translation = english
            self._translation_time = time.time()
            self._building_signs.clear()

    def set_speaker_text(self, text: str, glosses: str = ""):
        with self._lock:
            self._speaker_text = text
            self._speaker_time = time.time()
            self._speaker_glosses = glosses

    def set_mic_active(self, active: bool):
        with self._lock:
            self._mic_active = active

    def clear(self):
        with self._lock:
            self._current_sign = ""
            self._building_signs.clear()
            self._translation = ""
            self._speaker_text = ""

    # ── Compose and send frame ───────────────────────────────────────────

    def send_composed_frame(self):
        """Compose avatar + PiP + subtitles and send to virtual camera."""
        if not self.is_running:
            return

        frame = self._bg.copy()
        w, h = self._w, self._h

        with self._lock:
            avatar = self._avatar_frame
            webcam = self._webcam_frame
            sign = self._current_sign
            conf = self._sign_conf
            building = list(self._building_signs)
            translation = self._translation
            trans_age = time.time() - self._translation_time
            speaker = self._speaker_text
            speaker_age = time.time() - self._speaker_time
            glosses = self._speaker_glosses

        # ── Main area: Avatar or sign display ──
        if avatar is not None:
            ah, aw = avatar.shape[:2]
            target_h = h - 120
            target_w = int(aw * target_h / ah)
            if target_w > w - 200:  # leave room for PiP
                target_w = w - 200
                target_h = int(ah * target_w / aw)
            resized = cv2.resize(avatar, (target_w, target_h))
            x_off = (w - target_w) // 2 - 60  # offset left to make room for PiP
            y_off = 40
            frame[y_off:y_off + target_h, x_off:x_off + target_w] = resized

        # Show current sign name large in center if signing
        if sign and conf > 0.25:
            sign_label = sign.upper()
            (tw, th), _ = cv2.getTextSize(sign_label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            sx = (w - tw) // 2
            sy = h // 2 + th // 2 - 30
            # Shadow
            cv2.putText(frame, sign_label, (sx + 2, sy + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, sign_label, (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 255, 120), 4, cv2.LINE_AA)

        # ── PiP webcam (top-right corner) ──
        if webcam is not None:
            pip_h = 140
            pip_w = int(webcam.shape[1] * pip_h / webcam.shape[0])
            pip = cv2.resize(webcam, (pip_w, pip_h))
            px = w - pip_w - 16
            py = 48
            # Border
            cv2.rectangle(frame, (px - 2, py - 2), (px + pip_w + 2, py + pip_h + 2),
                          (255, 255, 255), 2)
            frame[py:py + pip_h, px:px + pip_w] = pip

        # ── Header bar ──
        cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(frame, "Bridge", (14, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Interpreter", (100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        # Live dot
        cv2.circle(frame, (w - 50, 18), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "LIVE", (w - 40, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # ── Footer: subtitle area ──
        footer_y = h - 80
        cv2.rectangle(frame, (0, footer_y), (w, h), (15, 15, 15), -1)

        y_line = footer_y + 24
        # Current sign activity
        if sign and conf > 0.25:
            cv2.putText(frame, f"Signing: {sign}", (16, y_line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)
        elif building:
            cv2.putText(frame, "Signs: " + " ".join(building[-6:]), (16, y_line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1, cv2.LINE_AA)

        # Translation or speaker text
        y_line2 = footer_y + 52
        if translation and trans_age < 10:
            cv2.putText(frame, translation, (16, y_line2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        elif speaker and speaker_age < 10:
            cv2.putText(frame, f'"{speaker}"', (16, y_line2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 180, 255), 1, cv2.LINE_AA)
            if glosses:
                cv2.putText(frame, glosses, (w - 300, y_line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 80), 1, cv2.LINE_AA)

        # Mic indicator
        if self._mic_active:
            cv2.circle(frame, (w - 24, footer_y + 38), 8, (0, 0, 200), -1, cv2.LINE_AA)

        self._vcam.send_frame(frame)
