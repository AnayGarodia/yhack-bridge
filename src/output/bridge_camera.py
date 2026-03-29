"""
BridgeCamera — ASL interpreter bot for Google Meet.

This IS the bot. It joins the meeting as a separate participant
(via OBS Virtual Camera). Other people join with their real cameras.

When people speak → bot shows ASL avatar signing
When someone signs → bot's TTS speaks the English translation

The feed is NOT mirrored — text reads correctly because Meet
mirrors self-view but other participants see it normally.
"""

import threading
import time

import cv2
import numpy as np

from .virtual_camera import VirtualCamera


def _make_gradient(w, h):
    """Dark professional gradient background."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        bg[y, :] = [int(40 + 12 * t), int(28 + 14 * t), int(22 + 10 * t)]
    return bg


class BridgeCamera:
    """ASL interpreter bot — avatar signing as virtual camera feed."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._w = width
        self._h = height
        self._vcam = VirtualCamera(width=width, height=height, fps=fps)
        self._bg = _make_gradient(width, height)

        self._lock = threading.Lock()
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
        self._status = "Listening..."

    def start(self) -> bool:
        ok = self._vcam.start()
        if ok:
            print(f"[bot] ASL interpreter bot started")
            print(f"[bot] Camera: {self._vcam._cam.device if self._vcam._cam else '?'}")
            print(f"[bot] Join Meet and select 'OBS Virtual Camera' as a second participant")
        return ok

    def stop(self):
        self._vcam.stop()

    @property
    def is_running(self) -> bool:
        return self._vcam.status == "running"

    def update_avatar(self, bgr_frame: np.ndarray):
        with self._lock:
            self._avatar_frame = bgr_frame.copy()

    def set_sign(self, sign: str, confidence: float):
        with self._lock:
            self._current_sign = sign
            self._sign_conf = confidence
            self._status = f"Signing: {sign}"

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
            self._status = "Listening..."

    def set_speaker_text(self, text: str, glosses: str = ""):
        with self._lock:
            self._speaker_text = text
            self._speaker_time = time.time()
            self._speaker_glosses = glosses
            self._status = "Interpreting..."

    def set_mic_active(self, active: bool):
        with self._lock:
            self._mic_active = active

    def clear(self):
        with self._lock:
            self._current_sign = ""
            self._building_signs.clear()
            self._translation = ""
            self._speaker_text = ""
            self._status = "Listening..."

    def send_composed_frame(self):
        """Compose and send frame to virtual camera."""
        if not self.is_running:
            return

        frame = self._bg.copy()
        w, h = self._w, self._h

        with self._lock:
            avatar = self._avatar_frame
            sign = self._current_sign
            conf = self._sign_conf
            building = list(self._building_signs)
            translation = self._translation
            trans_age = time.time() - self._translation_time
            speaker = self._speaker_text
            speaker_age = time.time() - self._speaker_time
            glosses = self._speaker_glosses
            status = self._status
            mic = self._mic_active

        # ── Avatar (center, large) ──
        if avatar is not None:
            ah, aw = avatar.shape[:2]
            # Fill the frame height minus header/footer
            avail_h = h - 140
            avail_w = w
            scale = min(avail_w / aw, avail_h / ah)
            new_w = int(aw * scale)
            new_h = int(ah * scale)
            resized = cv2.resize(avatar, (new_w, new_h))
            x_off = (w - new_w) // 2
            y_off = 50 + (avail_h - new_h) // 2
            frame[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # ── Header ──
        # Gradient header bar
        for y in range(44):
            alpha = 0.85 - y * 0.005
            frame[y, :] = (frame[y, :].astype(float) * (1 - alpha) + np.array([18, 18, 18]) * alpha).astype(np.uint8)

        cv2.putText(frame, "Bridge", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Interpreter", (120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        # Status pill
        pill_text = status
        (tw, _), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        pill_x = w - tw - 30
        pill_color = (60, 200, 80) if "Signing" in status else (80, 160, 220) if "Interpreting" in status else (120, 120, 120)
        cv2.rectangle(frame, (pill_x - 10, 12), (pill_x + tw + 10, 36), pill_color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (pill_x - 10, 12), (pill_x + tw + 10, 36), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, pill_text, (pill_x, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Footer subtitle area ──
        footer_h = 90
        footer_y = h - footer_h

        # Gradient footer
        for y in range(footer_h):
            row = footer_y + y
            alpha = 0.3 + y / footer_h * 0.6
            frame[row, :] = (frame[row, :].astype(float) * (1 - alpha) + np.array([12, 12, 12]) * alpha).astype(np.uint8)

        # Line 1: What's being signed or building sentence
        y1 = footer_y + 28
        if sign and conf > 0.25:
            # Show current sign with green accent
            cv2.putText(frame, sign.upper(), (24, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 120), 2, cv2.LINE_AA)
            # Confidence bar
            bar_x = 24
            bar_y = y1 + 8
            bar_w = 120
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * conf), bar_y + 4), (80, 255, 120), -1)
        elif building:
            signs_text = " ".join(building[-6:])
            cv2.putText(frame, signs_text, (24, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 1, cv2.LINE_AA)

        # Line 2: Translation or speaker text
        y2 = footer_y + 62
        if speaker and speaker_age < 12:
            # What the speaker said
            display = speaker if len(speaker) <= 70 else speaker[:67] + "..."
            cv2.putText(frame, f'"{display}"', (24, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # Show glosses on the right
            if glosses:
                (gw, _), _ = cv2.getTextSize(glosses, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(frame, glosses, (w - gw - 20, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 80), 1, cv2.LINE_AA)
        elif translation and trans_age < 12:
            display = translation if len(translation) <= 70 else translation[:67] + "..."
            cv2.putText(frame, display, (24, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Mic indicator
        if mic:
            cv2.circle(frame, (w - 20, footer_y + 45), 6, (0, 0, 220), -1, cv2.LINE_AA)
            # Pulse animation
            pulse = int(3 * abs(time.time() % 1 - 0.5) * 2)
            cv2.circle(frame, (w - 20, footer_y + 45), 6 + pulse, (0, 0, 180), 1, cv2.LINE_AA)

        self._vcam.send_frame(frame)
