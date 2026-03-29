"""
BridgeCamera — ASL interpreter bot for Google Meet.

Joins as a separate participant via OBS Virtual Camera.
Shows a professional interpreter display with:
- Large sign name when signing
- Speaker text when hearing speech
- ASL gloss bar
- Clean animated status indicators

Frame is pre-flipped horizontally so it reads correctly in
Google Meet's self-view (Meet mirrors self-view).
"""

import math
import threading
import time

import cv2
import numpy as np

from .virtual_camera import VirtualCamera


class BridgeCamera:
    """ASL interpreter bot — professional display as virtual camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._w = width
        self._h = height
        self._vcam = VirtualCamera(width=width, height=height, fps=fps)

        self._lock = threading.Lock()
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
            print(f"[bot] ASL interpreter bot started — select 'OBS Virtual Camera' in Meet")
        return ok

    def stop(self):
        self._vcam.stop()

    @property
    def is_running(self) -> bool:
        return self._vcam.status == "running"

    def set_sign(self, sign: str, confidence: float):
        with self._lock:
            self._current_sign = sign
            self._sign_conf = confidence

    def add_committed_sign(self, sign: str):
        with self._lock:
            self._building_signs.append(sign)
            if len(self._building_signs) > 10:
                self._building_signs = self._building_signs[-10:]

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

    def update_avatar(self, bgr_frame: np.ndarray):
        pass  # Not used — we render our own frame

    def send_composed_frame(self):
        """Render and send frame to virtual camera."""
        if not self.is_running:
            return

        with self._lock:
            sign = self._current_sign
            conf = self._sign_conf
            building = list(self._building_signs)
            translation = self._translation
            trans_age = time.time() - self._translation_time
            speaker = self._speaker_text
            speaker_age = time.time() - self._speaker_time
            glosses = self._speaker_glosses
            mic = self._mic_active

        frame = self._render(sign, conf, building, translation, trans_age,
                             speaker, speaker_age, glosses, mic)

        # Pre-flip horizontally — Meet mirrors self-view, so double-flip = correct
        frame = cv2.flip(frame, 1)

        self._vcam.send_frame(frame)

    def _render(self, sign, conf, building, translation, trans_age,
                speaker, speaker_age, glosses, mic):
        w, h = self._w, self._h
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # ── Background gradient ──
        for y in range(h):
            t = y / h
            r = int(18 + 8 * t)
            g = int(20 + 12 * t)
            b = int(32 + 16 * t)
            frame[y, :] = [b, g, r]

        # ── Subtle grid pattern ──
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (40, 35, 30), 1)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (40, 35, 30), 1)

        t_now = time.time()

        # ── Determine display mode ──
        has_speaker = speaker and speaker_age < 12
        has_sign = sign and conf > 0.25
        has_translation = translation and trans_age < 12
        has_building = len(building) > 0

        # ── Center: Main content area ──
        if has_speaker:
            # Someone is speaking → show what they said + ASL glosses
            self._draw_speaker_mode(frame, speaker, glosses, w, h)
        elif has_sign:
            # Recognizing a sign → show the sign name large
            self._draw_sign_mode(frame, sign, conf, w, h)
        elif has_translation:
            # Completed translation
            self._draw_translation_mode(frame, translation, w, h)
        elif has_building:
            # Building sentence from signs
            self._draw_building_mode(frame, building, w, h)
        else:
            # Idle
            self._draw_idle_mode(frame, w, h, t_now)

        # ── Header bar ──
        cv2.rectangle(frame, (0, 0), (w, 48), (18, 18, 18), -1)
        # Logo
        cv2.putText(frame, "Bridge", (24, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Interpreter", (140, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
        # Mic indicator
        if mic:
            pulse = 4 + int(2 * math.sin(t_now * 4))
            cv2.circle(frame, (w - 30, 24), pulse, (60, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (w - 30, 24), pulse + 3, (60, 60, 200), 1, cv2.LINE_AA)
        # Connection indicator
        cv2.circle(frame, (w - 60, 24), 4, (80, 220, 80), -1, cv2.LINE_AA)

        # ── Bottom bar ──
        cv2.rectangle(frame, (0, h - 40), (w, h), (12, 12, 12), -1)
        # Gloss bar
        if has_building:
            gloss_text = " ".join(building[-8:])
            cv2.putText(frame, gloss_text, (24, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 200, 100), 1, cv2.LINE_AA)
        elif glosses and has_speaker:
            cv2.putText(frame, f"ASL: {glosses}", (24, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 200, 100), 1, cv2.LINE_AA)

        # Status text
        status = "INTERPRETING" if (has_speaker or has_sign) else "LISTENING"
        color = (80, 220, 80) if status == "INTERPRETING" else (120, 120, 120)
        cv2.putText(frame, status, (w - 180, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return frame

    def _draw_speaker_mode(self, frame, speaker, glosses, w, h):
        """Someone is speaking — show large quote + ASL glosses below."""
        # "Hearing" label
        cv2.putText(frame, "HEARING", (w // 2 - 60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 180, 255), 1, cv2.LINE_AA)
        # Decorative line
        cv2.line(frame, (w // 2 - 100, 110), (w // 2 + 100, 110), (60, 120, 200), 2)

        # Speaker text — large, centered, with word wrap
        words = speaker.split()
        lines = []
        line = ""
        for word in words:
            test = line + " " + word if line else word
            (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            if tw > w - 100:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        y_start = h // 2 - len(lines) * 25
        for i, ln in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(frame, ln, ((w - tw) // 2, y_start + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # ASL glosses below — gold text
        if glosses:
            (tw, _), _ = cv2.getTextSize(glosses, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            gy = y_start + len(lines) * 50 + 40
            cv2.putText(frame, glosses, ((w - tw) // 2, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2, cv2.LINE_AA)

        # Arrow indicating "translating to sign"
        cv2.putText(frame, ">>> SIGNING >>>", (w // 2 - 80, gy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    def _draw_sign_mode(self, frame, sign, conf, w, h):
        """Active sign recognition — show sign name HUGE."""
        # Sign name
        (tw, th), _ = cv2.getTextSize(sign.upper(), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
        sx = (w - tw) // 2
        sy = h // 2 + th // 2 - 20
        # Glow effect
        cv2.putText(frame, sign.upper(), (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (40, 120, 40), 8, cv2.LINE_AA)
        cv2.putText(frame, sign.upper(), (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (80, 255, 120), 5, cv2.LINE_AA)

        # Confidence bar below
        bar_w = 300
        bar_x = (w - bar_w) // 2
        bar_y = sy + 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (40, 40, 40), -1)
        fill = int(bar_w * conf)
        color = (80, 255, 120) if conf > 0.7 else (80, 200, 255) if conf > 0.5 else (80, 100, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 8), color, -1)
        cv2.putText(frame, f"{conf:.0%}", (bar_x + bar_w + 10, bar_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # "RECOGNIZING SIGN" label
        cv2.putText(frame, "RECOGNIZING SIGN", (w // 2 - 100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.line(frame, (w // 2 - 120, 110), (w // 2 + 120, 110), (60, 200, 80), 2)

    def _draw_translation_mode(self, frame, translation, w, h):
        """Show completed English translation."""
        cv2.putText(frame, "TRANSLATED", (w // 2 - 65, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (w // 2 - 80, 110), (w // 2 + 80, 110), (80, 160, 220), 2)

        # Translation text
        (tw, th), _ = cv2.getTextSize(translation, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        if tw > w - 80:
            # Word wrap
            words = translation.split()
            lines = []
            line = ""
            for word in words:
                test = line + " " + word if line else word
                (ttw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                if ttw > w - 80:
                    lines.append(line)
                    line = word
                else:
                    line = test
            if line:
                lines.append(line)
            y = h // 2 - len(lines) * 25
            for i, ln in enumerate(lines):
                (lw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.putText(frame, ln, ((w - lw) // 2, y + i * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, translation, ((w - tw) // 2, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_building_mode(self, frame, building, w, h):
        """Building sentence from signs."""
        cv2.putText(frame, "BUILDING SENTENCE", (w // 2 - 100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 200, 100), 1, cv2.LINE_AA)
        cv2.line(frame, (w // 2 - 110, 110), (w // 2 + 110, 110), (180, 160, 60), 2)

        text = " ".join(building)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(frame, text, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 200, 100), 3, cv2.LINE_AA)

    def _draw_idle_mode(self, frame, w, h, t):
        """Idle — listening animation."""
        # Pulsing circle animation
        pulse = 30 + int(10 * math.sin(t * 2))
        alpha = 0.3 + 0.2 * math.sin(t * 2)
        cv2.circle(frame, (w // 2, h // 2 - 20), pulse,
                   (int(80 * alpha), int(180 * alpha), int(80 * alpha)), 2, cv2.LINE_AA)
        cv2.circle(frame, (w // 2, h // 2 - 20), pulse + 20,
                   (int(40 * alpha), int(100 * alpha), int(40 * alpha)), 1, cv2.LINE_AA)

        cv2.putText(frame, "Listening...", (w // 2 - 75, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA)

        # Animated dots
        n_dots = int(t * 2) % 4
        dots = "." * n_dots
        cv2.putText(frame, dots, (w // 2 + 65, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
