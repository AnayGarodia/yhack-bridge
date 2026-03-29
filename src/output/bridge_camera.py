"""
BridgeCamera — ASL interpreter bot for Google Meet.

Renders the SVG avatar character (same one from the website) into the
virtual camera feed using cairosvg for server-side SVG→image conversion.

Frame is pre-flipped so text reads correctly in Meet's self-view.
"""

import math
import os
import threading
import time

import cv2
import numpy as np

# Cairo needs Homebrew lib path on macOS
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import cairosvg

from .virtual_camera import VirtualCamera


def _svg_to_bgr(svg_string: str, width: int, height: int) -> np.ndarray:
    """Convert an SVG string to a BGR numpy array."""
    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    arr = np.frombuffer(png_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # RGBA
    if img.shape[2] == 4:
        # Alpha composite onto dark background
        alpha = img[:, :, 3:4].astype(float) / 255.0
        rgb = img[:, :, :3].astype(float)
        bg = np.full_like(rgb, [30, 28, 22])  # dark bg in BGR
        blended = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        return blended
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img


class BridgeCamera:
    """ASL interpreter bot — renders SVG avatar in virtual camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 sign_animator=None):
        self._w = width
        self._h = height
        self._vcam = VirtualCamera(width=width, height=height, fps=fps)
        self._animator = sign_animator

        self._lock = threading.Lock()
        self._current_sign = ""
        self._sign_conf = 0.0
        self._display_sign = ""  # sign currently showing in the avatar
        self._display_time = 0.0
        self._sign_queue: list[str] = []
        self._speaker_text = ""
        self._speaker_time = 0.0
        self._speaker_glosses = ""
        self._mic_active = False

        # Cache rendered avatar images
        self._svg_cache: dict[str, np.ndarray] = {}
        self._idle_img: np.ndarray | None = None

    def start(self) -> bool:
        ok = self._vcam.start()
        if ok:
            print(f"[bot] ASL interpreter bot started")
            print(f"[bot] Select 'OBS Virtual Camera' in Google Meet")
            # Pre-render idle avatar
            if self._animator:
                self._idle_img = self._render_svg(self._animator.idle_svg)
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
        pass  # handled by queue_signs

    def queue_signs(self, glosses: list[str]):
        """Queue a list of signs to display one by one."""
        with self._lock:
            self._sign_queue.extend(glosses)

    def set_translation(self, english: str):
        pass

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
            self._sign_queue.clear()
            self._display_sign = ""
            self._speaker_text = ""

    def update_avatar(self, bgr_frame):
        pass  # We render our own SVGs

    def send_composed_frame(self):
        """Render avatar + overlays and send to virtual camera."""
        if not self.is_running:
            return

        now = time.time()

        with self._lock:
            # Advance sign queue
            if self._sign_queue and (now - self._display_time > 1.2 or not self._display_sign):
                self._display_sign = self._sign_queue.pop(0)
                self._display_time = now

            # Clear display after 1.2s if queue empty
            if self._display_sign and not self._sign_queue and now - self._display_time > 1.5:
                self._display_sign = ""

            display_sign = self._display_sign
            speaker = self._speaker_text
            speaker_age = now - self._speaker_time
            glosses = self._speaker_glosses
            mic = self._mic_active
            queue_len = len(self._sign_queue)

        # Get avatar image for current sign
        if display_sign and self._animator:
            avatar_img = self._get_sign_image(display_sign)
        else:
            avatar_img = self._idle_img

        # Build frame
        frame = self._compose(avatar_img, display_sign, speaker, speaker_age,
                              glosses, mic, queue_len, now)

        # Pre-flip for correct Meet self-view
        frame = cv2.flip(frame, 1)
        self._vcam.send_frame(frame)

    def _get_sign_image(self, sign: str) -> np.ndarray | None:
        """Get rendered avatar image for a sign (cached)."""
        if sign in self._svg_cache:
            return self._svg_cache[sign]

        if not self._animator:
            return self._idle_img

        anim = self._animator.get_animation(sign)
        svg = anim.get("content", "")
        if not svg:
            return self._idle_img

        img = self._render_svg(svg)
        if img is not None:
            self._svg_cache[sign] = img
        return img

    def _render_svg(self, svg: str) -> np.ndarray | None:
        """Render SVG string to BGR image sized for the virtual camera."""
        try:
            # Render at avatar aspect ratio within the frame
            return _svg_to_bgr(svg, 560, 700)
        except Exception as e:
            print(f"[bot] SVG render error: {e}")
            return None

    def _compose(self, avatar_img, sign, speaker, speaker_age,
                 glosses, mic, queue_len, now):
        """Compose the full virtual camera frame."""
        w, h = self._w, self._h
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark gradient background
        for y in range(h):
            t = y / h
            frame[y, :] = [int(35 + 10 * t), int(28 + 12 * t), int(22 + 8 * t)]

        # ── Avatar (centered) ──
        if avatar_img is not None:
            ah, aw = avatar_img.shape[:2]
            # Scale to fit frame with margins
            avail_h = h - 100  # header + footer space
            scale = min((w * 0.6) / aw, avail_h / ah)
            new_w = int(aw * scale)
            new_h = int(ah * scale)
            resized = cv2.resize(avatar_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            x_off = (w - new_w) // 2
            y_off = 48 + (avail_h - new_h) // 2
            frame[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # ── Header ──
        cv2.rectangle(frame, (0, 0), (w, 44), (16, 16, 16), -1)
        cv2.putText(frame, "Bridge", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Interpreter", (130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

        # Status pill
        if sign:
            pill = f"SIGNING: {sign}"
            pill_col = (60, 200, 80)
        elif speaker and speaker_age < 10:
            pill = "INTERPRETING"
            pill_col = (80, 160, 220)
        else:
            pill = "LISTENING"
            pill_col = (100, 100, 100)

        (pw, _), _ = cv2.getTextSize(pill, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        px = w - pw - 24
        cv2.rectangle(frame, (px - 8, 12), (px + pw + 8, 34), pill_col, -1, cv2.LINE_AA)
        cv2.putText(frame, pill, (px, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Mic + connection dots
        if mic:
            pulse = 4 + int(2 * math.sin(now * 4))
            cv2.circle(frame, (px - 24, 23), pulse, (60, 60, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px - 44, 23), 4, (80, 220, 80), -1, cv2.LINE_AA)

        # ── Footer ──
        cv2.rectangle(frame, (0, h - 50), (w, h), (12, 12, 12), -1)

        if speaker and speaker_age < 10:
            # Show what speaker said
            disp = speaker if len(speaker) <= 60 else speaker[:57] + "..."
            cv2.putText(frame, f'"{disp}"', (20, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if glosses:
                cv2.putText(frame, glosses, (w - 300, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 180, 80), 1, cv2.LINE_AA)

        if queue_len > 0:
            cv2.putText(frame, f"{queue_len} more", (w - 80, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        return frame
