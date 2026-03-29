"""
AvatarRenderer — Bitmoji-style cartoon character from (543, 3) landmark data.

Renders a friendly cartoon avatar with filled body, arms, and detailed hands
using OpenCV primitives. Designed for real-time ASL sign display at 30fps.
"""

import math

import cv2
import numpy as np

# Landmark layout offsets
_LHAND_START = 468
_POSE_START = 489
_RHAND_START = 522

# Pose landmark indices (relative to POSE_SLICE)
_NOSE = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW = 13
_R_ELBOW = 14
_L_WRIST = 15
_R_WRIST = 16
_L_HIP = 23
_R_HIP = 24

# Finger landmark chains (hand-local indices 0-20)
_FINGER_CHAINS = [
    [1, 2, 3, 4],       # thumb
    [5, 6, 7, 8],       # index
    [9, 10, 11, 12],    # middle
    [13, 14, 15, 16],   # ring
    [17, 18, 19, 20],   # pinky
]
_PALM_INDICES = [0, 5, 9, 13, 17]
_THUMB_BRIDGE = [0, 1, 5]
_TIP_INDICES = [4, 8, 12, 16, 20]

# ── Color palette (BGR) ──────────────────────────────────────────────────────
_SKIN       = (140, 180, 225)
_SKIN_DARK  = (110, 150, 195)
_SKIN_LIGHT = (165, 205, 240)
_SHIRT      = (180, 120, 60)
_SHIRT_DARK = (150, 95, 40)
_HAIR       = (50, 40, 35)
_EYE_WHITE  = (240, 240, 240)
_EYE_IRIS   = (80, 60, 40)
_EYE_PUPIL  = (25, 25, 25)
_EYEBROW    = (55, 45, 40)
_MOUTH      = (100, 110, 190)
_NAIL       = (200, 210, 220)
_LABEL_BG   = (40, 40, 40)
_LABEL_TEXT  = (255, 255, 255)
_PROGRESS   = (80, 180, 255)


class AvatarRenderer:
    """Renders a Bitmoji-style cartoon avatar from MediaPipe (543, 3) landmarks."""

    def __init__(self, width: int = 480, height: int = 640,
                 bg_color: tuple = (20, 20, 20)):
        self._w = width
        self._h = height
        self._bg = bg_color
        self._idle_lm = self._make_idle_landmarks()
        self._idle_cache: np.ndarray | None = None
        self._idle_text: str = ""

    # ──────────────────────────────────────────────────────────────────────
    # Public API (unchanged signatures)
    # ──────────────────────────────────────────────────────────────────────

    def render_frame(self, lm543: np.ndarray, label: str = "",
                     progress: float = 0.0) -> np.ndarray:
        """Render one frame from (543, 3) normalized landmarks. Returns BGR image."""
        canvas = np.full((self._h, self._w, 3), self._bg, dtype=np.uint8)

        pose = lm543[_POSE_START: _POSE_START + 33]
        lhand = lm543[_LHAND_START: _LHAND_START + 21]
        rhand = lm543[_RHAND_START: _RHAND_START + 21]

        # Pose pixel coords
        pose_pts = [self._to_px(pose[i]) for i in range(min(len(pose), 33))]
        ls = pose_pts[_L_SHOULDER] if _L_SHOULDER < len(pose_pts) else None
        rs = pose_pts[_R_SHOULDER] if _R_SHOULDER < len(pose_pts) else None
        sw = self._distance(ls, rs) if ls and rs else self._w * 0.25

        # Z-depth for arm layering
        lw_z = pose[_L_WRIST][2] if _L_WRIST < len(pose) and not np.isnan(pose[_L_WRIST][2]) else 0
        rw_z = pose[_R_WRIST][2] if _R_WRIST < len(pose) and not np.isnan(pose[_R_WRIST][2]) else 0

        # Define arms
        l_arm = (pose_pts[_L_SHOULDER] if _L_SHOULDER < len(pose_pts) else None,
                 pose_pts[_L_ELBOW] if _L_ELBOW < len(pose_pts) else None,
                 pose_pts[_L_WRIST] if _L_WRIST < len(pose_pts) else None)
        r_arm = (pose_pts[_R_SHOULDER] if _R_SHOULDER < len(pose_pts) else None,
                 pose_pts[_R_ELBOW] if _R_ELBOW < len(pose_pts) else None,
                 pose_pts[_R_WRIST] if _R_WRIST < len(pose_pts) else None)

        if lw_z >= rw_z:
            back_arm, front_arm = l_arm, r_arm
            back_hand, front_hand = lhand, rhand
        else:
            back_arm, front_arm = r_arm, l_arm
            back_hand, front_hand = rhand, lhand

        # Draw order: torso → back arm+hand → head → front arm+hand → label
        self._draw_torso(canvas, pose_pts, sw)
        self._draw_arm(canvas, *back_arm, sw)
        self._draw_hand_cartoon(canvas, back_hand, sw)
        self._draw_head(canvas, pose_pts, sw)
        self._draw_arm(canvas, *front_arm, sw)
        self._draw_hand_cartoon(canvas, front_hand, sw)

        if label:
            self._draw_label(canvas, label, progress)

        return canvas

    def render_idle(self, text: str = "Ready") -> np.ndarray:
        """Render the character in a neutral rest pose with label."""
        if self._idle_cache is not None and self._idle_text == text:
            return self._idle_cache.copy()
        frame = self.render_frame(self._idle_lm, label=text, progress=0.0)
        self._idle_cache = frame.copy()
        self._idle_text = text
        return frame

    def render_text_card(self, gloss: str) -> np.ndarray:
        """Render idle character with sign name overlaid."""
        canvas = self.render_frame(self._idle_lm, label="", progress=0.0)
        h, w = canvas.shape[:2]

        # Semi-transparent overlay panel
        y1, y2 = h // 2 - 40, h // 2 + 50
        overlay = canvas[y1:y2, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, y2 - y1), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, canvas[y1:y2, :], 0.25, 0, canvas[y1:y2, :])

        # Sign name
        (tw, th), _ = cv2.getTextSize(gloss, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(canvas, gloss, ((w - tw) // 2, h // 2 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3, cv2.LINE_AA)

        return canvas

    # ──────────────────────────────────────────────────────────────────────
    # Coordinate helpers
    # ──────────────────────────────────────────────────────────────────────

    def _to_px(self, lm: np.ndarray) -> tuple[int, int] | None:
        if np.isnan(lm[0]) or np.isnan(lm[1]):
            return None
        x = int((1.0 - lm[0]) * self._w)
        y = int(lm[1] * self._h)
        return (x, y)

    @staticmethod
    def _midpoint(p1, p2):
        if p1 is None or p2 is None:
            return None
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    @staticmethod
    def _distance(p1, p2) -> float:
        if p1 is None or p2 is None:
            return 0.0
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _lerp(p1, p2, t: float):
        if p1 is None or p2 is None:
            return None
        return (int(p1[0] + (p2[0] - p1[0]) * t),
                int(p1[1] + (p2[1] - p1[1]) * t))

    # ──────────────────────────────────────────────────────────────────────
    # Drawing primitives
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_capsule(canvas, p1, p2, width, color):
        """Draw a filled capsule (rounded thick segment) between two points."""
        if p1 is None or p2 is None:
            return
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(1, int(math.sqrt(dx * dx + dy * dy)))
        hw = max(1, width // 2)
        nx = -dy / length
        ny = dx / length

        pts = np.array([
            [int(p1[0] + nx * hw), int(p1[1] + ny * hw)],
            [int(p1[0] - nx * hw), int(p1[1] - ny * hw)],
            [int(p2[0] - nx * hw), int(p2[1] - ny * hw)],
            [int(p2[0] + nx * hw), int(p2[1] + ny * hw)],
        ], dtype=np.int32)

        cv2.fillConvexPoly(canvas, pts, color, cv2.LINE_AA)
        cv2.circle(canvas, p1, hw, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, hw, color, -1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────
    # Body part renderers
    # ──────────────────────────────────────────────────────────────────────

    def _draw_head(self, canvas, pose_pts, sw):
        """Draw cartoon head with face features above the shoulders."""
        ls = pose_pts[_L_SHOULDER] if _L_SHOULDER < len(pose_pts) else None
        rs = pose_pts[_R_SHOULDER] if _R_SHOULDER < len(pose_pts) else None
        mid = self._midpoint(ls, rs)
        if mid is None:
            mid = (self._w // 2, int(self._h * 0.28))

        cx = mid[0]
        cy = mid[1] - int(sw * 0.75)
        rx = max(18, int(sw * 0.42))
        ry = max(22, int(sw * 0.52))

        # Neck
        neck_w = max(8, int(sw * 0.15))
        cv2.rectangle(canvas,
                      (cx - neck_w, cy + ry - 4),
                      (cx + neck_w, mid[1]),
                      _SKIN, -1)

        # Head shape
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, _SKIN, -1, cv2.LINE_AA)
        # Outline
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, _SKIN_DARK, 2, cv2.LINE_AA)

        # Hair cap
        hair_ry = max(12, int(ry * 0.55))
        cv2.ellipse(canvas, (cx, cy - int(ry * 0.15)),
                    (rx + 2, hair_ry), 0, 180, 360, _HAIR, -1, cv2.LINE_AA)

        # Eyes
        eye_y = cy - int(ry * 0.08)
        eye_sep = int(rx * 0.38)
        eye_rx = max(3, int(rx * 0.15))
        eye_ry = max(4, int(ry * 0.12))

        for ex in [cx - eye_sep, cx + eye_sep]:
            # White
            cv2.ellipse(canvas, (ex, eye_y), (eye_rx, eye_ry),
                        0, 0, 360, _EYE_WHITE, -1, cv2.LINE_AA)
            # Iris
            iris_r = max(2, int(eye_rx * 0.6))
            cv2.circle(canvas, (ex, eye_y), iris_r, _EYE_IRIS, -1, cv2.LINE_AA)
            # Pupil
            pupil_r = max(1, int(iris_r * 0.5))
            cv2.circle(canvas, (ex, eye_y), pupil_r, _EYE_PUPIL, -1, cv2.LINE_AA)
            # Highlight
            cv2.circle(canvas, (ex - 1, eye_y - 1), max(1, pupil_r), _EYE_WHITE, -1, cv2.LINE_AA)

        # Eyebrows
        brow_y = eye_y - int(ry * 0.15)
        brow_w = max(4, int(rx * 0.2))
        for bx in [cx - eye_sep, cx + eye_sep]:
            cv2.ellipse(canvas, (bx, brow_y), (brow_w, max(2, int(ry * 0.04))),
                        0, 200, 340, _EYEBROW, max(1, int(sw * 0.02)), cv2.LINE_AA)

        # Smile
        mouth_y = cy + int(ry * 0.32)
        mouth_w = max(5, int(rx * 0.35))
        mouth_h = max(3, int(ry * 0.12))
        cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, mouth_h),
                    0, 10, 170, _MOUTH, max(1, int(sw * 0.02)), cv2.LINE_AA)

    def _draw_torso(self, canvas, pose_pts, sw):
        """Draw filled shirt trapezoid from shoulders to hips."""
        ls = pose_pts[_L_SHOULDER] if _L_SHOULDER < len(pose_pts) else None
        rs = pose_pts[_R_SHOULDER] if _R_SHOULDER < len(pose_pts) else None
        lh = pose_pts[_L_HIP] if _L_HIP < len(pose_pts) else None
        rh = pose_pts[_R_HIP] if _R_HIP < len(pose_pts) else None

        if ls is None or rs is None:
            return

        # Estimate hips if missing
        if lh is None:
            lh = (ls[0] + int(sw * 0.05), ls[1] + int(sw * 1.2))
        if rh is None:
            rh = (rs[0] - int(sw * 0.05), rs[1] + int(sw * 1.2))

        pad = max(4, int(sw * 0.1))
        pts = np.array([
            [ls[0] - pad, ls[1]],
            [rs[0] + pad, rs[1]],
            [rh[0] + pad // 2, rh[1]],
            [lh[0] - pad // 2, lh[1]],
        ], dtype=np.int32)

        cv2.fillConvexPoly(canvas, pts, _SHIRT, cv2.LINE_AA)

        # Collar V-neckline
        mid = self._midpoint(ls, rs)
        if mid:
            collar_depth = int(sw * 0.12)
            cv2.line(canvas, (mid[0], mid[1] - 2), (mid[0] - int(sw * 0.08), mid[1] + collar_depth),
                     _SHIRT_DARK, max(1, int(sw * 0.02)), cv2.LINE_AA)
            cv2.line(canvas, (mid[0], mid[1] - 2), (mid[0] + int(sw * 0.08), mid[1] + collar_depth),
                     _SHIRT_DARK, max(1, int(sw * 0.02)), cv2.LINE_AA)

    def _draw_arm(self, canvas, shoulder, elbow, wrist, sw):
        """Draw a two-segment arm with short sleeve."""
        if shoulder is None:
            return

        ua_w = max(6, int(sw * 0.18))
        fa_w = max(5, int(sw * 0.14))

        if elbow is not None:
            # Sleeve: top 25% of upper arm in shirt color
            sleeve_end = self._lerp(shoulder, elbow, 0.25)
            self._draw_capsule(canvas, shoulder, sleeve_end, ua_w + 2, _SHIRT)
            # Upper arm skin
            self._draw_capsule(canvas, sleeve_end, elbow, ua_w, _SKIN)
            # Smooth the joint
            cv2.circle(canvas, elbow, ua_w // 2, _SKIN, -1, cv2.LINE_AA)
        elif wrist is not None:
            # No elbow detected — draw straight to wrist
            self._draw_capsule(canvas, shoulder, wrist, ua_w, _SKIN)
            return

        if elbow is not None and wrist is not None:
            # Forearm
            self._draw_capsule(canvas, elbow, wrist, fa_w, _SKIN)

    def _draw_hand_cartoon(self, canvas, hand_21, sw):
        """Draw a cartoon hand with palm fill and tapered finger segments."""
        pts = [self._to_px(hand_21[i]) for i in range(min(len(hand_21), 21))]
        valid = [p for p in pts if p is not None]
        if len(valid) < 5:
            return

        # Estimate hand scale
        wrist = pts[0]
        mid_mcp = pts[9] if len(pts) > 9 else None
        hand_scale = self._distance(wrist, mid_mcp) if wrist and mid_mcp else max(20, sw * 0.3)

        # Base/tip finger widths
        base_w = max(4, int(hand_scale * 0.24))
        tip_w = max(2, int(hand_scale * 0.14))

        # Palm fill
        self._draw_palm(canvas, pts)

        # Fingers — filled capsule segments
        for chain in _FINGER_CHAINS:
            n_seg = len(chain) - 1
            for si in range(n_seg):
                if chain[si] >= len(pts) or chain[si + 1] >= len(pts):
                    continue
                pa = pts[chain[si]]
                pb = pts[chain[si + 1]]
                if pa is None or pb is None:
                    continue
                # Taper width
                t = si / n_seg
                w = int(base_w * (1.0 - t) + tip_w * t)
                w = max(2, w)
                self._draw_capsule(canvas, pa, pb, w, _SKIN)

        # Fingertip highlights
        for ti in _TIP_INDICES:
            if ti < len(pts) and pts[ti] is not None:
                r = max(1, tip_w // 2)
                cv2.circle(canvas, pts[ti], r, _SKIN_LIGHT, -1, cv2.LINE_AA)

        # Fingernail detail
        for fi, ti in enumerate(_TIP_INDICES):
            chain = _FINGER_CHAINS[fi]
            prev_i = chain[-2]
            if ti < len(pts) and prev_i < len(pts) and pts[ti] and pts[prev_i]:
                dx = pts[ti][0] - pts[prev_i][0]
                dy = pts[ti][1] - pts[prev_i][1]
                angle = int(math.degrees(math.atan2(dy, dx)))
                nail_r = max(1, tip_w // 3)
                cv2.ellipse(canvas, pts[ti], (nail_r, nail_r), angle,
                            -90, 90, _NAIL, -1, cv2.LINE_AA)

        # Outline pass for definition
        for chain in _FINGER_CHAINS:
            for si in range(len(chain) - 1):
                if chain[si] >= len(pts) or chain[si + 1] >= len(pts):
                    continue
                pa = pts[chain[si]]
                pb = pts[chain[si + 1]]
                if pa and pb:
                    cv2.line(canvas, pa, pb, _SKIN_DARK, 1, cv2.LINE_AA)

    def _draw_palm(self, canvas, hand_pts):
        """Fill the palm area using convex hull of wrist + MCP landmarks."""
        palm_pts = []
        for idx in _PALM_INDICES:
            if idx < len(hand_pts) and hand_pts[idx] is not None:
                palm_pts.append(hand_pts[idx])

        if len(palm_pts) >= 3:
            arr = np.array(palm_pts, dtype=np.int32)
            hull = cv2.convexHull(arr)
            cv2.fillConvexPoly(canvas, hull, _SKIN, cv2.LINE_AA)

        # Thumb bridge (fill gap between thumb and index base)
        bridge_pts = []
        for idx in _THUMB_BRIDGE:
            if idx < len(hand_pts) and hand_pts[idx] is not None:
                bridge_pts.append(hand_pts[idx])
        if len(bridge_pts) == 3:
            arr = np.array(bridge_pts, dtype=np.int32)
            cv2.fillConvexPoly(canvas, arr, _SKIN, cv2.LINE_AA)

    def _draw_label(self, canvas, label: str, progress: float):
        """Draw sign name and progress bar at the bottom."""
        bar_h = 48
        y_top = self._h - bar_h

        overlay = canvas[y_top:self._h, :].copy()
        cv2.rectangle(overlay, (0, 0), (self._w, bar_h), _LABEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, canvas[y_top:self._h, :], 0.3, 0,
                        canvas[y_top:self._h, :])

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = (self._w - tw) // 2
        cv2.putText(canvas, label, (x, self._h - bar_h + th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _LABEL_TEXT, 2, cv2.LINE_AA)

        bar_y = self._h - 6
        bar_w = self._w - 40
        cv2.rectangle(canvas, (20, bar_y), (20 + bar_w, bar_y + 3), (60, 60, 60), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            cv2.rectangle(canvas, (20, bar_y), (20 + fill_w, bar_y + 3), _PROGRESS, -1)

    # ──────────────────────────────────────────────────────────────────────
    # Idle pose
    # ──────────────────────────────────────────────────────────────────────

    def _make_idle_landmarks(self) -> np.ndarray:
        """Create synthetic (543, 3) landmarks for a neutral standing pose."""
        lm = np.full((543, 3), np.nan, dtype=np.float32)
        cx, cy = 0.5, 0.33
        sw = 0.22

        lm[_POSE_START + _NOSE]        = [cx, cy, 0]
        lm[_POSE_START + _L_SHOULDER]  = [cx - sw / 2, cy + 0.12, 0]
        lm[_POSE_START + _R_SHOULDER]  = [cx + sw / 2, cy + 0.12, 0]
        lm[_POSE_START + _L_ELBOW]     = [cx - sw / 2 - 0.02, cy + 0.26, 0]
        lm[_POSE_START + _R_ELBOW]     = [cx + sw / 2 + 0.02, cy + 0.26, 0]
        lm[_POSE_START + _L_WRIST]     = [cx - sw / 2 - 0.01, cy + 0.39, 0]
        lm[_POSE_START + _R_WRIST]     = [cx + sw / 2 + 0.01, cy + 0.39, 0]
        lm[_POSE_START + _L_HIP]       = [cx - sw * 0.38, cy + 0.36, 0]
        lm[_POSE_START + _R_HIP]       = [cx + sw * 0.38, cy + 0.36, 0]

        # Hands left as NaN — arms hang at sides, no hand detail in idle
        return lm
