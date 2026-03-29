"""
AvatarRenderer — draws a stick-figure human from (543, 3) landmark data.

Renders upper-body pose skeleton, detailed hand landmarks, and an optional
face oval outline onto a dark canvas using OpenCV primitives.
"""

import cv2
import numpy as np

# Reuse slice constants from holistic_tracker
# Face: 0-467 (468), LHand: 468-488 (21), Pose: 489-521 (33), RHand: 522-542 (21)
_FACE_START = 0
_LHAND_START = 468
_POSE_START = 489
_RHAND_START = 522

# Pose landmark indices (relative to POSE_SLICE, i.e. add _POSE_START for absolute)
_NOSE = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW = 13
_R_ELBOW = 14
_L_WRIST = 15
_R_WRIST = 16
_L_HIP = 23
_R_HIP = 24

# Upper body pose connections (relative to pose start)
_POSE_CONNECTIONS = [
    (_L_SHOULDER, _R_SHOULDER),
    (_L_SHOULDER, _L_ELBOW),
    (_L_ELBOW, _L_WRIST),
    (_R_SHOULDER, _R_ELBOW),
    (_R_ELBOW, _R_WRIST),
    (_L_SHOULDER, _L_HIP),
    (_R_SHOULDER, _R_HIP),
    (_L_HIP, _R_HIP),
]

# MediaPipe hand connections (21 landmarks, indices 0-20)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (note: these are hand-local indices)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm cross-connections
]

# Face oval contour landmark indices (subset of 468 face landmarks)
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Colors (BGR)
_COLOR_BODY = (200, 200, 200)
_COLOR_JOINT = (220, 220, 220)
_COLOR_LHAND = (255, 200, 0)     # cyan-ish
_COLOR_RHAND = (0, 230, 0)      # green
_COLOR_FACE = (80, 80, 80)
_COLOR_LABEL_BG = (40, 40, 40)
_COLOR_LABEL_TEXT = (255, 255, 255)
_COLOR_PROGRESS = (80, 180, 255)


class AvatarRenderer:
    """Renders a stick-figure avatar from MediaPipe (543, 3) landmarks."""

    def __init__(self, width: int = 480, height: int = 640,
                 bg_color: tuple = (20, 20, 20)):
        self._w = width
        self._h = height
        self._bg = bg_color

    def render_frame(self, lm543: np.ndarray, label: str = "",
                     progress: float = 0.0) -> np.ndarray:
        """Render one frame from (543, 3) normalized landmarks. Returns BGR image."""
        canvas = np.full((self._h, self._w, 3), self._bg, dtype=np.uint8)

        # Extract slices
        pose = lm543[_POSE_START: _POSE_START + 33]     # (33, 3)
        lhand = lm543[_LHAND_START: _LHAND_START + 21]  # (21, 3)
        rhand = lm543[_RHAND_START: _RHAND_START + 21]  # (21, 3)
        face = lm543[_FACE_START: _FACE_START + 468]     # (468, 3)

        # Draw in back-to-front order: face → body → hands
        self._draw_face_oval(canvas, face)
        self._draw_pose(canvas, pose)
        self._draw_hand(canvas, lhand, _COLOR_LHAND)
        self._draw_hand(canvas, rhand, _COLOR_RHAND)

        # Label + progress bar at bottom
        if label:
            self._draw_label(canvas, label, progress)

        return canvas

    def render_idle(self, text: str = "Ready") -> np.ndarray:
        """Render a neutral idle frame with centered text."""
        canvas = np.full((self._h, self._w, 3), self._bg, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = (self._w - tw) // 2
        y = (self._h + th) // 2
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2, cv2.LINE_AA)
        return canvas

    def render_text_card(self, gloss: str) -> np.ndarray:
        """Render a text-only card for signs without landmark data."""
        canvas = np.full((self._h, self._w, 3), self._bg, dtype=np.uint8)
        # Sign name large and centered
        (tw, th), _ = cv2.getTextSize(gloss, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        x = (self._w - tw) // 2
        y = (self._h + th) // 2
        cv2.putText(canvas, gloss, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 200, 255), 3, cv2.LINE_AA)
        # Subtitle
        sub = "(no recording)"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, sub, ((self._w - sw) // 2, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        return canvas

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _to_px(self, lm: np.ndarray) -> tuple[int, int] | None:
        """Convert normalized (x, y, z) landmark to pixel coords. Returns None if NaN."""
        if np.isnan(lm[0]) or np.isnan(lm[1]):
            return None
        # Mirror horizontally so avatar faces the viewer
        x = int((1.0 - lm[0]) * self._w)
        y = int(lm[1] * self._h)
        return (x, y)

    def _draw_line(self, canvas: np.ndarray, p1, p2, color, thickness):
        if p1 is not None and p2 is not None:
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    def _draw_pose(self, canvas: np.ndarray, pose: np.ndarray) -> None:
        """Draw upper-body skeleton from 33 pose landmarks."""
        pts = [self._to_px(pose[i]) for i in range(len(pose))]

        # Draw connections
        for i, j in _POSE_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                self._draw_line(canvas, pts[i], pts[j], _COLOR_BODY, 3)

        # Draw joints
        for idx in [_NOSE, _L_SHOULDER, _R_SHOULDER, _L_ELBOW, _R_ELBOW,
                    _L_WRIST, _R_WRIST, _L_HIP, _R_HIP]:
            if idx < len(pts) and pts[idx] is not None:
                cv2.circle(canvas, pts[idx], 5, _COLOR_JOINT, -1, cv2.LINE_AA)

        # Draw head circle around nose
        nose_pt = pts[_NOSE] if _NOSE < len(pts) else None
        if nose_pt is not None:
            # Estimate head size from shoulder distance
            ls = pts[_L_SHOULDER] if _L_SHOULDER < len(pts) else None
            rs = pts[_R_SHOULDER] if _R_SHOULDER < len(pts) else None
            if ls is not None and rs is not None:
                shoulder_dist = abs(ls[0] - rs[0])
                head_r = max(int(shoulder_dist * 0.4), 20)
            else:
                head_r = 30
            cv2.circle(canvas, nose_pt, head_r, _COLOR_BODY, 2, cv2.LINE_AA)

    def _draw_hand(self, canvas: np.ndarray, hand: np.ndarray,
                   color: tuple) -> None:
        """Draw 21 hand landmarks with connections."""
        pts = [self._to_px(hand[i]) for i in range(len(hand))]

        # Check if hand has any valid points
        valid = [p for p in pts if p is not None]
        if len(valid) < 5:
            return

        # Draw connections
        for i, j in _HAND_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                self._draw_line(canvas, pts[i], pts[j], color, 2)

        # Draw joint circles
        for pt in pts:
            if pt is not None:
                cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)

    def _draw_face_oval(self, canvas: np.ndarray, face: np.ndarray) -> None:
        """Draw face outline using oval contour landmarks."""
        points = []
        for idx in _FACE_OVAL:
            if idx < len(face):
                pt = self._to_px(face[idx])
                if pt is not None:
                    points.append(pt)

        if len(points) >= 10:
            pts_arr = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts_arr], isClosed=True,
                          color=_COLOR_FACE, thickness=1, lineType=cv2.LINE_AA)

    def _draw_label(self, canvas: np.ndarray, label: str,
                    progress: float) -> None:
        """Draw sign name and progress bar at the bottom."""
        bar_h = 48
        y_top = self._h - bar_h

        # Semi-transparent background bar
        overlay = canvas[y_top:self._h, :].copy()
        cv2.rectangle(overlay, (0, 0), (self._w, bar_h), _COLOR_LABEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, canvas[y_top:self._h, :], 0.3, 0,
                        canvas[y_top:self._h, :])

        # Label text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = (self._w - tw) // 2
        cv2.putText(canvas, label, (x, self._h - bar_h + th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_LABEL_TEXT, 2, cv2.LINE_AA)

        # Progress bar
        bar_y = self._h - 6
        bar_w = self._w - 40
        cv2.rectangle(canvas, (20, bar_y), (20 + bar_w, bar_y + 3),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            cv2.rectangle(canvas, (20, bar_y), (20 + fill_w, bar_y + 3),
                          _COLOR_PROGRESS, -1)
