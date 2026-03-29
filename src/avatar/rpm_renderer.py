"""
RPMRenderer — renders a 3D GLB avatar with pyrender, driven by hand landmarks.

Produces 1280x720 BGR frames at 30fps for the virtual camera pipeline.
Falls back to a skeleton-only renderer if the GLB can't be loaded.
"""

import math
import os
import time

import cv2
import numpy as np

from .landmark_to_bones import landmarks_to_rotations, discover_bones

# Background gradient colors (BGR)
_BG_TOP = np.array([0x2e, 0x1a, 0x1a], dtype=np.uint8)     # #1a1a2e
_BG_BOT = np.array([0x3e, 0x21, 0x16], dtype=np.uint8)     # #16213e

# Hand skeleton connections for fallback rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _make_gradient_bg(width: int, height: int) -> np.ndarray:
    """Create a dark blue vertical gradient background (BGR)."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / height
        bg[y, :] = (1 - t) * _BG_TOP + t * _BG_BOT
    return bg


class RPMRenderer:
    """Renders a 3D avatar or falls back to high-quality skeleton drawing."""

    def __init__(self, glb_path: str = "models/avatar.glb",
                 width: int = 1280, height: int = 720):
        self._glb_path = glb_path
        self._w = width
        self._h = height
        # Render at lower res internally for performance, upscale to output
        self._render_w = 320
        self._render_h = 180
        self._scene = None
        self._renderer = None
        self._bone_names: list[str] = []
        self._use_3d = False
        self._bg = _make_gradient_bg(width, height)
        self._last_frame: np.ndarray | None = None
        self._start_time = time.time()

        # Current pose
        self._right_hand: np.ndarray | None = None
        self._left_hand: np.ndarray | None = None

    def load(self) -> bool:
        """Load GLB avatar. Returns False if 3D rendering unavailable."""
        if not os.path.exists(self._glb_path):
            print(f"[rpm] GLB not found: {self._glb_path}")
            print(f"[rpm] Run: python scripts/setup_rpm_avatar.py")
            print(f"[rpm] Using skeleton fallback renderer")
            return False

        # Discover bone names
        self._bone_names = discover_bones(self._glb_path)

        try:
            # Set platform for headless rendering before importing pyrender
            # EGL works with Homebrew mesa on macOS
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            # Ensure Homebrew libs are on the path
            dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
            if "/opt/homebrew/lib" not in dyld:
                os.environ["DYLD_LIBRARY_PATH"] = f"/opt/homebrew/lib:{dyld}" if dyld else "/opt/homebrew/lib"
            import pyrender
            import trimesh

            scene_data = trimesh.load(self._glb_path)
            self._trimesh_scene = scene_data

            # Build pyrender scene — lighter background so model is visible
            self._pyrender_scene = pyrender.Scene(
                bg_color=[0.12, 0.12, 0.18, 1.0],
                ambient_light=[0.6, 0.6, 0.6],
            )

            # Add meshes WITH their scene graph transforms
            if isinstance(scene_data, trimesh.Scene):
                for name, geom in scene_data.geometry.items():
                    try:
                        transform = scene_data.graph.get(name)[0]
                    except Exception:
                        transform = np.eye(4)
                    mesh = pyrender.Mesh.from_trimesh(geom)
                    self._pyrender_scene.add(mesh, pose=transform)
            else:
                mesh = pyrender.Mesh.from_trimesh(scene_data)
                self._pyrender_scene.add(mesh)

            # Camera — perspective, positioned based on actual model bounds
            cam = pyrender.PerspectiveCamera(yfov=math.radians(45), aspectRatio=self._w / self._h)
            cam_pose = np.eye(4)
            # Model centroid is at roughly (0, 2.2, 0), extents ~6.6 wide
            # Place camera at Z=6 looking at chest height (Y=2.5)
            cam_pose[2, 3] = 6.0    # 6 units in front
            cam_pose[1, 3] = 2.5    # chest height of model
            self._pyrender_scene.add(cam, pose=cam_pose)

            # 3-point lighting (scaled to model size)
            key = pyrender.PointLight(color=[1.0, 0.95, 0.9], intensity=30.0)
            key_pose = np.eye(4)
            key_pose[:3, 3] = [-3.0, 4.0, 5.0]
            self._pyrender_scene.add(key, pose=key_pose)

            fill = pyrender.PointLight(color=[0.9, 0.9, 1.0], intensity=15.0)
            fill_pose = np.eye(4)
            fill_pose[:3, 3] = [3.0, 3.0, 4.0]
            self._pyrender_scene.add(fill, pose=fill_pose)

            rim = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)
            rim_pose = np.eye(4)
            rim_pose[:3, 3] = [-2.0, 4.0, -3.0]
            self._pyrender_scene.add(rim, pose=rim_pose)

            # Offscreen renderer at reduced resolution for performance
            self._renderer = pyrender.OffscreenRenderer(self._render_w, self._render_h)
            self._use_3d = True
            print(f"[rpm] 3D renderer ready ({self._w}x{self._h})")
            return True

        except Exception as e:
            print(f"[rpm] 3D renderer failed: {e}")
            print(f"[rpm] Using skeleton fallback renderer")
            self._use_3d = False
            return False

    def set_pose(self, right_hand: np.ndarray, left_hand: np.ndarray):
        """Set current hand pose. right_hand/left_hand: (21, 3)."""
        self._right_hand = right_hand
        self._left_hand = left_hand

    def render(self) -> np.ndarray:
        """Render current pose to BGR frame. Must complete in <33ms."""
        t0 = time.perf_counter()

        if self._use_3d:
            frame = self._render_3d()
        else:
            frame = self._render_skeleton()

        self._last_frame = frame

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 100:
            print(f"[rpm] slow render: {elapsed_ms:.0f}ms")

        return frame

    def render_idle(self) -> np.ndarray:
        """Render neutral idle pose with subtle breathing."""
        t = time.time() - self._start_time
        breath = 0.002 * math.sin(t * math.pi)  # subtle oscillation

        # Create a neutral hand pose
        neutral = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            neutral[i] = [0.5, 0.5 + i * 0.005 + breath, 0.5]

        self.set_pose(neutral, neutral)
        frame = self.render()

        # Add "Idle" text
        cv2.putText(frame, "Listening...", (self._w // 2 - 80, self._h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # 3D Rendering (pyrender)
    # ------------------------------------------------------------------

    def _render_3d(self) -> np.ndarray:
        """Render the 3D avatar with current hand pose."""
        try:
            # TODO: Apply bone rotations from landmarks_to_rotations()
            # For now, just render the static model
            color, _ = self._renderer.render(self._pyrender_scene)
            # pyrender returns RGB, convert to BGR and upscale to output resolution
            frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            if frame.shape[1] != self._w or frame.shape[0] != self._h:
                frame = cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
            return frame
        except Exception as e:
            if self._last_frame is not None:
                return self._last_frame
            return self._bg.copy()

    # ------------------------------------------------------------------
    # Skeleton Fallback Rendering (OpenCV)
    # ------------------------------------------------------------------

    def _render_skeleton(self) -> np.ndarray:
        """High-quality skeleton rendering as fallback when 3D is unavailable."""
        frame = self._bg.copy()
        h, w = frame.shape[:2]

        # Draw a simple body silhouette
        body_cx, body_cy = w // 2, int(h * 0.35)
        shoulder_w = int(w * 0.15)

        # Head
        head_r = int(shoulder_w * 0.6)
        head_cy = body_cy - int(shoulder_w * 0.8)
        cv2.circle(frame, (body_cx, head_cy), head_r, (180, 200, 220), -1, cv2.LINE_AA)
        cv2.circle(frame, (body_cx, head_cy), head_r, (100, 120, 140), 2, cv2.LINE_AA)

        # Eyes
        eye_y = head_cy - int(head_r * 0.05)
        for ex in [body_cx - int(head_r * 0.3), body_cx + int(head_r * 0.3)]:
            cv2.circle(frame, (ex, eye_y), int(head_r * 0.12), (240, 240, 240), -1)
            cv2.circle(frame, (ex, eye_y), int(head_r * 0.06), (50, 40, 35), -1)

        # Smile
        cv2.ellipse(frame, (body_cx, head_cy + int(head_r * 0.3)),
                    (int(head_r * 0.3), int(head_r * 0.12)), 0, 10, 170,
                    (140, 120, 180), 2, cv2.LINE_AA)

        # Torso
        torso_top = body_cy
        torso_bot = body_cy + int(shoulder_w * 1.5)
        pts = np.array([
            [body_cx - shoulder_w, torso_top],
            [body_cx + shoulder_w, torso_top],
            [body_cx + int(shoulder_w * 0.8), torso_bot],
            [body_cx - int(shoulder_w * 0.8), torso_bot],
        ], dtype=np.int32)
        cv2.fillConvexPoly(frame, pts, (160, 100, 50), cv2.LINE_AA)

        # Neck
        cv2.rectangle(frame, (body_cx - int(shoulder_w * 0.15), head_cy + head_r - 3),
                      (body_cx + int(shoulder_w * 0.15), torso_top + 3),
                      (180, 200, 220), -1)

        # Draw hands
        if self._right_hand is not None:
            self._draw_hand_skeleton(frame, self._right_hand, (0, 230, 120),
                                     offset_x=w * 0.65, offset_y=h * 0.25, scale=w * 0.35)
        if self._left_hand is not None:
            self._draw_hand_skeleton(frame, self._left_hand, (230, 180, 0),
                                     offset_x=w * 0.0, offset_y=h * 0.25, scale=w * 0.35)

        return frame

    def _draw_hand_skeleton(self, frame: np.ndarray, hand: np.ndarray,
                            color: tuple, offset_x: float, offset_y: float,
                            scale: float):
        """Draw a styled hand skeleton from (21, 3) landmarks."""
        h, w = frame.shape[:2]
        pts = []
        for i in range(21):
            x = hand[i, 0]
            y = hand[i, 1]
            if np.isnan(x) or np.isnan(y):
                pts.append(None)
                continue
            px = int(offset_x + x * scale)
            py = int(offset_y + y * scale)
            pts.append((px, py))

        # Draw connections as thick capsule lines
        for i, j in HAND_CONNECTIONS:
            if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                cv2.line(frame, pts[i], pts[j], color, 3, cv2.LINE_AA)

        # Draw joints
        darker = tuple(max(0, c - 40) for c in color)
        for pt in pts:
            if pt is not None:
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, darker, 1, cv2.LINE_AA)

        # Fingertips highlighted
        for tip_idx in [4, 8, 12, 16, 20]:
            if tip_idx < len(pts) and pts[tip_idx]:
                cv2.circle(frame, pts[tip_idx], 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pts[tip_idx], 7, color, 2, cv2.LINE_AA)
