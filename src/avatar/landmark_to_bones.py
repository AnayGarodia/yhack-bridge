"""
Landmark to Bone Rotation Converter.

Converts MediaPipe hand landmark (21, 3) arrays to bone rotation quaternions
for driving a 3D avatar skeleton.

Uses scipy.spatial.transform.Rotation for stable quaternion math.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# MediaPipe finger chains: parent_landmark → child_landmark
# Each tuple: (chain of landmark indices from base to tip)
FINGER_CHAINS = {
    "Thumb":  [0, 1, 2, 3, 4],
    "Index":  [0, 5, 6, 7, 8],
    "Middle": [0, 9, 10, 11, 12],
    "Ring":   [0, 13, 14, 15, 16],
    "Pinky":  [0, 17, 18, 19, 20],
}

# Bone segment: (parent_lm_idx, child_lm_idx, bone_segment_name)
# The bone name is what we'll map to the actual skeleton
BONE_SEGMENTS_RIGHT = [
    # Thumb
    (0, 1, "Thumb.R"),     (1, 2, "Thumb2.R"),
    # Index
    (0, 5, "Palm1.R"),     (5, 6, "Index.R"),    (6, 7, "Index2.R"),
    # Middle
    (0, 9, "Palm2.R"),     (9, 10, "Middle1.R"),  (10, 11, "Middle2.R"),
    # Ring
    (0, 13, "Palm3.R"),    (13, 14, "Ring1.R"),   (14, 15, "Ring2.R"),
    # Pinky — mapped to palm/hand
    (0, 17, "Hand.R"),
]

BONE_SEGMENTS_LEFT = [
    (0, 1, "Thumb.L"),     (1, 2, "Thumb2.L"),
    (0, 5, "Palm1.L"),     (5, 6, "Index.L"),    (6, 7, "Index2.L"),
    (0, 9, "Palm2.L"),     (9, 10, "Middle1.L"),  (10, 11, "Middle2.L"),
    (0, 13, "Palm3.L"),    (13, 14, "Ring1.L"),   (14, 15, "Ring2.L"),
    (0, 17, "Hand.L"),
]

# Rest direction: MediaPipe hand in neutral pose points roughly along +Y
_REST_DIR = np.array([0, 1, 0], dtype=np.float64)


def _direction(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Unit direction vector from p1 to p2."""
    d = p2 - p1
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return _REST_DIR.copy()
    return d / norm


def _quat_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) that rotates v1 to align with v2."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if dot > 0.9999:
        return np.array([1, 0, 0, 0], dtype=np.float64)  # identity
    if dot < -0.9999:
        # 180-degree rotation around any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0, axis[0], axis[1], axis[2]], dtype=np.float64)

    w = 1 + dot
    q = np.array([w, cross[0], cross[1], cross[2]], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def landmarks_to_rotations(hand_landmarks: np.ndarray, side: str = "right") -> dict:
    """
    Convert (21, 3) MediaPipe hand landmarks to bone rotation quaternions.

    Args:
        hand_landmarks: (21, 3) normalized coordinates
        side: "right" or "left"

    Returns:
        dict mapping bone_name → quaternion (w, x, y, z) as np.ndarray
    """
    lm = hand_landmarks.astype(np.float64)
    segments = BONE_SEGMENTS_RIGHT if side == "right" else BONE_SEGMENTS_LEFT
    rotations = {}

    for parent_idx, child_idx, bone_name in segments:
        p = lm[parent_idx]
        c = lm[child_idx]

        if np.any(np.isnan(p)) or np.any(np.isnan(c)):
            rotations[bone_name] = np.array([1, 0, 0, 0], dtype=np.float64)
            continue

        direction = _direction(p, c)
        q = _quat_from_two_vectors(_REST_DIR, direction)
        rotations[bone_name] = q

    return rotations


def discover_bones(glb_path: str) -> list[str]:
    """Load a GLB file and print all node/bone names found."""
    import trimesh
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        nodes = sorted(scene.graph.nodes)
        print(f"[bones] Found {len(nodes)} nodes in {glb_path}:")
        for n in nodes:
            print(f"  {n}")
        return nodes
    print(f"[bones] Single mesh, no skeleton found in {glb_path}")
    return []
