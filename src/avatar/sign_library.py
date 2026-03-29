"""
SignLibrary — loads pre-extracted ASL sign animations from sign_animations.npz.

Each sign is stored as (30, 2, 21, 3):
  - 30 frames (normalized from variable-length originals)
  - 2 hands (index 0=left, 1=right)
  - 21 MediaPipe hand landmarks per hand
  - 3 coords (x, y, z) normalized 0.0-1.0

Data Source: Google Kaggle ASL Signs competition
Extraction: josephzahar's notebook reading logic + our normalization
"""

import json
import os

import numpy as np

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_NPZ = os.path.join(_PROJECT_ROOT, "models", "sign_animations.npz")
_LABELS_PATH = os.path.join(_PROJECT_ROOT, "models", "sign_to_prediction_index_map.json")

# MediaPipe hand connections (21 landmarks, 0-20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm cross
]

# Kaggle parquet structure (from josephzahar notebook):
# Columns: frame, type, landmark_index, x, y, z
# type values: "face", "left_hand", "right_hand", "pose"
# For hands: landmark_index 0-20 (21 MediaPipe hand landmarks)
# Coordinates: normalized 0.0-1.0, y=0 is top of image


_KAGGLE_INSTRUCTIONS = """
=================================================================
  sign_animations.npz NOT FOUND
=================================================================

To generate it from the Kaggle ASL Signs dataset:

1. Go to: kaggle.com/code/josephzahar/interactive-3d-animated-visualization-of-asl
2. Fork the notebook
3. Add the "asl-signs" competition dataset
4. Append this extraction code at the end of the notebook:

   [See src/avatar/sign_library.py for the full extraction code]

5. Run the notebook on Kaggle
6. Download sign_animations.npz from the Output tab
7. Place it at: models/sign_animations.npz

Alternatively, to generate synthetic test data for development:
   python -c "from src.avatar.sign_library import SignLibrary; SignLibrary.generate_synthetic('models/sign_animations.npz')"
=================================================================
"""


class SignLibrary:
    """Loads and serves ASL sign landmark animations."""

    def __init__(self, npz_path: str = _DEFAULT_NPZ):
        self._npz_path = npz_path
        self._signs: dict[str, np.ndarray] = {}
        self._loaded = False

    def load(self) -> bool:
        """Load sign_animations.npz. Returns False if file missing."""
        if not os.path.exists(self._npz_path):
            print(_KAGGLE_INSTRUCTIONS)
            return False

        try:
            data = np.load(self._npz_path, allow_pickle=True)
            sign_arrays = data["sign_arrays"]   # (N, 30, 2, 21, 3)
            sign_names = data["sign_names"]     # (N,) string array

            for name, arr in zip(sign_names, sign_arrays):
                self._signs[str(name).lower()] = arr

            self._loaded = True
            print(f"[sign_library] Loaded {len(self._signs)} sign animations")
            return True

        except Exception as e:
            print(f"[sign_library] Failed to load: {e}")
            return False

    def get(self, sign_name: str) -> np.ndarray | None:
        """Case-insensitive lookup. Returns (30, 2, 21, 3) or None."""
        key = sign_name.strip().lower()
        return self._signs.get(key)

    def has(self, sign_name: str) -> bool:
        return sign_name.strip().lower() in self._signs

    def vocabulary(self) -> list[str]:
        return sorted(self._signs.keys())

    def fingerspell(self, word: str) -> list[str]:
        """Return list of letter signs for a word (fallback for unknowns)."""
        if len(word) > 5:
            return []
        return [ch.upper() for ch in word if ch.isalpha()]

    @staticmethod
    def generate_synthetic(output_path: str = _DEFAULT_NPZ):
        """Generate synthetic sign animations for development/testing.

        Creates fake (30, 2, 21, 3) arrays for all 250 signs
        with recognizable hand motion patterns.
        """
        if not os.path.exists(_LABELS_PATH):
            print(f"[sign_library] Labels file not found: {_LABELS_PATH}")
            return

        with open(_LABELS_PATH) as f:
            sign_to_idx = json.load(f)

        sign_names = list(sign_to_idx.keys())
        n_signs = len(sign_names)
        n_frames = 30

        print(f"[sign_library] Generating synthetic data for {n_signs} signs...")

        all_arrays = np.zeros((n_signs, n_frames, 2, 21, 3), dtype=np.float32)

        for si, name in enumerate(sign_names):
            # Create a unique but plausible hand motion for each sign
            np.random.seed(si)

            for hand in range(2):
                # Base hand position (centered, slightly offset per hand)
                cx = 0.4 if hand == 0 else 0.6
                cy = 0.5

                # Wrist position with sign-specific motion
                freq = 1 + (si % 5) * 0.3
                amp = 0.05 + (si % 7) * 0.01

                for frame in range(n_frames):
                    t = frame / (n_frames - 1)
                    # Wrist motion
                    wx = cx + amp * np.sin(2 * np.pi * freq * t)
                    wy = cy + amp * np.cos(2 * np.pi * freq * t * 0.7)

                    # 21 landmarks relative to wrist
                    for lm in range(21):
                        # Spread fingers based on sign index
                        angle = (lm / 20) * np.pi * (0.5 + 0.5 * np.sin(si + t * np.pi))
                        r = 0.02 + (lm % 4) * 0.015

                        # Some signs have fingers extended, others curled
                        finger_extend = 0.5 + 0.5 * np.sin(si * 0.7 + lm * 0.3 + t * 2)

                        all_arrays[si, frame, hand, lm, 0] = wx + r * np.cos(angle) * finger_extend
                        all_arrays[si, frame, hand, lm, 1] = wy + r * np.sin(angle) * finger_extend
                        all_arrays[si, frame, hand, lm, 2] = 0.5 + 0.01 * np.sin(lm + t)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            sign_arrays=all_arrays,
            sign_names=np.array(sign_names),
            hand_connections=np.array(HAND_CONNECTIONS),
        )

        size_mb = os.path.getsize(output_path) / 1e6
        print(f"[sign_library] Saved {n_signs} signs to {output_path} ({size_mb:.1f} MB)")
        print(f"[sign_library] Shape: {all_arrays.shape}")
