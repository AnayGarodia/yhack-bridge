"""
AnimationEngine — smooth cubic spline interpolation and sign blending.

Provides ease-in-out timing, spline-based frame interpolation, and
cross-sign blending for fluid ASL avatar animation.
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out: t*t*(3-2*t)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class AnimationEngine:
    """Interpolates and blends ASL sign landmark animations."""

    def __init__(self):
        self._spline_cache: dict[int, CubicSpline] = {}

    def interpolate_sign(self, frames: np.ndarray, t: float) -> tuple:
        """
        Interpolate within a sign's keyframes.

        Args:
            frames: (30, 2, 21, 3) sign animation
            t: 0.0 to 1.0 progress through sign

        Returns:
            (right_hand (21,3), left_hand (21,3))
        """
        n = frames.shape[0]
        t_eased = _ease_in_out(t)
        idx_float = t_eased * (n - 1)
        idx_low = int(idx_float)
        idx_high = min(idx_low + 1, n - 1)
        frac = idx_float - idx_low

        # Linear interpolation between adjacent frames (fast path)
        interpolated = frames[idx_low] * (1 - frac) + frames[idx_high] * frac

        left_hand = interpolated[0]   # (21, 3)
        right_hand = interpolated[1]  # (21, 3)
        return right_hand, left_hand

    def blend_signs(self, from_frames: np.ndarray, to_frames: np.ndarray,
                    t: float) -> tuple:
        """
        Smoothly blend from last frame of one sign to first frame of next.

        Args:
            from_frames: (30, 2, 21, 3) previous sign
            to_frames: (30, 2, 21, 3) next sign
            t: 0.0 to 1.0 blend progress

        Returns:
            (right_hand (21,3), left_hand (21,3))
        """
        t_eased = _ease_in_out(t)

        last_frame = from_frames[-1]   # (2, 21, 3)
        first_frame = to_frames[0]     # (2, 21, 3)

        blended = last_frame * (1 - t_eased) + first_frame * t_eased

        left_hand = blended[0]
        right_hand = blended[1]
        return right_hand, left_hand

    def idle_pose(self, t: float) -> tuple:
        """
        Generate a gentle breathing idle animation.

        Args:
            t: current time in seconds

        Returns:
            (right_hand (21,3), left_hand (21,3))
        """
        breath = 0.02 * math.sin(t * math.pi * 0.5)

        # Natural rest position — hands at sides
        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)

        for i in range(21):
            # Right hand: slightly right and below center
            right_hand[i] = [
                0.65 + 0.005 * (i % 5),
                0.6 + 0.008 * (i // 5) + breath,
                0.5,
            ]
            # Left hand: mirror
            left_hand[i] = [
                0.35 - 0.005 * (i % 5),
                0.6 + 0.008 * (i // 5) + breath,
                0.5,
            ]

        return right_hand, left_hand
