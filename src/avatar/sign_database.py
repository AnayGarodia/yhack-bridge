"""
SignDatabase — load/save/lookup pre-recorded ASL sign landmark sequences.

Signs are stored as .npy files:
  data/signs/hello.npy   → shape (T, 543, 3)  — variable-length sequences
  data/letters/A.npy     → shape (1, 543, 3)  — single-frame static poses
"""

import os
import logging

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_SIGNS_DIR = os.path.join(_PROJECT_ROOT, "data", "signs")
_DEFAULT_LETTERS_DIR = os.path.join(_PROJECT_ROOT, "data", "letters")


class SignDatabase:
    """Stores and retrieves pre-recorded (T, 543, 3) landmark sequences per sign."""

    def __init__(self, signs_dir: str = _DEFAULT_SIGNS_DIR,
                 letters_dir: str = _DEFAULT_LETTERS_DIR):
        self._signs_dir = signs_dir
        self._letters_dir = letters_dir
        self._signs: dict[str, np.ndarray] = {}
        self._letters: dict[str, np.ndarray] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all .npy files from signs and letters directories into memory."""
        for d, store, label in [
            (self._signs_dir, self._signs, "signs"),
            (self._letters_dir, self._letters, "letters"),
        ]:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".npy"):
                    continue
                name = fname[:-4]  # strip .npy
                path = os.path.join(d, fname)
                try:
                    arr = np.load(path)
                    if arr.ndim == 2:
                        arr = arr[np.newaxis]  # (543,3) → (1,543,3)
                    store[name.lower()] = arr
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)
        logger.info(
            "SignDatabase loaded: %d signs, %d letters",
            len(self._signs), len(self._letters),
        )

    def get(self, gloss: str) -> np.ndarray | None:
        """Look up a sign or letter by name (case-insensitive). Returns (T,543,3) or None."""
        key = gloss.strip().lower()
        if key in self._signs:
            return self._signs[key]
        if key in self._letters:
            return self._letters[key]
        return None

    def has(self, gloss: str) -> bool:
        key = gloss.strip().lower()
        return key in self._signs or key in self._letters

    def save_sign(self, name: str, frames: np.ndarray) -> None:
        """Save a recorded sign sequence. frames shape: (T, 543, 3)."""
        os.makedirs(self._signs_dir, exist_ok=True)
        key = name.strip().lower()
        path = os.path.join(self._signs_dir, f"{key}.npy")
        np.save(path, frames.astype(np.float32))
        self._signs[key] = frames
        logger.info("Saved sign '%s' (%d frames) → %s", key, len(frames), path)

    def save_letter(self, letter: str, frame: np.ndarray) -> None:
        """Save a single-frame letter pose. frame shape: (543, 3) or (1, 543, 3)."""
        os.makedirs(self._letters_dir, exist_ok=True)
        key = letter.strip().upper()
        if frame.ndim == 2:
            frame = frame[np.newaxis]
        path = os.path.join(self._letters_dir, f"{key}.npy")
        np.save(path, frame.astype(np.float32))
        self._letters[key.lower()] = frame
        logger.info("Saved letter '%s' → %s", key, path)

    @property
    def available_signs(self) -> list[str]:
        return sorted(self._signs.keys())

    @property
    def available_letters(self) -> list[str]:
        return sorted(self._letters.keys())

    def summary(self) -> str:
        parts = [f"Signs ({len(self._signs)}): {', '.join(self.available_signs) or 'none'}"]
        parts.append(f"Letters ({len(self._letters)}): {', '.join(self.available_letters) or 'none'}")
        return "\n".join(parts)
