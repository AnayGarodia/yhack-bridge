#!/usr/bin/env python3
"""
Sign Recorder — record ASL signs via webcam into the sign database.

Usage:
    python -m src.avatar.recorder --letters                    # record A-Z
    python -m src.avatar.recorder --sign hello                 # record one sign
    python -m src.avatar.recorder --batch hello,thankyou,yes   # record multiple
    python -m src.avatar.recorder --list                       # show database
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.recognition.holistic_tracker import HolisticTracker
from src.avatar.sign_database import SignDatabase


class SignRecorder:
    """Interactive sign recording tool using webcam + MediaPipe Holistic."""

    def __init__(self, camera_index: int = 0):
        self._camera_index = camera_index
        self._tracker = HolisticTracker()
        self._db = SignDatabase()
        self._window = "Bridge Sign Recorder"

    def record_sign(self, name: str) -> np.ndarray | None:
        """Record a multi-frame sign interactively. Returns (T, 543, 3) or None."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {self._camera_index}")
            return None

        frames: list[np.ndarray] = []
        recording = False
        countdown = 0
        countdown_start = 0.0
        saved = False

        print(f"\n  Recording sign: '{name}'")
        print("  SPACE = start/stop recording  |  s = save  |  r = redo  |  q = skip\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            annotated, lm543, *_ = self._tracker.process_frame(frame)
            display = annotated.copy()
            h, w = display.shape[:2]

            # Countdown
            if countdown > 0:
                elapsed = time.time() - countdown_start
                remaining = countdown - int(elapsed)
                if remaining > 0:
                    cv2.putText(display, str(remaining), (w // 2 - 30, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5, cv2.LINE_AA)
                else:
                    countdown = 0
                    recording = True
                    frames = []
                    print("  >> Recording... (press SPACE to stop)")

            # Recording indicator
            if recording:
                frames.append(lm543.copy())
                cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(display, f"REC  {len(frames)} frames", (50, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Sign name
            cv2.putText(display, f"Sign: {name}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # Status
            if not recording and not countdown and not saved:
                cv2.putText(display, "Press SPACE to start", (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            if saved:
                cv2.putText(display, f"SAVED ({len(frames)} frames)", (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(self._window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording and not countdown:
                    # Start countdown
                    countdown = 3
                    countdown_start = time.time()
                    print("  >> 3... 2... 1...")
                elif recording:
                    # Stop recording
                    recording = False
                    print(f"  >> Stopped. Captured {len(frames)} frames.")
                    if len(frames) < 5:
                        print("  >> Too short (< 5 frames). Try again.")
                        frames = []

            elif key == ord("s") and frames and not recording:
                arr = np.stack(frames)
                self._db.save_sign(name, arr)
                saved = True
                print(f"  >> Saved '{name}' ({len(frames)} frames)")
                time.sleep(0.5)
                break

            elif key == ord("r"):
                frames = []
                recording = False
                saved = False
                print("  >> Reset. Press SPACE to re-record.")

            elif key == ord("q"):
                print(f"  >> Skipped '{name}'")
                cap.release()
                return None

        cap.release()
        return np.stack(frames) if frames else None

    def record_letter(self, letter: str) -> np.ndarray | None:
        """Record a static letter pose. Captures the best frame from a 1-second hold."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {self._camera_index}")
            return None

        letter = letter.upper()
        best_frame: np.ndarray | None = None
        capturing = False
        capture_start = 0.0
        capture_frames: list[np.ndarray] = []
        saved = False

        print(f"\n  Recording letter: '{letter}'")
        print("  SPACE = capture (hold for 1 sec)  |  s = save  |  r = redo  |  q = skip\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            annotated, lm543, lhand_px, rhand_px, *_ = self._tracker.process_frame(frame)
            display = annotated.copy()
            h, w = display.shape[:2]

            hands_ok = lhand_px is not None or rhand_px is not None

            if capturing:
                elapsed = time.time() - capture_start
                capture_frames.append(lm543.copy())
                bar_w = int(min(elapsed / 1.0, 1.0) * (w - 40))
                cv2.rectangle(display, (20, h - 60), (20 + bar_w, h - 50),
                              (0, 200, 255), -1)
                if elapsed >= 1.0:
                    # Pick the middle frame (most stable)
                    mid = len(capture_frames) // 2
                    best_frame = capture_frames[mid]
                    capturing = False
                    print(f"  >> Captured letter '{letter}' (from {len(capture_frames)} frames)")

            # Display
            color = (0, 255, 0) if hands_ok else (0, 0, 180)
            status = "Hands detected" if hands_ok else "Show your hand"
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(display, f"Letter: {letter}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

            if best_frame is not None and not capturing:
                cv2.putText(display, "Press 's' to save, 'r' to redo", (10, h - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            if saved:
                cv2.putText(display, "SAVED", (w - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(self._window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and not capturing and hands_ok:
                capturing = True
                capture_start = time.time()
                capture_frames = []
                print(f"  >> Hold steady for 1 second...")

            elif key == ord("s") and best_frame is not None:
                self._db.save_letter(letter, best_frame)
                saved = True
                print(f"  >> Saved letter '{letter}'")
                time.sleep(0.3)
                break

            elif key == ord("r"):
                best_frame = None
                capturing = False
                capture_frames = []
                saved = False
                print("  >> Reset. Press SPACE to re-capture.")

            elif key == ord("q"):
                print(f"  >> Skipped '{letter}'")
                cap.release()
                return None

        cap.release()
        return best_frame

    def record_batch(self, names: list[str]) -> None:
        """Record multiple signs in sequence."""
        print(f"\n  Batch recording: {len(names)} signs")
        print(f"  Signs: {', '.join(names)}\n")
        for i, name in enumerate(names):
            print(f"  [{i + 1}/{len(names)}]")
            self.record_sign(name)
        cv2.destroyAllWindows()
        print("\n  Batch complete!")
        print(f"\n  Database:\n  {self._db.summary()}")

    def record_all_letters(self) -> None:
        """Record A-Z in order."""
        print(f"\n  Recording all 26 letters")
        letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        for i, letter in enumerate(letters):
            print(f"  [{i + 1}/26]")
            self.record_letter(letter)
        cv2.destroyAllWindows()
        print("\n  All letters recorded!")
        print(f"\n  Database:\n  {self._db.summary()}")

    def close(self) -> None:
        self._tracker.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Bridge Sign Recorder")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--sign", type=str, help="Record a single sign")
    parser.add_argument("--batch", type=str, help="Comma-separated list of signs to record")
    parser.add_argument("--letters", action="store_true", help="Record all 26 letters A-Z")
    parser.add_argument("--list", action="store_true", help="List database contents")
    args = parser.parse_args()

    if args.list:
        db = SignDatabase()
        print(f"\n  {db.summary()}\n")
        return

    recorder = SignRecorder(camera_index=args.camera)

    try:
        if args.letters:
            recorder.record_all_letters()
        elif args.batch:
            names = [n.strip() for n in args.batch.split(",") if n.strip()]
            recorder.record_batch(names)
        elif args.sign:
            recorder.record_sign(args.sign)
        else:
            print("Usage:")
            print("  python -m src.avatar.recorder --letters")
            print("  python -m src.avatar.recorder --sign hello")
            print("  python -m src.avatar.recorder --batch hello,thankyou,yes,no")
            print("  python -m src.avatar.recorder --list")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
