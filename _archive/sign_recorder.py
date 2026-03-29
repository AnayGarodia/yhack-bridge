"""
Record training data for word-level ASL signs.

Usage:
    venv/bin/python src/recognition/sign_recorder.py [camera_index]

Controls:
    n         — type a new sign label
    SPACE     — start 2-second recording
    q         — quit
"""

import collections
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.recognition.holistic_tracker import HolisticTracker

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "recordings")

# Target signs list (for reference / quick-select)
TARGET_SIGNS = [
    "hello", "thankyou", "please", "help", "yes", "no", "sorry",
    "eat", "drink", "water", "mom", "dad", "happy", "sad",
    "good", "bad", "stop", "go", "like", "want", "more",
    "finish", "name", "what", "how",
]

RECORD_SECS = 2.0
FPS_EST     = 30       # approximate camera fps


def count_recordings():
    """Return dict: sign_name -> count of .npy files."""
    counts = {}
    if not os.path.isdir(DATA_DIR):
        return counts
    for d in sorted(os.listdir(DATA_DIR)):
        dp = os.path.join(DATA_DIR, d)
        if os.path.isdir(dp):
            n = len([f for f in os.listdir(dp) if f.endswith(".npy")])
            if n > 0:
                counts[d] = n
    return counts


def draw_status(frame, label, recording, countdown, counts, buf_len):
    h, w = frame.shape[:2]

    def put(text, y, scale=0.6, color=(200, 200, 200), thick=1):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    # Current label (large, top-left)
    put(f"Sign: {label or '(none)'}", 35, scale=1.0,
        color=(0, 255, 255), thick=2)

    # Recording indicator
    if recording:
        # Flashing red circle + countdown
        if int(time.time() * 4) % 2 == 0:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
        put(f"RECORDING  {countdown:.1f}s", 70, scale=0.8,
            color=(0, 0, 255), thick=2)
        # Progress bar
        frac = 1.0 - countdown / RECORD_SECS
        bar_w = int(frac * (w - 20))
        cv2.rectangle(frame, (10, 80), (10 + bar_w, 90), (0, 0, 255), -1)
        cv2.rectangle(frame, (10, 80), (w - 10, 90), (100, 100, 100), 1)
    else:
        put("SPACE=record  n=new label  q=quit", 70, scale=0.55)
        put(f"Frames buffered: {buf_len}", 90, scale=0.45, color=(150, 150, 150))

    # Counts sidebar (right side)
    x0 = w - 200
    put("Recordings:", 120, scale=0.5, color=(180, 180, 180))
    cv2.putText(frame, "Recordings:", (x0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1, cv2.LINE_AA)
    y = 50
    for sign_name in TARGET_SIGNS:
        n = counts.get(sign_name, 0)
        color = (0, 200, 0) if n >= 8 else (0, 200, 255) if n >= 4 else (120, 120, 120)
        marker = "*" if sign_name == label else " "
        cv2.putText(frame, f"{marker}{sign_name}: {n}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        y += 16
        if y > h - 10:
            break

    total = sum(counts.values())
    cv2.putText(frame, f"Total: {total}", (x0, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def prompt_label(window_name):
    """Show a text prompt overlay and read keystrokes until Enter."""
    chars = []
    while True:
        # Draw a simple prompt on a black frame
        prompt_img = np.zeros((100, 500, 3), dtype=np.uint8)
        text = "".join(chars)
        cv2.putText(prompt_img, f"Sign label: {text}_", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(prompt_img, "Enter=confirm  Esc=cancel", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.imshow(window_name, prompt_img)

        key = cv2.waitKey(0) & 0xFF
        if key == 13:   # Enter
            return "".join(chars).strip().lower() or None
        elif key == 27:  # Esc
            return None
        elif key == 8 or key == 127:   # Backspace
            if chars:
                chars.pop()
        elif 32 <= key < 127:
            chars.append(chr(key))


def main():
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    window  = "Sign Recorder"

    tracker = HolisticTracker()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open webcam {cam_idx}")
        return

    current_label = None
    recording     = False
    rec_start     = 0.0
    rec_frames    = []
    counts        = count_recordings()

    print("=== Sign Recorder ===")
    print(f"Camera {cam_idx}  |  Data dir: {DATA_DIR}")
    print(f"Target signs: {', '.join(TARGET_SIGNS)}")
    print("Press 'n' to set a label, SPACE to record 2s, 'q' to quit.\n")

    # Start by prompting for a label
    current_label = prompt_label(window)
    if current_label:
        print(f"  Label set: {current_label}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, lm543, *_ = tracker.process_frame(frame)

        # ---- Recording logic -----------------------------------------------
        if recording:
            # Collect frame
            lm_clean = lm543.copy()
            np.nan_to_num(lm_clean, copy=False, nan=0.0)
            rec_frames.append(lm_clean)

            elapsed   = time.time() - rec_start
            countdown = max(0.0, RECORD_SECS - elapsed)

            if elapsed >= RECORD_SECS:
                # Save
                sign_dir = os.path.join(DATA_DIR, current_label)
                os.makedirs(sign_dir, exist_ok=True)
                ts = int(time.time() * 1000)
                path = os.path.join(sign_dir, f"{ts}.npy")
                arr = np.stack(rec_frames, axis=0)   # (T, 543, 3)
                np.save(path, arr)

                n_frames = len(rec_frames)
                counts = count_recordings()
                n_clips = counts.get(current_label, 0)
                print(f"  Saved {current_label} clip #{n_clips}: "
                      f"{n_frames} frames → {path}")

                recording  = False
                rec_frames = []
        else:
            countdown = RECORD_SECS

        draw_status(annotated, current_label, recording, countdown,
                    counts, len(rec_frames))

        cv2.imshow(window, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("n"):
            new_label = prompt_label(window)
            if new_label:
                current_label = new_label
                print(f"  Label set: {current_label}")

        elif key == ord(" "):
            if current_label is None:
                print("  Set a label first! Press 'n'.")
            elif recording:
                pass   # already recording
            else:
                recording  = True
                rec_start  = time.time()
                rec_frames = []
                print(f"  Recording '{current_label}' ...")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    print("\n=== Final counts ===")
    for sign, n in sorted(count_recordings().items()):
        print(f"  {sign}: {n}")


if __name__ == "__main__":
    main()
