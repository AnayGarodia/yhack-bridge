#!/usr/bin/env python3
"""
Bridge — Real-time ASL Sign Language Recognition

250-sign vocabulary using hoyso48's 1st-place Kaggle TFLite model.
MediaPipe Holistic extracts landmarks, TFLite model classifies sequences.

Usage:
    python main.py                  # default camera 0
    python main.py --camera 1       # specific camera index
    python main.py --threshold 0.40 # confidence threshold

Controls:
    q     — quit
    r     — reset sentence buffer
    SPACE — insert space in sentence
    f     — toggle fingerspelling mode
"""

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.recognition.tflite_classifier import TFLiteClassifier
from src.recognition.asl_classifier import ASLClassifier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COOLDOWN_S         = 2.0
LETTER_THRESHOLD   = 0.55
LETTER_COOLDOWN_S  = 0.8
WINDOW_NAME        = "Bridge — ASL Recognition"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame, sign, conf, mode, top5, sentence, fps,
                 buf_fill, hands_visible, fingerspell_only):
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    # Mode indicator
    mode_text = "[FINGERSPELL]" if fingerspell_only else "[WORD]"
    mode_color = (0, 200, 255) if fingerspell_only else (0, 220, 0)
    cv2.putText(frame, mode_text, (w - 180, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)

    # Hand status + buffer
    hand_text = "Hands detected" if hands_visible else "No hands"
    hand_color = (0, 200, 0) if hands_visible else (0, 0, 180)
    cv2.putText(frame, hand_text, (w - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, hand_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"buf: {buf_fill}", (w - 180, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # Primary prediction
    if sign is not None:
        color = (0, 255, 80) if mode == "word" else (0, 200, 255)
        display = f"{sign}  ({conf:.0%})"
        (tw, _), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        x = (w - tw) // 2
        cv2.putText(frame, display, (x, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Top-5 bar
    if top5 and not fingerspell_only:
        parts = [f"{g} {c:.0%}" for g, c in top5]
        cv2.putText(frame, "  |  ".join(parts), (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 100), 1, cv2.LINE_AA)

    # Sentence
    if sentence:
        cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, " ".join(sentence[-12:]), (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 1, cv2.LINE_AA)

    # Controls
    cv2.putText(frame, "q=quit  r=reset  SPACE=space  f=fingerspell",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (80, 80, 80), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bridge — ASL Recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    print("Initializing Bridge ASL Recognition...")
    print(f"  Camera: {args.camera}  Threshold: {args.threshold:.0%}")

    word_clf = TFLiteClassifier(confidence_threshold=args.threshold)
    letter_clf = ASLClassifier()

    if not word_clf.ready:
        print("ERROR: TFLite model not found at models/model.tflite")
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam {args.camera}")
        sys.exit(1)

    # State
    sentence          = []
    fingerspell_only  = False
    last_emit_word    = None
    last_emit_time    = 0.0
    last_emit_letter  = None
    last_letter_time  = 0.0
    fps_times         = collections.deque(maxlen=60)

    print(f"\n{WINDOW_NAME}")
    print("  q=quit  r=reset  SPACE=space  f=fingerspell\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # Downscale if needed
        if frame.shape[1] > args.width:
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        # ---- WORD MODE: Holistic tracking + LSTM ----
        sign, conf, mode = None, 0.0, "idle"
        top5 = []
        hands_visible = False

        if not fingerspell_only:
            # process_frame runs holistic tracker + buffers landmarks
            display_frame, hands_visible = word_clf.process_frame(frame)

            # Trigger async inference
            word_clf.maybe_run_async()

            # Get latest result
            (word_sign, word_conf), top5 = word_clf.get_async_result()

            now = time.monotonic()
            if word_sign is not None and word_conf >= args.threshold:
                if word_sign != last_emit_word or (now - last_emit_time) > COOLDOWN_S:
                    sign, conf, mode = word_sign, word_conf, "word"
        else:
            display_frame = frame.copy()

        # ---- FINGERSPELL MODE ----
        if fingerspell_only:
            # Reuse holistic tracker for hand landmarks
            annotated, hv = word_clf.process_frame(frame)
            display_frame = annotated
            hands_visible = hv

            # Get hand landmarks for fingerspelling from the tracker
            if word_clf._tracker is not None:
                # Use the last holistic result for hand landmarks
                norm_right = None
                norm_left = None
                # Re-run to get the normalized landmarks
                _, lm543, _, _, nl, nr = word_clf._tracker.process_frame(frame)
                active = nr or nl
                if active:
                    letter_sign, letter_conf = letter_clf.classify(active)
                    now = time.monotonic()
                    if letter_sign and letter_conf >= LETTER_THRESHOLD:
                        if letter_sign != last_emit_letter or (now - last_letter_time) > LETTER_COOLDOWN_S:
                            sign, conf, mode = letter_sign, letter_conf, "letter"

        # ---- Emit to sentence ----
        if sign is not None:
            sentence.append(sign)
            if mode == "word":
                last_emit_word = sign
                last_emit_time = time.monotonic()
                word_clf.on_sign_boundary()
                print(f"  >> WORD: {sign} ({conf:.0%})")
            elif mode == "letter":
                last_emit_letter = sign
                last_letter_time = time.monotonic()
                print(f"  >> LETTER: {sign} ({conf:.0%})")

        # ---- FPS ----
        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

        # ---- Draw + display ----
        draw_overlay(display_frame, sign, conf, mode, top5, sentence, fps,
                     word_clf.buf_fill, hands_visible, fingerspell_only)
        cv2.imshow(WINDOW_NAME, display_frame)

        # ---- Keys ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            sentence.clear()
            word_clf.reset()
            last_emit_word = None
            last_emit_letter = None
            print("  [reset]")
        elif key == ord(" "):
            sentence.append(" ")
        elif key == ord("f"):
            fingerspell_only = not fingerspell_only
            print(f"  [mode: {'fingerspell' if fingerspell_only else 'word'}]")

    cap.release()
    cv2.destroyAllWindows()
    word_clf.close()

    print(f"\nSentence: {' '.join(sentence)}")


if __name__ == "__main__":
    main()
