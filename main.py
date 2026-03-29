#!/usr/bin/env python3
"""
Bridge — Real-time ASL Sign Language Recognition & Avatar

Modes:
    python main.py                       # ASL recognition (webcam → signs)
    python main.py --mode avatar         # Avatar mode (speech → ASL signs display)
    python main.py --camera 1            # specific camera
    python main.py --threshold 0.40      # confidence threshold

Controls:
    q     — quit
    r     — reset
    SPACE — insert space (recognition mode)
    f     — toggle fingerspelling (recognition mode)
"""

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Drawing (recognition mode)
# ---------------------------------------------------------------------------

def draw_overlay(frame, sign, conf, mode, top5, sentence, fps,
                 buf_fill, hands_visible, fingerspell_only):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{fps:.0f} fps", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    mode_text = "[FINGERSPELL]" if fingerspell_only else "[WORD]"
    mode_color = (0, 200, 255) if fingerspell_only else (0, 220, 0)
    cv2.putText(frame, mode_text, (w - 180, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)
    hand_text = "Hands detected" if hands_visible else "No hands"
    hand_color = (0, 200, 0) if hands_visible else (0, 0, 180)
    cv2.putText(frame, hand_text, (w - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, hand_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"buf: {buf_fill}", (w - 180, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    if sign is not None:
        color = (0, 255, 80) if mode == "word" else (0, 200, 255)
        display = f"{sign}  ({conf:.0%})"
        (tw, _), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        x = (w - tw) // 2
        cv2.putText(frame, display, (x, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    if top5 and not fingerspell_only:
        parts = [f"{g} {c:.0%}" for g, c in top5]
        cv2.putText(frame, "  |  ".join(parts), (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 100), 1, cv2.LINE_AA)
    if sentence:
        cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, " ".join(sentence[-12:]), (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "q=quit  r=reset  SPACE=space  f=fingerspell",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (80, 80, 80), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Recognition mode (ASL → text)
# ---------------------------------------------------------------------------

def run_recognition(args):
    from src.recognition.tflite_classifier import TFLiteClassifier
    from src.recognition.asl_classifier import ASLClassifier

    word_clf = TFLiteClassifier(confidence_threshold=args.threshold)
    letter_clf = ASLClassifier()
    if not word_clf.ready:
        print("ERROR: TFLite model not found at models/model.tflite")
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam {args.camera}")
        sys.exit(1)

    sentence = []
    fingerspell_only = False
    last_emit_word = None
    last_emit_time = 0.0
    last_emit_letter = None
    last_letter_time = 0.0
    fps_times = collections.deque(maxlen=60)
    COOLDOWN_S = 2.0
    LETTER_THRESHOLD = 0.55
    LETTER_COOLDOWN_S = 0.8
    WINDOW = "Bridge — ASL Recognition"

    print(f"\n{WINDOW}\n  q=quit  r=reset  SPACE=space  f=fingerspell\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        if frame.shape[1] > args.width:
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        sign, conf, mode = None, 0.0, "idle"
        top5 = []
        hands_visible = False

        if not fingerspell_only:
            display_frame, hands_visible = word_clf.process_frame(frame)
            word_clf.maybe_run_async()
            (word_sign, word_conf), top5 = word_clf.get_async_result()
            now = time.monotonic()
            if word_sign is not None and word_conf >= args.threshold:
                if word_sign != last_emit_word or (now - last_emit_time) > COOLDOWN_S:
                    sign, conf, mode = word_sign, word_conf, "word"
        else:
            display_frame = frame.copy()

        if fingerspell_only:
            annotated, hv = word_clf.process_frame(frame)
            display_frame = annotated
            hands_visible = hv
            if word_clf._tracker is not None:
                _, lm543, _, _, nl, nr = word_clf._tracker.process_frame(frame)
                active = nr or nl
                if active:
                    letter_sign, letter_conf = letter_clf.classify(active)
                    now = time.monotonic()
                    if letter_sign and letter_conf >= LETTER_THRESHOLD:
                        if letter_sign != last_emit_letter or (now - last_letter_time) > LETTER_COOLDOWN_S:
                            sign, conf, mode = letter_sign, letter_conf, "letter"

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

        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        draw_overlay(display_frame, sign, conf, mode, top5, sentence, fps,
                     word_clf.buf_fill, hands_visible, fingerspell_only)
        cv2.imshow(WINDOW, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            sentence.clear()
            word_clf.reset()
            last_emit_word = None
            last_emit_letter = None
        elif key == ord(" "):
            sentence.append(" ")
        elif key == ord("f"):
            fingerspell_only = not fingerspell_only

    cap.release()
    cv2.destroyAllWindows()
    word_clf.close()
    print(f"\nSentence: {' '.join(sentence)}")


# ---------------------------------------------------------------------------
# Avatar mode (speech → ASL signs display)
# ---------------------------------------------------------------------------

def run_avatar(args):
    from src.speech.stt import SpeechToText
    from src.avatar.avatar_pipeline import AvatarPipeline
    from src.avatar.hand_renderer import HandRenderer

    pipeline = AvatarPipeline()
    renderer = HandRenderer(width=args.width, height=int(args.width * 3 / 4))

    # STT callback: each word goes to the pipeline
    def on_word(word):
        pipeline.on_word(word)

    stt = SpeechToText(on_text=on_word, model_name="tiny")

    print(f"\n  Bridge Avatar Mode")
    print(f"  Whisper model: {stt.model_name}")
    print(f"  Audio device: (starting...)")
    print(f"  q=quit  r=reset\n")

    stt.start()
    # Print audio device after STT starts (it detects in its thread)
    time.sleep(0.5)
    if stt.audio_device:
        print(f"  Audio device: {stt.audio_device}")

    WINDOW = "Bridge — Avatar (Speech → ASL)"
    fps_times = collections.deque(maxlen=60)
    current_sign = None
    signs_shown = []

    while True:
        t0 = time.perf_counter()

        # If renderer is idle, get next sign from pipeline
        if renderer.is_idle:
            next_sign = pipeline.next_sign()
            if next_sign is not None:
                renderer.set_word(next_sign)
                current_sign = next_sign
                signs_shown.append(next_sign)

        # Get next animation frame (always returns a frame, even blank)
        frame, done = renderer.next_frame()

        # Draw status overlays
        h, w = frame.shape[:2]

        # FPS
        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(frame, f"{fps:.0f} fps", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # WPM indicator
        wpm = pipeline.words_per_minute
        wpm_color = (0, 200, 0) if wpm < 120 else (0, 200, 255) if wpm < 180 else (0, 0, 255)
        cv2.putText(frame, f"{wpm:.0f} WPM", (w - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, wpm_color, 1, cv2.LINE_AA)

        # Queue depth
        qdepth = pipeline.queue_depth
        q_color = (0, 200, 0) if qdepth <= 1 else (0, 200, 255) if qdepth <= 2 else (0, 0, 255)
        cv2.putText(frame, f"Queue: {qdepth}", (w - 120, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, q_color, 1, cv2.LINE_AA)

        # "Speaking too fast" warning
        if qdepth > 3:
            cv2.putText(frame, "SPEAKING TOO FAST", (w // 2 - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Recent signs history at bottom
        if signs_shown:
            history = " ".join(signs_shown[-8:])
            cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, history, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1, cv2.LINE_AA)

        # Controls
        cv2.putText(frame, "q=quit  r=reset  |  Speak into mic",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (80, 80, 80), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, frame)

        # Key handling
        key = cv2.waitKey(33) & 0xFF  # 33ms = ~30fps
        if key == ord("q"):
            break
        elif key == ord("r"):
            pipeline.clear()
            signs_shown.clear()
            renderer.set_word("")

    stt.stop()
    cv2.destroyAllWindows()
    print(f"\nSigns shown: {' '.join(signs_shown)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bridge — ASL Recognition & Avatar")
    parser.add_argument("--mode", choices=["recognition", "avatar"], default="recognition",
                        help="recognition = webcam ASL→text, avatar = speech→ASL signs")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    print("Bridge — Real-time ASL Translator")
    print(f"  Mode: {args.mode}")

    if args.mode == "avatar":
        run_avatar(args)
    else:
        run_recognition(args)


if __name__ == "__main__":
    main()
