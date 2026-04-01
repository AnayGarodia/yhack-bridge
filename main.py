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
# Avatar mode (speech → ASL signs → Google Meet via OBS Virtual Camera)
# ---------------------------------------------------------------------------

def run_avatar(args):
    from src.speech.stt import SpeechToText
    from src.avatar.avatar_pipeline import AvatarPipeline
    from src.avatar.avatar_renderer import AvatarRenderer
    from src.output.virtual_camera import VirtualCamera

    # Load .env for API keys (STT needs GROQ_API_KEY)
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    # HD resolution for Google Meet
    MEET_W, MEET_H = 1280, 720
    FRAMES_PER_SIGN = 18  # 0.6s at 30fps

    pipeline = AvatarPipeline()
    avatar = AvatarRenderer(width=MEET_W, height=MEET_H, bg_color=(25, 25, 30))

    # Sign animation state
    current_sign = None
    sign_frame_idx = 0

    # -- Virtual camera for Google Meet --
    vcam = VirtualCamera(width=MEET_W, height=MEET_H, fps=30)
    vcam_ok = vcam.start()
    if not vcam_ok:
        print("\nWARNING: Virtual camera not available -- local display only")
        print("To enable Google Meet: open OBS -> Start Virtual Camera -> rerun\n")

    # -- Phrase buffer for demo script matching --
    phrase_buffer = []
    last_word_time = [0.0]
    display_text = [""]
    PHRASE_FLUSH_S = 1.5

    def on_word(word):
        pipeline.on_word(word)
        phrase_buffer.append(word)
        last_word_time[0] = time.monotonic()

    stt = SpeechToText(on_text=on_word, model_name="tiny")

    print(f"\n{'='*50}")
    print(f"  Bridge Avatar Mode (Cartoon Character)")
    if vcam_ok:
        print(f"  Virtual camera: {vcam.device}")
        print(f"  In Google Meet: Settings -> Video -> '{vcam.device}'")
    print(f"  q=quit  r=reset  |  Speak into mic")
    print(f"{'='*50}\n")

    stt.start()
    time.sleep(0.5)
    if stt.audio_device:
        print(f"  Audio device: {stt.audio_device}")

    WINDOW = "Bridge — Avatar (Speech → ASL)"
    fps_times = collections.deque(maxlen=60)
    signs_shown = []

    while True:
        t0 = time.perf_counter()

        # -- Flush phrase buffer for demo matching --
        now_mono = time.monotonic()
        if phrase_buffer and (now_mono - last_word_time[0]) > PHRASE_FLUSH_S:
            phrase = " ".join(phrase_buffer)
            phrase_buffer.clear()
            result = pipeline.push_text(phrase)
            if result:
                display_text[0] = result

        # -- Check if current sign animation is done --
        sign_done = current_sign is None or sign_frame_idx >= FRAMES_PER_SIGN
        if sign_done:
            next_sign = pipeline.next_sign()
            if next_sign is not None:
                current_sign = next_sign
                sign_frame_idx = 0
                signs_shown.append(next_sign)
            else:
                current_sign = None

        # -- Render avatar frame --
        if current_sign:
            progress = (sign_frame_idx + 1) / FRAMES_PER_SIGN
            frame = avatar.render_text_card(current_sign)
            # Draw progress bar at very bottom
            bar_w = int(MEET_W * progress)
            cv2.rectangle(frame, (0, MEET_H - 6), (bar_w, MEET_H), (80, 180, 255), -1)
            sign_frame_idx += 1
        else:
            frame = avatar.render_idle("Listening...")

        h, w = frame.shape[:2]

        # -- Overlays --
        # Header bar
        cv2.rectangle(frame, (0, 0), (w, 36), (16, 16, 16), -1)
        cv2.putText(frame, "Bridge", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Interpreter", (110, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # Status pill
        if current_sign:
            pill = f"SIGNING: {current_sign}"
            pill_col = (60, 200, 80)
        else:
            pill = "LISTENING"
            pill_col = (100, 100, 100)
        (pw, _), _ = cv2.getTextSize(pill, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        px = w - pw - 20
        cv2.rectangle(frame, (px - 6, 8), (px + pw + 6, 30), pill_col, -1, cv2.LINE_AA)
        cv2.putText(frame, pill, (px, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS (small, corner)
        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

        # Virtual camera indicator
        vcam_label = "LIVE" if vcam_ok else "LOCAL"
        cv2.circle(frame, (w - 14, 24), 5, (0, 0, 255) if vcam_ok else (80, 80, 80), -1)

        # Demo display text (subtitle)
        dt = display_text[0] or pipeline.current_display_text
        if dt:
            cv2.rectangle(frame, (0, h - 90), (w, h - 50), (12, 12, 12), -1)
            cv2.putText(frame, dt, (20, h - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Signs history bar
        if signs_shown:
            history = " ".join(signs_shown[-12:])
            cv2.rectangle(frame, (0, h - 50), (w, h), (12, 12, 12), -1)
            cv2.putText(frame, history, (12, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1, cv2.LINE_AA)
            qdepth = pipeline.queue_depth
            if qdepth > 0:
                cv2.putText(frame, f"+{qdepth}", (w - 50, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        # -- Push frame to virtual camera (Google Meet) --
        if vcam_ok:
            vcam.send_frame(frame)

        # -- Local preview (smaller window) --
        preview = cv2.resize(frame, (640, 360))
        cv2.imshow(WINDOW, preview)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            pipeline.clear()
            signs_shown.clear()
            display_text[0] = ""
            current_sign = None

    stt.stop()
    if vcam_ok:
        vcam.stop()
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
