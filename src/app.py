"""
Bridge Web UI — Flask + SocketIO backend.

Wires together: SignRouter (ASL recognition), SpeechPipeline (ASL→English→TTS),
SpeechToText (mic→Whisper), EnglishToSigns (English→ASL glosses).

Run:  python src/app.py
Open: http://localhost:5000
"""

import os
import sys
import threading
import time
import atexit

# Ensure project root is on sys.path so `from src.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

# ── Load .env ────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Flask + SocketIO ─────────────────────────────────────────────────────────
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

_template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
app = Flask(__name__, template_folder=_template_dir)
app.config["SECRET_KEY"] = "bridge-yhack-2026"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ── Project imports ──────────────────────────────────────────────────────────
from src.recognition.sign_router import SignRouter
from src.speech.stt import SpeechToText
from src.speech.tts import TTSEngine
from src.speech.pipeline import SpeechPipeline
from src.translation.text_smoother import TextSmoother
from src.translation.english_to_signs import EnglishToSigns


# ── WebSpeechPipeline — adds SocketIO emit on sentence completion ────────────
class WebSpeechPipeline(SpeechPipeline):
    def __init__(self, smoother, tts, sio, **kw):
        super().__init__(smoother, tts, **kw)
        self._sio = sio

    def _process(self, tokens):
        try:
            raw = " ".join(tokens)
            print(f"[pipeline] smoothing: {raw!r}")
            text = self._smoother.smooth(tokens)
            if text:
                print(f"[pipeline] speaking:  {text!r}")
                self._tts.speak_async(text)
                self._sio.emit("sentence_complete", {"glosses": raw, "english": text})
        except Exception as e:
            print(f"[pipeline] error: {e}")


# ── Initialize modules ───────────────────────────────────────────────────────
lava_token = os.environ.get("LAVA_TOKEN", "")
eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")

if not lava_token:
    print("ERROR: LAVA_TOKEN missing from .env")
    sys.exit(1)

sign_router = SignRouter()
smoother = TextSmoother(lava_token=lava_token)
tts = TTSEngine(eleven_api_key=eleven_key)
pipeline = WebSpeechPipeline(smoother, tts, socketio, pause_s=2.0)
e2s = EnglishToSigns(lava_token=lava_token)

# ── STT with SocketIO callback ───────────────────────────────────────────────
def _on_speech(text: str):
    print(f"[stt] heard: {text!r}")
    glosses = e2s.convert(text)
    print(f"[stt] glosses: {glosses}")
    socketio.emit("speech_transcribed", {"english": text, "asl_glosses": glosses})

stt = SpeechToText(on_text=_on_speech, energy_threshold=0.03)

# ── Shared state ─────────────────────────────────────────────────────────────
_latest_frame: bytes | None = None
_frame_lock = threading.Lock()
_running = True
_mic_active = False

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))


# ── Recognition thread ───────────────────────────────────────────────────────
def _recognition_loop():
    global _latest_frame, _running

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[cam] WARNING: cannot open camera {CAMERA_INDEX}, trying 0")
        cap = cv2.VideoCapture(0)

    sign_router.open()
    print(f"[cam] recognition loop started (camera {CAMERA_INDEX})")

    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        annotated, sign, conf, mode = sign_router.process_frame(frame)

        # Encode for MJPEG stream
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

        # Emit sign detection + feed into pipeline
        if sign:
            socketio.emit("sign_update", {
                "sign": sign,
                "confidence": round(conf, 3),
                "mode": mode,
            })
            pipeline.on_sign(sign, mode)

    cap.release()
    sign_router.close()
    print("[cam] recognition loop stopped")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while _running:
            with _frame_lock:
                frame = _latest_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)  # ~30 fps cap
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── SocketIO event handlers ──────────────────────────────────────────────────
@socketio.on("toggle_mic")
def handle_toggle_mic():
    global _mic_active
    _mic_active = not _mic_active
    if _mic_active:
        stt.start()
        print("[mic] started listening")
    else:
        stt.stop()
        print("[mic] stopped listening")
    socketio.emit("mic_status", {"active": _mic_active})


@socketio.on("clear_conversation")
def handle_clear():
    sign_router.reset_text()
    socketio.emit("conversation_cleared")
    print("[ui] conversation cleared")


@socketio.on("set_mode")
def handle_set_mode(data):
    mode = data.get("mode", "auto") if data else "auto"
    print(f"[ui] mode set to {mode}")


# ── Cleanup ───────────────────────────────────────────────────────────────────
def _cleanup():
    global _running, _mic_active
    _running = False
    pipeline.flush_now()
    pipeline.stop()
    if _mic_active:
        stt.stop()
        _mic_active = False

atexit.register(_cleanup)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    threading.Thread(target=_recognition_loop, daemon=True).start()
    pipeline.start()
    print(f"\n  Bridge is running at http://localhost:{port}\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
