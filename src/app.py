"""
Bridge Web UI — Flask + SocketIO backend.

Wires together: SignRouter (ASL recognition), SpeechPipeline (ASL→English→TTS),
SpeechToText (mic→Whisper), EnglishToSigns (English→ASL glosses),
SignAnimator (animated SVG avatar).

Run:  python src/app.py
Open: http://localhost:5001
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
from src.translation.sign_decoder import SignDecoder
from src.output.bridge_camera import BridgeCamera
from src.avatar.sign_animator import SignAnimator
from src.avatar.sign_library import SignLibrary
from src.avatar.rpm_renderer import RPMRenderer
from src.avatar.animation_engine import AnimationEngine
from src.avatar.rpm_controller import RPMAvatarController


# ── WebSpeechPipeline — adds SocketIO emit on sentence completion ────────────
class WebSpeechPipeline(SpeechPipeline):
    def __init__(self, smoother, tts, sio, router, **kw):
        super().__init__(smoother, tts, **kw)
        self._sio = sio
        self._router = router

    def _process(self, tokens):
        try:
            raw = " ".join(tokens)
            print(f"[pipeline] smoothing: {raw!r}")
            ctx = self._history[-3:] if self._history else None
            text = self._smoother.smooth(tokens, context=ctx)
            if text:
                print(f"[pipeline] speaking:  {text!r}")
                self._tts.speak_async(text)
                self._sio.emit("sentence_complete", {"glosses": raw, "english": text})
                bridge_cam.set_translation(text)
                _stats["sentences_translated"] += 1
                self._history.append(text)
                if len(self._history) > 5:
                    self._history.pop(0)
                self._router.add_to_history(text)
        except Exception as e:
            print(f"[pipeline] error: {e}")


# ── Initialize modules ───────────────────────────────────────────────────────
lava_token = os.environ.get("LAVA_TOKEN", "")
eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")

if not lava_token:
    print("ERROR: LAVA_TOKEN missing from .env")
    sys.exit(1)

sign_decoder = SignDecoder(lava_token=lava_token)
sign_router = SignRouter(sign_decoder=sign_decoder, word_threshold=0.65)  # high threshold to avoid false positives
smoother = TextSmoother(lava_token=lava_token)
tts = TTSEngine(eleven_api_key=eleven_key)
pipeline = WebSpeechPipeline(smoother, tts, socketio, sign_router, pause_s=3.5)
e2s = EnglishToSigns(lava_token=lava_token)
sign_animator = SignAnimator(
    gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
    lava_token=lava_token,
)

# ── RPM Avatar System ───────────────────────────────────────────────────────
# NOTE: RPM renderer uses OpenGL which is thread-local. The renderer must be
# initialized AND called from the SAME thread. So we defer init to a dedicated
# render thread and share frames via _avatar_frame / _avatar_frame_lock.
sign_library = SignLibrary()
_rpm_avatar_ok = sign_library.load()
rpm_controller = None  # initialized in _avatar_render_loop thread

_avatar_frame: bytes | None = None
_avatar_frame_lock = threading.Lock()

# ── Google Meet Virtual Camera ───────────────────────────────────────────────
bridge_cam = BridgeCamera(width=1280, height=720, fps=30)

# ── STT with word-buffering callback ─────────────────────────────────────────
_stt_word_buffer: list[str] = []
_stt_last_word_time = 0.0
_stt_flush_interval = 1.5
_stt_lock = threading.Lock()


def _on_speech_word(word: str):
    with _stt_lock:
        _stt_word_buffer.append(word)
        global _stt_last_word_time
        _stt_last_word_time = time.time()
    print(f"[stt] word: {word!r}  (buffer: {len(_stt_word_buffer)} words)")


def _stt_flush_loop():
    global _stt_last_word_time
    while _running:
        time.sleep(0.2)
        with _stt_lock:
            if not _stt_word_buffer:
                continue
            elapsed = time.time() - _stt_last_word_time
            if elapsed < _stt_flush_interval:
                continue
            sentence = " ".join(_stt_word_buffer)
            _stt_word_buffer.clear()

        print(f"[stt] sentence: {sentence!r}")
        try:
            glosses = e2s.convert(sentence)
            print(f"[stt] glosses: {glosses}")
            socketio.emit("speech_transcribed", {"english": sentence, "asl_glosses": glosses})
            bridge_cam.set_speaker_text(sentence, " ".join(glosses))
            _stats["words_spoken"] += len(sentence.split())

            # Feed glosses to RPM avatar controller (3D) if available
            if rpm_controller is not None:
                for gloss in glosses:
                    rpm_controller.queue_word(gloss)

            # Also send animated SVG for each gloss (web fallback)
            for gloss in glosses:
                anim = sign_animator.get_animation(gloss)
                socketio.emit("avatar_sign", {
                    "sign": gloss,
                    "type": anim["type"],
                    "content": anim.get("content", ""),
                    "frames": anim.get("frames", []),
                })
                time.sleep(0.1)
        except Exception as e:
            print(f"[stt] error: {e}")


stt = SpeechToText(on_text=_on_speech_word, energy_threshold=0.03)

# ── Shared state ─────────────────────────────────────────────────────────────
_latest_frame: bytes | None = None
_frame_lock = threading.Lock()
_running = True
_mic_active = False

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

# ── Session stats ────────────────────────────────────────────────────────────
_stats = {
    "signs_recognized": 0,
    "words_spoken": 0,
    "sentences_translated": 0,
    "session_start": time.time(),
}


# ── Recognition thread (instant webcam start) ───────────────────────────────
def _recognition_loop():
    global _latest_frame, _running

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[cam] WARNING: cannot open camera {CAMERA_INDEX}, trying 0")
        cap = cv2.VideoCapture(0)

    # Stream raw webcam frames IMMEDIATELY while models load
    print("[cam] camera open — streaming raw frames while models load...")
    for _ in range(60):  # ~2 seconds of raw frames
        ret, frame = cap.read()
        if ret:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with _frame_lock:
                _latest_frame = jpeg.tobytes()
        if not _running:
            cap.release()
            return
        time.sleep(0.033)

    # NOW load models (webcam already visible to user)
    sign_router.open()
    print(f"[cam] models loaded — full recognition active (camera {CAMERA_INDEX})")

    frame_n = 0
    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        annotated, committed_sign, committed_conf, committed_mode = sign_router.process_frame(frame)

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

        # Note: webcam feed stays in the web UI only — the virtual camera
        # shows the bot avatar, not the webcam

        frame_n += 1
        if frame_n % 5 == 0:
            live_sign, live_conf, live_top5 = sign_router.get_live_display()
            if live_sign:
                socketio.emit("sign_update", {
                    "sign": live_sign,
                    "confidence": round(live_conf, 3),
                    "mode": "word",
                    "candidates": [
                        {"sign": name, "prob": round(prob, 3)}
                        for name, prob in (live_top5 or [])[:5]
                    ],
                })
                bridge_cam.set_sign(live_sign, live_conf)
            # Send continuously-decoded sentence preview
            live_decoded = sign_router.get_live_decoded()
            if live_decoded:
                socketio.emit("live_sentence", {
                    "signs": live_decoded,
                })

        if committed_sign:
            socketio.emit("sign_committed", {
                "sign": committed_sign,
                "confidence": round(committed_conf, 3),
            })
            pipeline.on_sign(committed_sign, committed_mode)
            bridge_cam.add_committed_sign(committed_sign)
            _stats["signs_recognized"] += 1

    cap.release()
    sign_router.close()
    print("[cam] recognition loop stopped")


# ── Avatar render loop (runs in its own thread for OpenGL context) ───────────
def _avatar_render_loop():
    """Initialize and run RPM renderer in a dedicated thread (OpenGL is thread-local)."""
    global rpm_controller, _avatar_frame

    import sys as _sys
    print("[rpm] Avatar render loop starting...", flush=True)

    if not _rpm_avatar_ok:
        print("[rpm] Sign library unavailable — avatar render loop not started", flush=True)
        return

    try:
        renderer = RPMRenderer("models/avatar.glb", width=1280, height=720)
        renderer._use_3d = False
        print("[rpm] Skeleton renderer created", flush=True)

        engine = AnimationEngine()
        ctrl = RPMAvatarController(renderer, sign_library, engine)

        rpm_controller = ctrl
        print("[rpm] Avatar render loop started", flush=True)

        while _running:
            try:
                frame = ctrl.get_frame()
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with _avatar_frame_lock:
                    _avatar_frame = jpeg.tobytes()

                # Send bot frame to virtual camera for Google Meet
                if bridge_cam.is_running:
                    bridge_cam.send_composed_frame()
            except Exception as e:
                print(f"[rpm] render error: {e}")
            time.sleep(0.033)

        print("[rpm] Avatar render loop stopped")
    except Exception as e:
        print(f"[rpm] CRASH: {e}")
        import traceback
        traceback.print_exc()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    _placeholder = np.full((480, 640, 3), 20, dtype=np.uint8)
    cv2.putText(_placeholder, "Starting camera...", (170, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
    _, _ph_jpeg = cv2.imencode(".jpg", _placeholder)
    _ph_bytes = _ph_jpeg.tobytes()

    def generate():
        while _running:
            with _frame_lock:
                frame = _latest_frame
            data = frame if frame else _ph_bytes
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/avatar_idle_svg")
def avatar_idle_svg():
    return Response(sign_animator.idle_svg, mimetype="image/svg+xml")


@app.route("/api/stats")
def api_stats():
    """Live session statistics."""
    elapsed = int(time.time() - _stats["session_start"])
    return {
        "signs_recognized": _stats["signs_recognized"],
        "words_spoken": _stats["words_spoken"],
        "sentences_translated": _stats["sentences_translated"],
        "uptime_seconds": elapsed,
        "bot_active": bridge_cam.is_running,
        "vocabulary_size": 250,
        "cached_animations": len(sign_animator.cached_signs),
    }


@app.route("/api/vocabulary")
def api_vocabulary():
    """Return the full 250-sign vocabulary."""
    import json as _json
    labels_path = os.path.join(os.path.dirname(__file__), "..", "models", "sign_to_prediction_index_map.json")
    with open(labels_path) as f:
        return _json.load(f)


@app.route("/avatar_feed")
def avatar_feed():
    """MJPEG stream of the RPM 3D avatar (served from render thread buffer)."""
    _ph = np.full((720, 1280, 3), 30, dtype=np.uint8)
    cv2.putText(_ph, "Avatar loading...", (480, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2, cv2.LINE_AA)
    _, _ph_jpeg = cv2.imencode(".jpg", _ph)
    _ph_bytes = _ph_jpeg.tobytes()

    def generate():
        while _running:
            with _avatar_frame_lock:
                frame = _avatar_frame
            data = frame if frame else _ph_bytes
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.033)
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
    bridge_cam.set_mic_active(_mic_active)


@socketio.on("test_avatar")
def handle_test_avatar(data):
    """Test input: type text, convert to glosses, send SVGs to avatar."""
    text = data.get("text", "").strip()
    if not text:
        return
    print(f"[test] input: {text!r}")
    glosses = e2s.convert(text)
    print(f"[test] glosses: {glosses}")
    socketio.emit("speech_transcribed", {"english": text, "asl_glosses": glosses})
    for gloss in glosses:
        anim = sign_animator.get_animation(gloss)
        socketio.emit("avatar_sign", {
            "sign": gloss,
            "type": anim["type"],
            "content": anim["content"],
        })
        time.sleep(0.1)


@socketio.on("clear_conversation")
def handle_clear():
    sign_router.reset_text()
    socketio.emit("conversation_cleared")
    print("[ui] conversation cleared")


@socketio.on("set_mode")
def handle_set_mode(data):
    mode = data.get("mode", "auto") if data else "auto"
    print(f"[ui] mode set to {mode}")


@socketio.on("get_stats")
def handle_get_stats():
    elapsed = int(time.time() - _stats["session_start"])
    mins, secs = divmod(elapsed, 60)
    socketio.emit("stats", {
        "signs": _stats["signs_recognized"],
        "words": _stats["words_spoken"],
        "sentences": _stats["sentences_translated"],
        "uptime": f"{mins}m {secs}s",
        "bot_active": bridge_cam.is_running,
    })


# ── Cleanup ───────────────────────────────────────────────────────────────────
def _cleanup():
    global _running, _mic_active
    _running = False
    bridge_cam.stop()
    pipeline.flush_now()
    pipeline.stop()
    if _mic_active:
        stt.stop()
        _mic_active = False

atexit.register(_cleanup)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    # Start virtual camera for Google Meet (non-fatal if OBS not installed)
    bridge_cam.start()

    threading.Thread(target=_recognition_loop, daemon=True).start()
    threading.Thread(target=_avatar_render_loop, daemon=True).start()
    threading.Thread(target=_stt_flush_loop, daemon=True).start()
    pipeline.start()

    # Auto-start mic listening (no need to click button in UI)
    _mic_active = True
    stt.start()
    bridge_cam.set_mic_active(True)
    print("[mic] auto-started listening")

    print(f"\n  Bridge is running at http://localhost:{port}")
    if bridge_cam.is_running:
        print(f"  Virtual camera active — select 'OBS Virtual Camera' in Google Meet")
    print(f"  Mic is listening — speak to trigger ASL avatar")
    print()
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
