"""
Quick end-to-end test: ASL tokens → fluid English → ElevenLabs voice.
Run from project root: python test_pipeline.py
"""
import os
import sys
import time

# Load .env without any extra dependencies
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from src.translation.text_smoother import TextSmoother
from src.speech.tts import TTSEngine
from src.speech.pipeline import SpeechPipeline

lava_token = os.environ.get("LAVA_TOKEN", "")
eleven_key  = os.environ.get("ELEVENLABS_API_KEY", "")

if not lava_token:
    print("ERROR: LAVA_TOKEN missing from .env")
    sys.exit(1)
if not eleven_key:
    print("WARNING: ELEVENLABS_API_KEY missing — will use pyttsx3")

smoother = TextSmoother(lava_token=lava_token)
tts      = TTSEngine(eleven_api_key=eleven_key)
pipeline = SpeechPipeline(smoother, tts, pause_s=2.0)
pipeline.start()

# Simulate two sentences with a pause between them
sentences = [
    ["HELLO", "MY", "NAME", "BRIDGE"],
    ["HELP", "PLEASE"],
]

for tokens in sentences:
    for sign in tokens:
        print(f"  sign: {sign!r}")
        pipeline.on_sign(sign, "word")
        time.sleep(0.4)
    print("  [pause — waiting for flush...]")
    time.sleep(3.0)   # trigger the 2s pause flush

pipeline.flush_now()
time.sleep(6)         # wait for async TTS playback
pipeline.stop()
print("Done.")
