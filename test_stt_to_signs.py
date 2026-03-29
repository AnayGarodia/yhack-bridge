"""
Test: microphone → Whisper STT → EnglishToSigns → ASL gloss tokens
Run: python test_stt_to_signs.py
"""
import os
import sys
import time

# Load .env
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.speech.stt import SpeechToText
from src.translation.english_to_signs import EnglishToSigns

lava_token = os.environ.get("LAVA_TOKEN", "")
if not lava_token:
    print("ERROR: LAVA_TOKEN missing from .env")
    sys.exit(1)

converter = EnglishToSigns(lava_token=lava_token)

def on_text(text: str) -> None:
    print(f"\n  heard:  {text!r}")
    glosses = converter.convert(text)
    print(f"  signs:  {glosses}")

stt = SpeechToText(on_text=on_text, energy_threshold=0.03)
print("Listening... speak a sentence, pause, see ASL tokens. Ctrl+C to stop.\n")
stt.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    stt.stop()
    print("\nStopped.")
