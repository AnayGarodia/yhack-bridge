"""
TTS — text-to-speech with ElevenLabs primary and pyttsx3 fallback.

ElevenLabs: uses mp3_44100_128 (free-tier compatible), plays via system player.
pyttsx3:    fully offline fallback, no API key needed.

Use TTSEngine for automatic fallback behaviour:
    engine = TTSEngine(eleven_api_key="xi-...")
    engine.speak("Hello")           # tries ElevenLabs, falls back if it fails
    engine.speak_async("Hi there")  # same, non-blocking

Or use the individual classes directly if you know what you want:
    tts = ElevenLabsTTS(api_key="xi-...")
    tts.speak("Hello")
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading

import requests

ELEVEN_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah — clear, natural English


def _find_mp3_player() -> list[str] | None:
    """Return a subprocess command prefix that can play an MP3 file path."""
    candidates = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],  # ffmpeg suite
        ["mpg123", "-q"],    # lightweight Linux
        ["mpg321", "-q"],    # alternative
        ["afplay"],          # macOS
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd
    return None


# Detect player once at import time
_MP3_PLAYER = _find_mp3_player()


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------

class ElevenLabsTTS:
    """
    Args:
        api_key:   ElevenLabs API key (xi-api-key header).
        voice_id:  ElevenLabs voice ID. Defaults to Rachel.
        model_id:  ElevenLabs model. eleven_turbo_v2 is fastest.
        timeout:   HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = "eleven_flash_v2_5",
        timeout: int = 15,
    ):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._timeout = timeout
        self._play_lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Synthesize and play. Blocks until playback finishes."""
        if not text.strip():
            return

        print(f"[tts] key={self._api_key[:8]}... voice={self._voice_id} model={self._model_id}")
        r = requests.post(
            f"{ELEVEN_BASE}/text-to-speech/{self._voice_id}",
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": self._model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=self._timeout,
        )
        r.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(r.content)
            tmp = f.name

        try:
            if _MP3_PLAYER:
                with self._play_lock:
                    subprocess.run(
                        _MP3_PLAYER + [tmp],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
            else:
                print(f"[tts] no mp3 player found (install ffmpeg or mpg123). text: {text!r}")
        finally:
            os.unlink(tmp)

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# pyttsx3 fallback (fully offline)
# ---------------------------------------------------------------------------

class Pyttsx3TTS:
    """Offline TTS using pyttsx3. No API key required."""

    def __init__(self):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# TTSEngine — tries ElevenLabs, falls back to pyttsx3
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Auto-fallback TTS engine.

    Args:
        eleven_api_key: If provided, tries ElevenLabs first.
        voice_id:       ElevenLabs voice ID.
    """

    def __init__(self, eleven_api_key: str = "", voice_id: str = DEFAULT_VOICE_ID):
        self._primary = ElevenLabsTTS(api_key=eleven_api_key, voice_id=voice_id) if eleven_api_key else None
        self._fallback: Pyttsx3TTS | None = None
        try:
            self._fallback = Pyttsx3TTS()
        except Exception as e:
            print(f"[tts] pyttsx3 unavailable: {e}")

    def speak(self, text: str) -> None:
        if self._primary:
            try:
                self._primary.speak(text)
                return
            except Exception as e:
                print(f"[tts] ElevenLabs failed ({e}), falling back to pyttsx3")
        if self._fallback:
            self._fallback.speak(text)
        else:
            print(f"[tts] no TTS available. text: {text!r}")

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


if __name__ == "__main__":
    api_key = os.environ.get("ELEVENLABS_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else "")

    if api_key:
        print("Testing ElevenLabs...")
        tts = TTSEngine(eleven_api_key=api_key)
    else:
        print("No ELEVENLABS_API_KEY set — testing pyttsx3 fallback only")
        tts = TTSEngine()

    tts.speak("Hello. My name is Bridge. I translate American Sign Language.")
    print("Done.")
