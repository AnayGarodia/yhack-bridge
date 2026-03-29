"""
SpeechToText — Groq Whisper API for ultra-fast transcription.

Groq processes audio ~400x faster than real-time.
A 2-second chunk returns transcribed in ~100ms.

Falls back to local faster-whisper if GROQ_API_KEY is not set.

Usage:
    stt = SpeechToText(on_text=lambda t: print(f"> {t}"))
    stt.start()
    ...
    stt.stop()
"""

import collections
import io
import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHUNK_S = 2.0       # 2-second chunks to Groq
CHUNK_SAMP = int(SAMPLE_RATE * CHUNK_S)


class SpeechToText:
    """
    Continuous speech-to-text with word-level emission.

    Args:
        on_text:           Called with each transcribed word.
        model_name:        Ignored (kept for compatibility). Groq uses whisper-large-v3-turbo.
        energy_threshold:  RMS threshold for speech detection.
        silence_s:         Unused with Groq (kept for compat).
        min_phrase_s:      Unused with Groq (kept for compat).
    """

    def __init__(self, on_text, model_name: str = "tiny",
                 energy_threshold: float = 0.005, silence_s: float = 0.5,
                 min_phrase_s: float = 0.3):
        self._on_text = on_text
        self._energy_threshold = energy_threshold
        self._running = False
        self._capture_thread = None
        self._transcribe_thread = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=5)
        self._last_words = collections.deque(maxlen=5)
        self._last_transcript = ""

        self.audio_device = None

        # Try Groq first
        self._use_groq = False
        self._groq_client = None
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=groq_key)
                self._use_groq = True
                print(f"[stt] Groq whisper-large-v3-turbo ready")
            except Exception as e:
                print(f"[stt] Groq init failed ({e}), falling back to local whisper")

        # Fallback: local faster-whisper
        if not self._use_groq:
            print(f"[stt] loading faster-whisper '{model_name}'...")
            from faster_whisper import WhisperModel
            self._local_model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print(f"[stt] faster-whisper '{model_name}' ready")

    def start(self) -> None:
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="stt-capture")
        self._capture_thread.start()

        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop, daemon=True, name="stt-transcribe")
        self._transcribe_thread.start()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Audio capture — records 2-second chunks, skips silence
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            self.audio_device = f"{devices[default_in]['name']} (index {default_in})"
        else:
            self.audio_device = "system default"
        print(f"[stt] audio device: {self.audio_device}")
        print(f"[stt] energy threshold: {self._energy_threshold}")

        while self._running:
            try:
                audio = sd.rec(CHUNK_SAMP, samplerate=SAMPLE_RATE,
                               channels=1, dtype='float32')
                sd.wait()

                if not self._running:
                    break

                chunk = audio[:, 0]
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms < self._energy_threshold:
                    continue  # silence, skip

                # Drop oldest if queue full
                if self._audio_queue.full():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass

                self._audio_queue.put(chunk.copy())

            except Exception as e:
                if self._running:
                    print(f"[stt] capture error: {e}")
                    time.sleep(0.5)

    # ------------------------------------------------------------------
    # Transcription — Groq API or local whisper
    # ------------------------------------------------------------------

    def _transcribe_loop(self) -> None:
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if self._use_groq:
                    self._transcribe_groq(audio)
                else:
                    self._transcribe_local(audio)
            except Exception as e:
                if self._running:
                    print(f"[stt] transcription error: {e}")

    def _transcribe_groq(self, audio: np.ndarray) -> None:
        """Send audio to Groq Whisper API."""
        import soundfile as sf

        t0 = time.time()

        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buf.seek(0)
        buf.name = 'audio.wav'

        result = self._groq_client.audio.transcriptions.create(
            file=buf,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
            temperature=0.0,
        )

        elapsed_ms = (time.time() - t0) * 1000
        transcript = result.strip() if isinstance(result, str) else result.text.strip()

        if not transcript:
            return

        # Skip repeats
        if transcript.lower() == self._last_transcript.lower():
            return
        self._last_transcript = transcript

        print(f"[stt] Groq [{elapsed_ms:.0f}ms]: '{transcript}'")
        self._emit_words(transcript)

    def _transcribe_local(self, audio: np.ndarray) -> None:
        """Fallback: local faster-whisper."""
        segments, _ = self._local_model.transcribe(audio, language="en")
        text = " ".join(seg.text for seg in segments).strip()
        if text:
            self._emit_words(text)

    def _emit_words(self, text: str) -> None:
        """Emit individual words with dedup."""
        words = text.split()
        for word in words:
            clean = word.strip('.,!?;:"\'-()[]{}').lower()
            if not clean:
                continue
            # Skip if same as last emitted
            if self._last_words and clean == self._last_words[-1]:
                continue
            self._last_words.append(clean)
            if self._on_text:
                self._on_text(clean)


if __name__ == "__main__":
    import sys

    def on_text(text: str) -> None:
        print(f"> {text}")

    # Load .env
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    stt = SpeechToText(on_text=on_text)
    print("Listening... speak into your mic. Ctrl+C to stop.\n")
    stt.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stt.stop()
        print("\nStopped.")
        sys.exit(0)
