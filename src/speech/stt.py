"""
SpeechToText — continuous microphone transcription using faster-whisper.

Listens in a background thread, detects speech via energy level,
and calls on_text(str) whenever a phrase completes.

Usage:
    stt = SpeechToText(on_text=lambda t: print(f"> {t}"))
    stt.start()
    ...
    stt.stop()

Install: pip install faster-whisper
"""

import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000       # Whisper expects 16 kHz
CHUNK_S     = 0.1          # audio chunk size in seconds
CHUNK_SAMP  = int(SAMPLE_RATE * CHUNK_S)


class SpeechToText:
    """
    Args:
        on_text:           Callback called with transcribed string on phrase completion.
        model_name:        Whisper model size. "base" is fast enough for M1.
        energy_threshold:  RMS energy (0–1 scale) above which audio counts as speech.
                           Increase if background noise triggers false phrases.
        silence_s:         Seconds of silence needed to end a phrase (default 0.8s).
        min_phrase_s:      Minimum phrase length in seconds — shorter clips are ignored.
    """

    def __init__(
        self,
        on_text,
        model_name: str = "base",
        energy_threshold: float = 0.01,
        silence_s: float = 0.8,
        min_phrase_s: float = 0.5,
    ):
        self._on_text          = on_text
        self._energy_threshold = energy_threshold
        self._silence_chunks   = int(silence_s / CHUNK_S)
        self._min_phrase_samp  = int(min_phrase_s * SAMPLE_RATE)

        self._buffer: list[np.ndarray] = []
        self._silent_count = 0
        self._in_speech    = False
        self._running      = False
        self._thread: threading.Thread | None = None

        print("[stt] loading faster-whisper model...")
        # int8 quantization — fastest on CPU/M1, negligible accuracy loss
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"[stt] faster-whisper '{model_name}' ready")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening in a background thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="stt")
        self._thread.start()

    def stop(self) -> None:
        """Stop listening."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        def _audio_callback(indata, frames, time_info, status):
            if not self._running:
                return

            chunk  = indata[:, 0].copy()                   # mono float32
            energy = float(np.sqrt(np.mean(chunk ** 2)))

            if energy > self._energy_threshold:
                self._in_speech    = True
                self._silent_count = 0
                self._buffer.append(chunk)

            elif self._in_speech:
                self._buffer.append(chunk)
                self._silent_count += 1

                if self._silent_count >= self._silence_chunks:
                    audio              = np.concatenate(self._buffer)
                    self._buffer       = []
                    self._in_speech    = False
                    self._silent_count = 0

                    if len(audio) >= self._min_phrase_samp:
                        threading.Thread(
                            target=self._transcribe,
                            args=(audio,),
                            daemon=True,
                        ).start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SAMP,
            callback=_audio_callback,
        ):
            while self._running:
                sd.sleep(100)

    def _transcribe(self, audio: np.ndarray) -> None:
        try:
            segments, _ = self._model.transcribe(audio, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            if text:
                self._on_text(text)
        except Exception as e:
            print(f"[stt] transcription error: {e}")


if __name__ == "__main__":
    import sys

    def on_text(text: str) -> None:
        print(f"> {text}")

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
