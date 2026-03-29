"""
SpeechToText — low-latency continuous transcription using faster-whisper tiny.

Emits individual words as they come, not full sentences.
Deduplicates overlap between consecutive Whisper chunks.

Usage:
    stt = SpeechToText(on_text=lambda t: print(f"> {t}"))
    stt.start()
    ...
    stt.stop()
"""

import collections
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
CHUNK_S     = 0.1
CHUNK_SAMP  = int(SAMPLE_RATE * CHUNK_S)
MAX_AUDIO_S = 1.5   # max audio buffer before forced flush (keeps latency low)
MAX_AUDIO_SAMP = int(SAMPLE_RATE * MAX_AUDIO_S)


class SpeechToText:
    """
    Low-latency speech-to-text with word-level emission and dedup.

    Args:
        on_text:           Called with each transcribed word (one at a time).
        model_name:        Whisper model — "tiny" for lowest latency.
        energy_threshold:  RMS threshold for speech detection.
        silence_s:         Silence duration to end a phrase.
        min_phrase_s:      Minimum phrase length.
    """

    def __init__(
        self,
        on_text,
        model_name: str = "tiny",
        energy_threshold: float = 0.01,
        silence_s: float = 0.5,
        min_phrase_s: float = 0.3,
    ):
        self._on_text          = on_text
        self._energy_threshold = energy_threshold
        self._silence_chunks   = int(silence_s / CHUNK_S)
        self._min_phrase_samp  = int(min_phrase_s * SAMPLE_RATE)

        self._buffer: list[np.ndarray] = []
        self._buffer_samples   = 0
        self._silent_count     = 0
        self._in_speech        = False
        self._running          = False
        self._thread: threading.Thread | None = None

        # Dedup: last 3 emitted words
        self._last_emitted = collections.deque(maxlen=3)

        # Transcription queue: audio goes in, never blocks capture
        self._audio_queue: queue.Queue = queue.Queue(maxsize=2)
        self._transcribe_thread: threading.Thread | None = None

        self.model_name = model_name
        self.audio_device = None

        print(f"[stt] loading faster-whisper '{model_name}'...")
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"[stt] faster-whisper '{model_name}' ready")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        # Transcription worker thread
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop, daemon=True, name="stt-transcribe"
        )
        self._transcribe_thread.start()
        # Audio capture thread
        self._thread = threading.Thread(target=self._run, daemon=True, name="stt-capture")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _run(self) -> None:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            dev_name = devices[default_in]['name']
            self.audio_device = f"{dev_name} (index {default_in})"
        else:
            self.audio_device = "system default"
        print(f"[stt] audio device: {self.audio_device}")

        def _audio_callback(indata, frames, time_info, status):
            if not self._running:
                return

            chunk  = indata[:, 0].copy()
            energy = float(np.sqrt(np.mean(chunk ** 2)))

            if energy > self._energy_threshold:
                self._in_speech    = True
                self._silent_count = 0
                self._buffer.append(chunk)
                self._buffer_samples += len(chunk)

                # Force flush if buffer is too long (keeps latency under MAX_AUDIO_S)
                if self._buffer_samples >= MAX_AUDIO_SAMP:
                    self._flush_buffer()

            elif self._in_speech:
                self._buffer.append(chunk)
                self._buffer_samples += len(chunk)
                self._silent_count += 1

                if self._silent_count >= self._silence_chunks:
                    self._flush_buffer()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SAMP,
            callback=_audio_callback,
        ):
            while self._running:
                sd.sleep(50)

    def _flush_buffer(self):
        """Send buffered audio to transcription queue, reset state."""
        if self._buffer and self._buffer_samples >= self._min_phrase_samp:
            audio = np.concatenate(self._buffer)

            # Drop old audio if queue is backed up — latency > completeness
            if self._audio_queue.full():
                try:
                    self._audio_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass

            try:
                self._audio_queue.put_nowait(audio)
            except queue.Full:
                pass  # drop this chunk too

        self._buffer = []
        self._buffer_samples = 0
        self._in_speech    = False
        self._silent_count = 0

    # ------------------------------------------------------------------
    # Transcription worker (separate thread, never blocks audio capture)
    # ------------------------------------------------------------------

    def _transcribe_loop(self):
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                segments, _ = self._model.transcribe(audio, language="en")
                text = " ".join(seg.text for seg in segments).strip()
                if text:
                    self._emit_words(text)
            except Exception as e:
                print(f"[stt] transcription error: {e}")

    def _emit_words(self, text: str):
        """Emit individual words with overlap dedup."""
        words = text.split()
        if not words:
            return

        # Overlap detection: if first word matches last emitted, drop it
        # (Whisper often repeats the last word of previous chunk)
        if self._last_emitted and words[0].lower() == list(self._last_emitted)[-1].lower():
            words = words[1:]

        for word in words:
            clean = word.strip()
            if not clean:
                continue

            # Skip if duplicate of last emitted word
            if self._last_emitted and clean.lower() == list(self._last_emitted)[-1].lower():
                continue

            self._last_emitted.append(clean.lower())
            self._on_text(clean)


if __name__ == "__main__":
    import sys

    def on_text(text: str) -> None:
        print(f"> {text}")

    stt = SpeechToText(on_text=on_text)
    print(f"Model: {stt.model_name}")
    print(f"Listening... speak into your mic. Ctrl+C to stop.\n")
    stt.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stt.stop()
        print("\nStopped.")
        sys.exit(0)
