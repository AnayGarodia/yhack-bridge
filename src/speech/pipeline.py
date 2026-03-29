"""
SpeechPipeline — buffers recognized ASL tokens and fires TTS after a pause.

Workflow:
  1. SignRouter emits a sign → call pipeline.on_sign(sign, mode)
  2. Pipeline buffers tokens
  3. After `pause_s` seconds of silence → smooth tokens → speak

Usage:
    pipeline = SpeechPipeline(smoother, tts, pause_s=2.0)
    pipeline.start()

    # Inside your recognition loop:
    _, sign, conf, mode = router.process_frame(frame)
    if sign:
        pipeline.on_sign(sign, mode)

    # When done:
    pipeline.flush_now()   # speak any remaining tokens
    pipeline.stop()

Standalone test (requires LAVA_TOKEN + ELEVENLABS_API_KEY env vars):
    python pipeline.py
"""

import os
import sys
import threading
import time

# Allow running as __main__ from anywhere in the repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.translation.text_smoother import TextSmoother
from src.speech.tts import TTSEngine


class SpeechPipeline:
    """
    Args:
        smoother:    TextSmoother instance for ASL→English conversion.
        tts:         TTSEngine instance for speech synthesis.
        pause_s:     Seconds of silence before flushing buffer (default 2.0).
        min_tokens:  Minimum tokens required before flushing (default 1).
    """

    def __init__(
        self,
        smoother: TextSmoother,
        tts: TTSEngine,
        pause_s: float = 2.0,
        min_tokens: int = 1,
    ):
        self._smoother = smoother
        self._tts = tts
        self._pause_s = pause_s
        self._min_tokens = min_tokens

        self._buffer: list[str] = []
        self._last_sign_t: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._flush_thread: threading.Thread | None = None
        self._history: list[str] = []   # last N smoothed English sentences

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background flush-loop thread."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="speech-pipeline"
        )
        self._flush_thread.start()

    def stop(self):
        """Stop the background thread (does not flush remaining tokens)."""
        self._running = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_sign(self, sign: str, mode: str = "word") -> None:
        """
        Called each time SignRouter emits a new token.

        Args:
            sign: recognized word or letter, e.g. "HELLO" or "A"
            mode: 'word' | 'letter' | 'idle'
        """
        if not sign or mode == "idle":
            return
        with self._lock:
            self._buffer.append(sign)
            self._last_sign_t = time.monotonic()

    def flush_now(self) -> None:
        """Force-flush any buffered tokens immediately (call on program exit)."""
        with self._lock:
            tokens = self._buffer[:]
            self._buffer.clear()
        if len(tokens) >= self._min_tokens:
            self._process(tokens)

    @property
    def buffered_text(self) -> str:
        """Current unspoken tokens as a string (for UI display)."""
        with self._lock:
            return " ".join(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_loop(self):
        while self._running:
            time.sleep(0.1)
            with self._lock:
                if not self._buffer:
                    continue
                elapsed = time.monotonic() - self._last_sign_t
                if elapsed < self._pause_s:
                    continue
                tokens = self._buffer[:]
                self._buffer.clear()

            if len(tokens) >= self._min_tokens:
                self._process(tokens)

    def _process(self, tokens: list[str]) -> None:
        try:
            raw = " ".join(tokens)
            print(f"[pipeline] smoothing: {raw!r}")
            ctx = self._history[-3:] if self._history else None
            text = self._smoother.smooth(tokens, context=ctx)
            if text:
                print(f"[pipeline] speaking:  {text!r}")
                self._tts.speak_async(text)
                self._history.append(text)
                if len(self._history) > 5:
                    self._history.pop(0)
        except Exception as e:
            print(f"[pipeline] error: {e}")


def _build_from_env() -> SpeechPipeline:
    lava_token = os.environ.get("LAVA_TOKEN", "")
    eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not lava_token:
        print("Set LAVA_TOKEN environment variable.")
        sys.exit(1)
    smoother = TextSmoother(lava_token=lava_token)
    tts = TTSEngine(eleven_api_key=eleven_key)  # eleven_key optional; falls back to pyttsx3
    return SpeechPipeline(smoother, tts)


if __name__ == "__main__":
    print("SpeechPipeline — interactive test")
    print("Simulating sign recognition stream...\n")

    pipeline = _build_from_env()
    pipeline.start()

    # Simulate a stream of ASL signs with realistic timing
    signs = [
        ("HELLO", "word", 0.5),
        ("MY",    "word", 0.5),
        ("NAME",  "word", 0.5),
        ("BRIDGE","word", 3.0),   # 3s pause → should flush and speak
        ("HELP",  "word", 0.5),
        ("PLEASE","word", 3.0),   # second sentence
    ]

    for sign, mode, delay in signs:
        print(f"  sign: {sign!r}")
        pipeline.on_sign(sign, mode)
        time.sleep(delay)

    pipeline.flush_now()
    time.sleep(5)   # wait for async TTS to finish
    pipeline.stop()
    print("Done.")
