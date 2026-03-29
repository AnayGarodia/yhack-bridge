"""
SignRouter — unified interface for ASL recognition with continuous decoding.

How it works:
  1. While signing (hands visible) → collect top-5 prediction snapshots
  2. Every ~1.5s, reset the frame buffer so predictions stay fresh
  3. Every ~1.5s, send the prediction stream to the LLM decoder (async)
     → UI shows the evolving sentence in real-time ("hello" → "hello dad")
  4. When hands drop → final decode, commit all signs to the pipeline
"""

import collections
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Lazy imports — these pull in tensorflow/mediapipe which are slow
TFLiteClassifier = None
ASLClassifier = None

def _ensure_imports():
    global TFLiteClassifier, ASLClassifier
    if TFLiteClassifier is None:
        from src.recognition.tflite_classifier import TFLiteClassifier as _T
        from src.recognition.asl_classifier import ASLClassifier as _A
        TFLiteClassifier = _T
        ASLClassifier = _A

_MODE_WORD = "word"
_MODE_LETTER = "letter"
_MODE_IDLE = "idle"

# Hand presence thresholds
HAND_DROP_FRAMES = 3        # ~0.1s at 30fps
HAND_START_FRAMES = 2       # start signing after just 2 frames with hands

# Safety-net buffer reset (commit-on-stability handles normal transitions)
BUFFER_RESET_FRAMES = 60    # ~2s — only fires if no commit for a very long time

# Continuous decode interval (seconds) — async LLM preview only
DECODE_INTERVAL_S = 0.7

# Commit-on-stability: as soon as the same sign appears N times in a row, commit it.
# No need to wait for a sign change — reset buffer immediately and start fresh.
SIGN_STABLE_COUNT = 2       # 2 consecutive same predictions → commit
SAME_SIGN_COOLDOWN = 45     # frames (~1.5s) before same sign can be committed again

# Known model confusions: the TFLite model frequently misclassifies these.
# Maps raw model output → the sign that was actually meant.
# Applied at commit time so these never reach the output uncorrected.
_CONFUSION_REMAP = {
    "scissors": "name",     # model sees scissor shape → almost always means "name"
    "grass":    "please",   # grass handshape identical to please in this model
    "hat":      "think",    # hat tap → think/know confusion; think is far more common
    "elephant": "that",     # elephant trunk shape ≈ "that" pointing
    "sun":      "no",       # sun shape ≈ "no" shake
    "fireman":  "red",      # fireman hat ≈ "red" on lips
    "taste":    "food",     # taste fingers → food is more likely in conversation
    "empty":    "finish",   # empty sweep ≈ "finish" sweep
}


class SignRouter:
    def __init__(self, word_threshold=0.40, letter_threshold=0.55,
                 word_cooldown_s=1.5, sign_decoder=None):
        self.word_threshold = word_threshold
        self.letter_threshold = letter_threshold
        self.word_cooldown_s = word_cooldown_s

        self._word_clf = None
        self._letter_clf = None
        self._text_buffer = []

        # Conversation history: list of recent English sentences for context
        self._conversation_history = []

        # Sign decoder (LLM-based, optional)
        self._decoder = sign_decoder

        # State tracking
        self._hands_visible_count = 0
        self._no_hand_count = 0
        self._signing = False
        self._signing_frame_count = 0    # frames since signing started

        # Prediction stream: list of {"t": float, "top5": [(name, prob), ...]}
        self._prediction_stream = []

        # Stability tracking for mid-sign commits
        self._stable_sign = None        # sign consistently seen in recent predictions
        self._stable_count = 0          # consecutive matching prediction count
        self._last_commit_sign = None   # last committed sign (for cooldown)
        self._last_commit_frame = -999  # signing_frame_count when last committed

        # Live display state
        self._live_sign = None
        self._live_conf = 0.0
        self._live_top5 = []

        # Continuous decode state
        self._last_decode_time = 0.0
        self._decode_lock = threading.Lock()
        self._decode_running = False
        self._live_decoded = []          # latest decoded sentence (updated async)

        # Commit queue: decoded signs waiting to be emitted one per frame
        self._commit_queue = collections.deque()

        # Committed state
        self._last_committed = None
        self._last_commit_time = 0.0

    def open(self):
        _ensure_imports()
        if self._word_clf is None:
            self._word_clf = TFLiteClassifier(confidence_threshold=self.word_threshold)
            self._letter_clf = ASLClassifier()
            print(f"[SignRouter] ready={self._word_clf.ready}")

    def close(self):
        if self._word_clf is not None:
            self._word_clf.close()
            self._word_clf = None

    def process_frame(self, frame_bgr):
        """
        Returns:
            annotated_frame, sign_or_None, confidence, mode

        sign is non-None when:
          - A completed sign is committed (hands dropped → decoder ran)
          - A queued decoded sign is being drained (one per frame)
        """
        if self._word_clf is None:
            raise RuntimeError("Call SignRouter.open() first")

        # 0. Drain commit queue (one sign per frame)
        if self._commit_queue:
            queued_sign = self._commit_queue.popleft()
            annotated, _ = self._word_clf.process_frame(frame_bgr)
            return annotated, queued_sign, 0.9, _MODE_WORD

        # 1. Run holistic tracker + buffer landmarks
        annotated, hands_visible = self._word_clf.process_frame(frame_bgr)

        # 2. Track hand presence
        if hands_visible:
            self._hands_visible_count += 1
            self._no_hand_count = 0

            if self._hands_visible_count >= HAND_START_FRAMES and not self._signing:
                self._signing = True
                self._signing_frame_count = 0
                self._prediction_stream.clear()
                self._live_decoded = []
                self._last_decode_time = time.monotonic()
                self._stable_sign = None
                self._stable_count = 0
                self._last_commit_sign = None
                self._last_commit_frame = -999

            if self._signing:
                self._signing_frame_count += 1

                # Safety net: force reset if no sign committed in a long time
                if self._signing_frame_count % BUFFER_RESET_FRAMES == 0:
                    self._word_clf.on_sign_boundary()
                    self._stable_sign = None
                    self._stable_count = 0
        else:
            self._no_hand_count += 1
            self._hands_visible_count = 0

        # 3. Run async inference + collect predictions
        self._word_clf.maybe_run_async()
        (raw_sign, raw_conf), top5 = self._word_clf.get_async_result()

        if raw_sign is not None:
            self._live_sign = raw_sign
            self._live_conf = raw_conf
            self._live_top5 = top5

            # Append to prediction stream during signing
            if self._signing and top5:
                self._prediction_stream.append({
                    "t": time.monotonic(),
                    "top5": [(name, prob) for name, prob in top5],
                })

                # --- Commit-on-stability ---
                # Same sign N times in a row → commit it now, reset, start fresh.
                # No waiting for a sign-change signal — that was the bottleneck.
                if raw_sign == self._stable_sign:
                    self._stable_count += 1
                else:
                    self._stable_sign = raw_sign
                    self._stable_count = 1

                if self._stable_count >= SIGN_STABLE_COUNT:
                    frames_since = self._signing_frame_count - self._last_commit_frame
                    if (raw_sign != self._last_commit_sign or
                            frames_since >= SAME_SIGN_COOLDOWN):
                        # Use weighted vote on accumulated stream (handles confusions
                        # better than raw top-1), then apply known confusion remap.
                        best = self._pick_best_from_stream(self._prediction_stream)
                        commit_sign = _CONFUSION_REMAP.get(best[0] if best else raw_sign,
                                                           best[0] if best else raw_sign)
                        self._mid_sign_commit(commit_sign)
                        self._last_commit_sign = raw_sign  # track raw for cooldown
                        self._last_commit_frame = self._signing_frame_count
                        self._stable_sign = None
                        self._stable_count = 0
                        self._word_clf.on_sign_boundary()
                        self._prediction_stream.clear()

        # 4. Continuous decode: every DECODE_INTERVAL_S, decode the stream so far
        if (self._signing and self._decoder and
                len(self._prediction_stream) >= 3):
            now = time.monotonic()
            if (now - self._last_decode_time >= DECODE_INTERVAL_S and
                    not self._decode_running):
                self._last_decode_time = now
                stream_copy = list(self._prediction_stream)
                self._decode_running = True
                threading.Thread(
                    target=self._async_decode,
                    args=(stream_copy, False),
                    daemon=True,
                ).start()

        # 5. Hand drop → final decode and commit
        committed_sign, committed_conf, committed_mode = None, 0.0, _MODE_IDLE

        if (self._signing and
                self._no_hand_count >= HAND_DROP_FRAMES and
                len(self._prediction_stream) >= 1):

            signs = self._final_decode()

            if signs:
                # Commit first sign now, queue the rest
                committed_sign = signs[0]
                committed_conf = 0.9
                committed_mode = _MODE_WORD
                self._text_buffer.append(signs[0])
                print(f"[SignRouter] COMMIT: {signs[0]}")

                for s in signs[1:]:
                    self._commit_queue.append(s)
                    self._text_buffer.append(s)
                    print(f"[SignRouter] COMMIT (queued): {s}")

                self._last_committed = signs[-1]
                self._last_commit_time = time.monotonic()

            # Full reset
            self._signing = False
            self._signing_frame_count = 0
            self._prediction_stream.clear()
            self._live_decoded = []
            self._word_clf.on_sign_boundary()
            self._live_sign = None
            self._live_conf = 0.0
            self._stable_sign = None
            self._stable_count = 0

        return annotated, committed_sign, committed_conf, committed_mode

    def _mid_sign_commit(self, sign: str) -> None:
        """Commit a sign detected mid-signing (boundary detected while hands are still up)."""
        self._commit_queue.append(sign)
        self._text_buffer.append(sign)
        self._last_committed = sign
        self._last_commit_time = time.monotonic()
        print(f"[SignRouter] MID-COMMIT: {sign}")

    def add_to_history(self, sentence: str) -> None:
        """Add a completed English sentence to conversation history (for LLM context)."""
        if sentence:
            self._conversation_history.append(sentence)
            if len(self._conversation_history) > 5:
                self._conversation_history.pop(0)

    def _async_decode(self, stream, is_final):
        """Run decoder in background thread, update _live_decoded."""
        try:
            ctx = self._conversation_history[-3:] if self._conversation_history else None
            signs = self._decoder.decode(stream, context=ctx)
            with self._decode_lock:
                self._live_decoded = signs
        except Exception as e:
            print(f"[SignRouter] decode error: {e}")
        finally:
            self._decode_running = False

    def _final_decode(self):
        """Fast decode on hand-drop: weighted vote + confusion remap, no LLM blocking."""
        signs = self._pick_best_from_stream(list(self._prediction_stream))
        return [_CONFUSION_REMAP.get(s, s) for s in signs]

    def _pick_best_from_stream(self, stream):
        """Rank + frequency weighted vote across all snapshots.
        Considers top-3 candidates with decaying rank weights."""
        from src.translation.sign_decoder import _freq_weight
        if not stream:
            return []
        votes = {}
        for snap in stream:
            for rank, (name, prob) in enumerate(snap["top5"][:3]):
                rank_w = [1.0, 0.4, 0.15][rank]
                votes[name] = votes.get(name, 0.0) + prob * _freq_weight(name) * rank_w
        if not votes:
            return []
        return [max(votes, key=votes.get)]

    def get_live_display(self):
        """
        Get the current live prediction for UI display.
        Returns (sign, confidence, top5).
        """
        return self._live_sign, self._live_conf, self._live_top5

    def get_live_decoded(self):
        """
        Get the latest continuously-decoded sentence.
        Returns list of sign names, e.g. ["hello", "dad"].
        """
        with self._decode_lock:
            return list(self._live_decoded)

    @property
    def text_so_far(self):
        return " ".join(self._text_buffer)

    def reset_text(self):
        self._text_buffer.clear()
        if self._word_clf:
            self._word_clf.reset()
        self._last_committed = None
        self._signing = False
        self._signing_frame_count = 0
        self._prediction_stream.clear()
        self._live_decoded = []
        self._commit_queue.clear()
        self._live_sign = None
        self._live_conf = 0.0
        self._stable_sign = None
        self._stable_count = 0
        self._last_commit_sign = None
        self._last_commit_frame = -999
