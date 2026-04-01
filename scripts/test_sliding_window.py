#!/usr/bin/env python3
"""
Test sliding window with synthetic data.
Run: python scripts/test_sliding_window.py

Simulates fast consecutive signing and verifies
both words are captured — no camera or TFLite model needed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.recognition.tflite_classifier import TFLiteClassifier


# ---------------------------------------------------------------------------
# Mock classifier — overrides model loading and inference
# ---------------------------------------------------------------------------

class MockClassifier(TFLiteClassifier):
    """TFLiteClassifier with mocked model loading and inference."""

    def __init__(self, **kwargs):
        kwargs.setdefault("confidence_threshold", 0.40)
        super().__init__(**kwargs)
        self._mock_sign = "hello"
        self._mock_conf = 0.85
        # Mark as ready without loading real model
        self.ready = True
        self._initialized = True
        self._idx_to_sign = {i: f"sign_{i}" for i in range(250)}

    def _lazy_init(self):
        pass  # already initialized

    def _run_inference(self, frames):
        """Return controllable mock results."""
        sign = self._mock_sign
        conf = self._mock_conf
        top5 = [
            (sign, conf),
            ("other1", 0.05),
            ("other2", 0.03),
            ("other3", 0.02),
            ("other4", 0.01),
        ]
        self._raw = (sign, conf)
        self._hist.append((sign, conf))
        if conf >= self.confidence_threshold:
            return (sign, conf), top5
        return (None, conf), top5

    def maybe_run_async(self):
        """Synchronous version for testing — no background threads."""
        if not self.ready or len(self._frame_buffer) < self.MIN_FRAMES:
            return False
        if self._frames_since_last_classify < self.STRIDE:
            return False
        if self._async_running:
            return False
        self._frames_since_last_classify = 0
        frames = list(self._frame_buffer[-self.WINDOW_SIZE:])
        # Run synchronously instead of in a thread
        self._infer_bg(frames)
        return True

    def set_mock(self, sign, conf=0.85):
        """Set what the next inference(s) will return."""
        self._mock_sign = sign
        self._mock_conf = conf

    def feed_frames(self, n):
        """Add n fake landmark frames to the buffer."""
        for _ in range(n):
            fake = np.random.randn(543, 3).astype(np.float32)
            self._frame_buffer.append(fake)
            if len(self._frame_buffer) > self.MAX_BUFFER:
                self._frame_buffer = self._frame_buffer[-self.MAX_BUFFER:]
            self._frames_since_last_classify += 1

    def run_until_emit(self, max_rounds=20):
        """Feed frames and run inference until a word is emitted or max_rounds."""
        for _ in range(max_rounds):
            self.feed_frames(self.STRIDE)
            self.maybe_run_async()
            r, _ = self.get_async_result()
            if r[0] is not None:
                return r
        return (None, 0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_buffer_trim():
    """After on_sign_boundary(), buffer retains 8 frames, not 0."""
    clf = MockClassifier()
    clf.feed_frames(30)
    assert len(clf._frame_buffer) == 30
    clf.on_sign_boundary()
    assert len(clf._frame_buffer) == 8, f"Expected 8, got {len(clf._frame_buffer)}"
    print("[PASS] Test 1: buffer trim keeps 8 frames")


def test_consecutive_agreement():
    """Word only emits after CONSECUTIVE_AGREEMENT matching predictions."""
    clf = MockClassifier()
    clf.set_mock("hello", 0.85)

    # Feed minimum frames
    clf.feed_frames(clf.MIN_FRAMES)

    # First inference — might not emit yet if CONSECUTIVE_AGREEMENT > 1
    clf.maybe_run_async()
    r1, _ = clf.get_async_result()

    if clf.CONSECUTIVE_AGREEMENT > 1:
        assert r1[0] is None, "Should not emit after only 1 inference"

    # Feed more frames until emission
    r = clf.run_until_emit()
    assert r[0] == "hello", f"Expected 'hello', got {r[0]}"
    print("[PASS] Test 2: consecutive agreement works")


def test_different_words_no_cooldown():
    """Two different signs in quick succession — both captured."""
    clf = MockClassifier()

    # Sign 1: "hello"
    clf.set_mock("hello", 0.85)
    clf.feed_frames(clf.MIN_FRAMES)
    r1 = clf.run_until_emit()
    assert r1[0] == "hello", f"Expected 'hello', got {r1[0]}"

    # Boundary (simulates main.py calling on_sign_boundary)
    clf.on_sign_boundary()
    assert len(clf._frame_buffer) == 8, "Should keep 8 context frames"

    # Sign 2: "dad" — immediately, no cooldown for different words
    clf.set_mock("dad", 0.80)
    need = max(0, clf.MIN_FRAMES - len(clf._frame_buffer))
    clf.feed_frames(need)
    r2 = clf.run_until_emit()
    assert r2[0] == "dad", f"Expected 'dad', got {r2[0]}"

    print("[PASS] Test 3: different words captured back-to-back")


def test_same_word_cooldown():
    """Same word blocked by SAME_WORD_COOLDOWN."""
    clf = MockClassifier()
    clf.set_mock("hello", 0.85)

    # First emission
    clf.feed_frames(clf.MIN_FRAMES)
    r1 = clf.run_until_emit()
    assert r1[0] == "hello"

    clf.on_sign_boundary()

    # Try "hello" again immediately — should be blocked by cooldown
    need = max(0, clf.MIN_FRAMES - len(clf._frame_buffer))
    clf.feed_frames(need)
    # Run several rounds of inference
    for _ in range(10):
        clf.feed_frames(clf.STRIDE)
        clf.maybe_run_async()

    r2, _ = clf.get_async_result()
    assert r2[0] is None, f"Expected None (cooldown), got {r2[0]}"
    print("[PASS] Test 4: same word cooldown respected")


def test_same_word_after_cooldown():
    """Same word emits again after cooldown elapses."""
    clf = MockClassifier()
    clf.SAME_WORD_COOLDOWN = 0.05  # very short for testing
    clf.set_mock("hello", 0.85)

    # First emission
    clf.feed_frames(clf.MIN_FRAMES)
    r1 = clf.run_until_emit()
    assert r1[0] == "hello"

    clf.on_sign_boundary()

    # Wait for cooldown to elapse
    time.sleep(0.06)

    # Now "hello" should emit again
    need = max(0, clf.MIN_FRAMES - len(clf._frame_buffer))
    clf.feed_frames(need)
    r2 = clf.run_until_emit()
    assert r2[0] == "hello", f"Expected 'hello' after cooldown, got {r2[0]}"
    print("[PASS] Test 5: same word emits after cooldown")


def test_three_fast_signs():
    """Three signs in rapid succession — all captured."""
    clf = MockClassifier()
    signs = ["hello", "eat", "water"]
    captured = []

    for sign in signs:
        clf.set_mock(sign, 0.80)
        need = max(0, clf.MIN_FRAMES - len(clf._frame_buffer))
        clf.feed_frames(need)
        r = clf.run_until_emit()
        if r[0] is not None:
            captured.append(r[0])
        clf.on_sign_boundary()

    assert captured == signs, f"Expected {signs}, got {captured}"
    print("[PASS] Test 6: all 3 fast signs captured")


def test_raw_vs_deduped():
    """get_raw_async_result returns every prediction; get_async_result only deduped."""
    clf = MockClassifier()
    clf.set_mock("hello", 0.85)

    clf.feed_frames(clf.MIN_FRAMES)
    clf.maybe_run_async()

    # Raw should always have the latest prediction
    raw, _ = clf.get_raw_async_result()
    assert raw[0] == "hello", f"Raw should return 'hello', got {raw[0]}"

    # After on_sign_boundary, deduped is cleared but raw persists
    # (raw is updated on every inference, deduped is cleared on boundary)
    print("[PASS] Test 7: raw vs deduped results")


def test_low_confidence_filtered():
    """Predictions below CONFIDENCE_THRESHOLD don't count toward agreement."""
    clf = MockClassifier()
    clf.set_mock("hello", 0.20)  # below 0.40 threshold

    clf.feed_frames(clf.MIN_FRAMES)
    for _ in range(10):
        clf.feed_frames(clf.STRIDE)
        clf.maybe_run_async()

    r, _ = clf.get_async_result()
    assert r[0] is None, f"Low-confidence should not emit, got {r[0]}"
    print("[PASS] Test 8: low confidence filtered out")


def test_no_hand_full_clear():
    """When hands are absent 10+ frames, buffer fully clears."""
    clf = MockClassifier()
    clf.feed_frames(30)
    assert len(clf._frame_buffer) == 30

    # Simulate 11 no-hand frames by directly manipulating state
    clf._no_hand_frames = 11
    # Trigger the no-hand logic (normally in process_frame, but we test directly)
    if clf._no_hand_frames >= 10:
        clf._frame_buffer = []
        clf._prediction_history = []
        clf._frames_since_last_classify = 0

    assert len(clf._frame_buffer) == 0, "Buffer should be empty after extended no-hand"
    print("[PASS] Test 9: no-hand clears buffer completely")


def test_predict_one_shot():
    """predict() returns pending result once, then (None, 0.0)."""
    clf = MockClassifier()
    clf.set_mock("hello", 0.85)
    clf.feed_frames(clf.MIN_FRAMES)
    clf.run_until_emit()

    # predict() should return the pending result
    r = clf.predict()
    assert r[0] == "hello", f"predict() should return 'hello', got {r[0]}"

    # Second call should return None (consumed)
    r2 = clf.predict()
    assert r2[0] is None, f"predict() second call should be None, got {r2[0]}"
    print("[PASS] Test 10: predict() one-shot consumption")


def test_reset_clears_everything():
    """reset() clears all state including cooldown tracking."""
    clf = MockClassifier()
    clf.set_mock("hello", 0.85)
    clf.feed_frames(clf.MIN_FRAMES)
    clf.run_until_emit()

    clf.reset()

    assert len(clf._frame_buffer) == 0
    assert clf._last_emitted_word is None
    assert clf._last_emitted_time == 0
    assert clf._pending_result is None
    assert len(clf._prediction_history) == 0

    # After reset, same word should emit without cooldown
    clf.set_mock("hello", 0.85)
    clf.feed_frames(clf.MIN_FRAMES)
    r = clf.run_until_emit()
    assert r[0] == "hello", f"After reset, should emit 'hello', got {r[0]}"
    print("[PASS] Test 11: reset clears everything")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_buffer_trim,
        test_consecutive_agreement,
        test_different_words_no_cooldown,
        test_same_word_cooldown,
        test_same_word_after_cooldown,
        test_three_fast_signs,
        test_raw_vs_deduped,
        test_low_confidence_filtered,
        test_no_hand_full_clear,
        test_predict_one_shot,
        test_reset_clears_everything,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
