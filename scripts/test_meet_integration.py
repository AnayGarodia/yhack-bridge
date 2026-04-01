#!/usr/bin/env python3
"""
Full Google Meet integration test.
Run: python scripts/test_meet_integration.py

This will:
1. Start virtual camera
2. Push test frames for 3 seconds (red screen)
3. Push avatar frames with sign animations
4. Demo the Q&A script matching

Open Google Meet or Photo Booth while this runs.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2


def test_phase_1():
    """Phase 1: Virtual camera with red test frame."""
    print("\n--- Phase 1: Red test frame (3 seconds) ---")
    from src.output.virtual_camera import VirtualCamera

    cam = VirtualCamera(1280, 720, 30)
    if not cam.start():
        print("FAIL: Could not start virtual camera")
        print("Make sure OBS is open and Virtual Camera is started")
        return False

    print(f"Camera: {cam.device}")
    print("Check Google Meet or Photo Booth for a RED screen...")

    red_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255  # BGR red
    cv2.putText(red_frame, "VIRTUAL CAMERA TEST", (350, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    for i in range(90):  # 3s at 30fps
        cam.send_frame(red_frame)
        time.sleep(1 / 30)

    cam.stop()
    print("[PASS] Phase 1: Virtual camera sends frames")
    return True


def test_phase_2():
    """Phase 2: HandRenderer frames through virtual camera."""
    print("\n--- Phase 2: Avatar HandRenderer (5 seconds) ---")
    from src.output.virtual_camera import VirtualCamera
    from src.avatar.hand_renderer import HandRenderer

    cam = VirtualCamera(1280, 720, 30)
    if not cam.start():
        print("FAIL: Virtual camera")
        return False

    renderer = HandRenderer(width=1280, height=720)
    signs = ["HELLO", "NICE", "MEET", "YOU"]

    for sign in signs:
        print(f"  Signing: {sign}")
        renderer.set_word(sign)
        while not renderer.is_idle:
            frame, done = renderer.next_frame()
            cam.send_frame(frame)
            time.sleep(1 / 30)

    # Show idle for 1 second
    for _ in range(30):
        frame, _ = renderer.next_frame()
        cam.send_frame(frame)
        time.sleep(1 / 30)

    cam.stop()
    print("[PASS] Phase 2: Avatar signs rendered in virtual camera")
    return True


def test_phase_3():
    """Phase 3: Demo Q&A script matching."""
    print("\n--- Phase 3: Demo Q&A matching ---")
    from src.avatar.demo_script import find_demo_response

    test_cases = [
        ("hello", ["HELLO", "NICE", "MEET", "YOU"]),
        ("what is this", ["THIS", "HELP", "DEAF", "PEOPLE", "COMMUNICATE"]),
        ("how does it work", ["CAMERA", "WATCH", "HANDS", "COMPUTER", "UNDERSTAND"]),
        ("thank you", ["THANK-YOU", "VERY", "MUCH"]),
    ]

    for phrase, expected_signs in test_cases:
        result = find_demo_response(phrase)
        if result is None:
            print(f"  FAIL: '{phrase}' -> no match")
            return False
        signs, display = result
        if signs != expected_signs:
            print(f"  FAIL: '{phrase}' -> {signs} (expected {expected_signs})")
            return False
        print(f"  OK: '{phrase}' -> {signs} | '{display}'")

    # Test no match
    result = find_demo_response("asdfghjkl random nonsense")
    if result is not None:
        print(f"  FAIL: random text should not match")
        return False
    print(f"  OK: random text -> no match (correct)")

    print("[PASS] Phase 3: Demo Q&A matching works")
    return True


def test_phase_4():
    """Phase 4: AvatarPipeline.push_text() with demo matching."""
    print("\n--- Phase 4: Pipeline push_text integration ---")
    from src.avatar.avatar_pipeline import AvatarPipeline

    pipeline = AvatarPipeline()

    # Test demo match
    result = pipeline.push_text("hello")
    assert result is not None, "push_text('hello') should match demo"
    print(f"  OK: push_text('hello') -> '{result}'")

    # Drain the queue
    signs = []
    while True:
        s = pipeline.next_sign()
        if s is None:
            break
        signs.append(s)
    print(f"  OK: signs queued: {signs}")
    assert len(signs) > 0, "Should have queued signs"

    # Test normal word (no demo match)
    pipeline.push_text("xyzzy")
    # Should fall through to word-by-word (and skip unknown long word)
    print(f"  OK: push_text('xyzzy') -> normal word processing")

    print("[PASS] Phase 4: Pipeline demo integration works")
    return True


def test_phase_5():
    """Phase 5: Full pipeline - demo signs through virtual camera."""
    print("\n--- Phase 5: Full demo through virtual camera (8 seconds) ---")
    from src.output.virtual_camera import VirtualCamera
    from src.avatar.hand_renderer import HandRenderer
    from src.avatar.avatar_pipeline import AvatarPipeline

    cam = VirtualCamera(1280, 720, 30)
    if not cam.start():
        print("FAIL: Virtual camera")
        return False

    renderer = HandRenderer(width=1280, height=720)
    pipeline = AvatarPipeline()

    demo_phrases = ["hello", "what is this", "thank you"]

    for phrase in demo_phrases:
        display = pipeline.push_text(phrase)
        print(f"  Demo: '{phrase}' -> '{display}'")

        # Render all queued signs
        timeout = time.time() + 3.0
        while time.time() < timeout:
            if renderer.is_idle:
                next_sign = pipeline.next_sign()
                if next_sign is not None:
                    renderer.set_word(next_sign)
                elif pipeline.queue_depth == 0:
                    break

            frame, done = renderer.next_frame()

            # Add display text overlay
            if display:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, h - 80), (w, h - 40), (20, 20, 20), -1)
                cv2.putText(frame, display, (20, h - 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cam.send_frame(frame)
            time.sleep(1 / 30)

    cam.stop()
    print("[PASS] Phase 5: Full demo pipeline works")
    return True


if __name__ == "__main__":
    results = []

    for test in [test_phase_1, test_phase_2, test_phase_3, test_phase_4, test_phase_5]:
        try:
            ok = test()
            results.append((test.__name__, ok))
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    print(f"\n{'='*50}")
    print("RESULTS:")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{passed}/{len(results)} phases passed")

    if passed == len(results):
        print("\nAll tests passed! Avatar should be visible in Google Meet.")
    else:
        print("\nSome tests failed. Fix the issues above.")
        sys.exit(1)
