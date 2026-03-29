#!/usr/bin/env python3
"""
Test RPM avatar rendering independently.
Run: python scripts/test_rpm_avatar.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.avatar.sign_library import SignLibrary
from src.avatar.rpm_renderer import RPMRenderer
from src.avatar.animation_engine import AnimationEngine
from src.avatar.rpm_controller import RPMAvatarController


def main():
    print("=" * 60)
    print("  RPM Avatar Test")
    print("=" * 60)

    # 1. Load avatar
    print("\n[1] Loading avatar GLB...")
    renderer = RPMRenderer("models/avatar.glb", width=1280, height=720)
    rpm_ok = renderer.load()
    print(f"    3D renderer: {'YES' if rpm_ok else 'NO (using skeleton fallback)'}")

    # 2. Load sign library
    print("\n[2] Loading sign animations...")
    library = SignLibrary()
    lib_ok = library.load()
    if not lib_ok:
        print("    FAILED — sign_animations.npz missing")
        print("    Generate synthetic data: python -c 'from src.avatar.sign_library import SignLibrary; SignLibrary.generate_synthetic()'")
        sys.exit(1)
    print(f"    Vocabulary: {len(library.vocabulary())} signs")

    # 3. Test animation engine
    print("\n[3] Testing animation engine...")
    engine = AnimationEngine()
    hello = library.get("hello")
    if hello is not None:
        rh, lh = engine.interpolate_sign(hello, 0.5)
        print(f"    interpolate_sign: right_hand={rh.shape}, left_hand={lh.shape}")
        rh2, lh2 = engine.idle_pose(0.0)
        print(f"    idle_pose: right_hand={rh2.shape}, left_hand={lh2.shape}")
    else:
        print("    WARNING: 'hello' not in library")

    # 4. Render test signs
    test_words = ["hello", "eat", "water", "yes", "please"]
    print(f"\n[4] Rendering {len(test_words)} signs...")

    os.makedirs("debug/rpm_test", exist_ok=True)
    total_frames = 0
    total_time = 0
    render_times = []

    for word in test_words:
        frames = library.get(word)
        if frames is None:
            print(f"    {word}: NOT FOUND in library")
            continue

        word_dir = f"debug/rpm_test/{word}"
        os.makedirs(word_dir, exist_ok=True)

        word_times = []
        for i in range(30):
            t = i / 29.0
            rh, lh = engine.interpolate_sign(frames, t)
            renderer.set_pose(rh, lh)

            t0 = time.perf_counter()
            frame = renderer.render()
            elapsed = (time.perf_counter() - t0) * 1000
            word_times.append(elapsed)

            cv2.imwrite(f"{word_dir}/frame_{i:03d}.png", frame)
            total_frames += 1

        avg_ms = sum(word_times) / len(word_times)
        render_times.extend(word_times)
        print(f"    {word}: 30 frames, avg {avg_ms:.1f}ms/frame")

    # 5. Test transition blending
    print("\n[5] Testing transition blend...")
    hello_frames = library.get("hello")
    eat_frames = library.get("eat")
    if hello_frames is not None and eat_frames is not None:
        trans_dir = "debug/rpm_test/transition"
        os.makedirs(trans_dir, exist_ok=True)
        for i in range(6):
            t = i / 5.0
            rh, lh = engine.blend_signs(hello_frames, eat_frames, t)
            renderer.set_pose(rh, lh)
            frame = renderer.render()
            cv2.imwrite(f"{trans_dir}/blend_{i:03d}.png", frame)
        print(f"    Saved 6 transition frames to {trans_dir}/")
    else:
        print("    Skipped (missing hello or eat)")

    # 6. Test controller state machine
    print("\n[6] Testing controller state machine...")
    controller = RPMAvatarController(renderer, library, engine)
    controller.queue_word("hello")
    controller.queue_word("water")

    controller_frames = 0
    controller_start = time.perf_counter()
    for _ in range(60):  # 2 seconds at 30fps
        frame = controller.get_frame()
        controller_frames += 1
        time.sleep(0.01)  # simulate frame timing
    controller_elapsed = (time.perf_counter() - controller_start) * 1000
    print(f"    {controller_frames} frames in {controller_elapsed:.0f}ms")
    print(f"    Current word: '{controller.current_word}'")
    print(f"    Queue length: {controller.queue_length}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  3D renderer:      {'Active' if rpm_ok else 'Skeleton fallback'}")
    print(f"  Signs rendered:    {len(test_words)}")
    print(f"  Total frames:      {total_frames}")
    if render_times:
        avg = sum(render_times) / len(render_times)
        mx = max(render_times)
        print(f"  Avg render time:   {avg:.1f}ms")
        print(f"  Max render time:   {mx:.1f}ms")
        print(f"  Budget (33ms):     {'PASS' if mx < 33 else 'FAIL'}")
    print(f"  Output dir:        debug/rpm_test/")
    print("=" * 60)


if __name__ == "__main__":
    main()
