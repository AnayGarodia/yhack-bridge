#!/usr/bin/env python3
"""
Virtual Camera Diagnosis — run this FIRST before anything else.
Run: python scripts/diagnose_virtual_cam.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=== VIRTUAL CAMERA DIAGNOSIS ===\n")

# Test 1: Is pyvirtualcam installed?
print("TEST 1: pyvirtualcam")
try:
    import pyvirtualcam
    print(f"  OK: pyvirtualcam {pyvirtualcam.__version__} installed")
except ImportError as e:
    print(f"  PROBLEM: {e}")
    print("  Fix: pip install pyvirtualcam")

# Test 2: Is OBS Virtual Camera backend available?
print("\nTEST 2: OBS Virtual Camera backend")
try:
    import pyvirtualcam
    with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
        print(f"  OK: Virtual camera opened: {cam.device}")
except Exception as e:
    print(f"  PROBLEM: {e}")
    print("  Fix: Open OBS -> click 'Start Virtual Camera' before running")

# Test 3: Can we push a test frame?
print("\nTEST 3: Push test frame")
try:
    import pyvirtualcam
    import numpy as np
    with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
        # Push 30 bright red frames
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # red channel (RGB)
        for _ in range(30):
            cam.send(frame)
            cam.sleep_until_next_frame()
        print("  OK: 30 test frames pushed")
        print("  CHECK: Open Google Meet or Photo Booth and look for a red screen")
        print(f"  Camera name: {cam.device}")
except Exception as e:
    print(f"  PROBLEM: {e}")

# Test 4: Check existing VirtualCamera class
print("\nTEST 4: VirtualCamera class")
try:
    from src.output.virtual_camera import VirtualCamera
    print(f"  OK: VirtualCamera imported")
    print(f"  is_available: {VirtualCamera.is_available()}")
except Exception as e:
    print(f"  PROBLEM: {e}")

# Test 5: Check HandRenderer (avatar frame source)
print("\nTEST 5: HandRenderer frame output")
try:
    from src.avatar.hand_renderer import HandRenderer
    renderer = HandRenderer(width=1280, height=720)
    renderer.set_word("HELLO")
    frame, done = renderer.next_frame()
    if frame is not None:
        print(f"  OK: Got frame shape {frame.shape}")
    else:
        print("  PROBLEM: next_frame() returned None")
except Exception as e:
    print(f"  PROBLEM: {e}")

# Test 6: Check AvatarPipeline
print("\nTEST 6: AvatarPipeline")
try:
    from src.avatar.avatar_pipeline import AvatarPipeline
    pipeline = AvatarPipeline()
    pipeline.on_word("hello")
    sign = pipeline.next_sign()
    print(f"  OK: 'hello' -> sign '{sign}'")
except Exception as e:
    print(f"  PROBLEM: {e}")

print("\n=== DIAGNOSIS COMPLETE ===")
print("Fix every PROBLEM above before proceeding")
