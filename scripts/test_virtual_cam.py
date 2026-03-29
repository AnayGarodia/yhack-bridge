#!/usr/bin/env python3
"""
Standalone test — opens the real webcam, overlays test text,
and pipes through the virtual camera for 30 seconds.

Usage:
    python scripts/test_virtual_cam.py
    python scripts/test_virtual_cam.py --camera 1
    python scripts/test_virtual_cam.py --duration 60

Then open Google Meet / Photo Booth / Zoom and select the virtual camera.
"""

import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.output.virtual_camera import VirtualCamera, _platform_setup_hint


def main():
    parser = argparse.ArgumentParser(description="Test the virtual camera pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--duration", type=int, default=30, help="Seconds to run")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # ---- Check availability ----
    if not VirtualCamera.is_available():
        print("\n[ERROR] pyvirtualcam is not installed.")
        print(_platform_setup_hint())
        sys.exit(1)

    # ---- Open real webcam ----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {args.camera}")
        sys.exit(1)

    # ---- Start virtual camera ----
    vcam = VirtualCamera(width=args.width, height=args.height, fps=args.fps)
    if not vcam.start():
        print("\n[ERROR] Could not start virtual camera.")
        print(_platform_setup_hint())
        cap.release()
        sys.exit(1)

    # ---- Instructions ----
    system = platform.system()
    cam_name = "OBS Virtual Camera" if system in ("Darwin", "Windows") else "v4l2loopback virtual cam"
    print(f"\n{'=' * 60}")
    print(f"  VIRTUAL CAMERA TEST — ASL BRIDGE")
    print(f"  Running for {args.duration} seconds")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"")
    print(f"  Open Google Meet, Zoom, or Photo Booth and select:")
    print(f"    \"{cam_name}\" as your camera input.")
    print(f"")
    print(f"  You should see your webcam feed with red overlay text.")
    print(f"  Press Ctrl+C to stop early.")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < args.duration:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Webcam read failed, retrying...")
                time.sleep(0.01)
                continue

            # Resize to target
            frame = cv2.resize(frame, (args.width, args.height))

            # Draw overlay
            elapsed = time.time() - start_time
            remaining = max(0, args.duration - int(elapsed))

            cv2.putText(
                frame, "VIRTUAL CAM TEST -- ASL BRIDGE",
                (80, args.height // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"Time remaining: {remaining}s",
                (80, args.height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"Frame: {frame_count}",
                (80, args.height // 2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1, cv2.LINE_AA,
            )

            vcam.send_frame(frame)
            frame_count += 1

            # Maintain target FPS
            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    vcam.stop()
    cap.release()
    print(f"\nDone. Sent {frame_count} frames in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
