"""
Hand tracking using MediaPipe Tasks API (mediapipe >= 0.10).
Downloads the hand_landmarker.task model on first run.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

_vision = mp.tasks.vision
_HandLandmarker = _vision.HandLandmarker
_HandLandmarkerOptions = _vision.HandLandmarkerOptions
_RunningMode = _vision.RunningMode
_BaseOptions = mp.tasks.BaseOptions
_draw = _vision.drawing_utils
_HandConnections = _vision.HandLandmarksConnections.HAND_CONNECTIONS

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading hand_landmarker.task model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        _ensure_model()

        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process_frame(self, frame):
        """
        Process a BGR OpenCV frame.

        Returns:
            annotated_frame: frame with hand landmarks drawn
            landmarks_list: list of hands, each hand is 21 (x, y, z) tuples in pixel coords
            normalized_landmarks: same but coordinates relative to wrist (landmark 0)
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        annotated_frame = frame.copy()
        landmarks_list = []
        normalized_landmarks = []

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # Draw landmarks
                _draw.draw_landmarks(
                    annotated_frame,
                    hand_lms,
                    _HandConnections,
                    landmark_drawing_spec=_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                )

                # Raw landmarks in pixel coordinates
                hand = [(lm.x * w, lm.y * h, lm.z * w) for lm in hand_lms]
                landmarks_list.append(hand)

                # Normalize relative to wrist (landmark 0)
                wx, wy, wz = hand[0]
                normalized = [(x - wx, y - wy, z - wz) for x, y, z in hand]
                normalized_landmarks.append(normalized)

        return annotated_frame, landmarks_list, normalized_landmarks

    def close(self):
        self._landmarker.close()


if __name__ == "__main__":
    tracker = HandTracker()
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: could not open webcam")
        raise SystemExit(1)

    print("HandTracker running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame")
            break

        annotated, landmarks, normalized = tracker.process_frame(frame)

        for i, hand in enumerate(landmarks):
            wrist = hand[0]
            cv2.putText(
                annotated,
                f"Hand {i + 1} wrist: ({wrist[0]:.0f}, {wrist[1]:.0f})",
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("HandTracker", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
