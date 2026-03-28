"""
MediaPipe HolisticLandmarker — extracts all 543 landmarks per frame.

Returns landmarks in Kaggle ASL competition layout:
    indices 0-467   : face (468 landmarks)
    indices 468-488 : left hand (21 landmarks)
    indices 489-521 : pose (33 landmarks)
    indices 522-542 : right hand (21 landmarks)

Also exposes left_hand_21 / right_hand_21 for drop-in use with ASLClassifier.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

_vision = mp.tasks.vision
_HolisticLandmarker = _vision.HolisticLandmarker
_HolisticLandmarkerOptions = _vision.HolisticLandmarkerOptions
_RunningMode = _vision.RunningMode
_BaseOptions = mp.tasks.BaseOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "holistic_landmarker.task")

# Kaggle layout sizes
_N_FACE  = 468
_N_LHAND = 21
_N_POSE  = 33
_N_RHAND = 21
_N_TOTAL = _N_FACE + _N_LHAND + _N_POSE + _N_RHAND  # 543

# Slices for each group in the (543, 3) array
FACE_SLICE  = slice(0,   _N_FACE)
LHAND_SLICE = slice(_N_FACE, _N_FACE + _N_LHAND)
POSE_SLICE  = slice(_N_FACE + _N_LHAND, _N_FACE + _N_LHAND + _N_POSE)
RHAND_SLICE = slice(_N_FACE + _N_LHAND + _N_POSE, _N_TOTAL)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading holistic_landmarker.task → {_MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


def _lm_list_to_array(lm_list, n_expected):
    """Convert mediapipe landmark list → float32 ndarray (n, 3). Returns NaN array if None."""
    if lm_list is None:
        return np.full((n_expected, 3), np.nan, dtype=np.float32)
    return np.array([(lm.x, lm.y, lm.z) for lm in lm_list], dtype=np.float32)


class HolisticTracker:
    """
    Drop-in replacement / extension of HandTracker that also provides
    pose and face landmarks for the word-level TFLite model.
    """

    def __init__(self, detection_confidence=0.5, presence_confidence=0.5,
                 tracking_confidence=0.5):
        _ensure_model()
        options = _HolisticLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_RunningMode.VIDEO,
            min_face_detection_confidence=detection_confidence,
            min_face_landmarks_confidence=presence_confidence,
            min_pose_detection_confidence=detection_confidence,
            min_pose_landmarks_confidence=presence_confidence,
            min_hand_landmarks_confidence=presence_confidence,
        )
        self._landmarker = _HolisticLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process_frame(self, frame_bgr):
        """
        Process one BGR frame.

        Returns:
            annotated_frame : frame with landmarks drawn
            landmarks_543   : ndarray (543, 3) in Kaggle order — NaN where undetected
            left_hand_21    : list of 21 (x,y,z) pixel-coord tuples, or None
            right_hand_21   : list of 21 (x,y,z) pixel-coord tuples, or None
            norm_left       : wrist-relative normalized left hand landmarks, or None
            norm_right      : wrist-relative normalized right hand landmarks, or None
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        annotated = frame_bgr.copy()

        # ---- Build 543-landmark array ----------------------------------------
        lm543 = np.full((_N_TOTAL, 3), np.nan, dtype=np.float32)

        face_lms  = result.face_landmarks
        pose_lms  = result.pose_landmarks
        lhand_lms = result.left_hand_landmarks
        rhand_lms = result.right_hand_landmarks

        if face_lms:
            # Tasks API HolisticLandmarker returns 478 face landmarks;
            # Kaggle layout expects 468 — truncate to first 468.
            lm543[FACE_SLICE]  = _lm_list_to_array(face_lms,  _N_FACE)[:_N_FACE]
        if lhand_lms:
            lm543[LHAND_SLICE] = _lm_list_to_array(lhand_lms, _N_LHAND)
        if pose_lms:
            lm543[POSE_SLICE]  = _lm_list_to_array(pose_lms,  _N_POSE)
        if rhand_lms:
            lm543[RHAND_SLICE] = _lm_list_to_array(rhand_lms, _N_RHAND)

        # ---- Per-hand pixel-coord lists (for ASLClassifier) ------------------
        def to_pixel(lm_arr):
            return [(lm.x * w, lm.y * h, lm.z * w) for lm in lm_arr] if lm_arr else None

        def normalize_wrist(pixel_list):
            if not pixel_list:
                return None
            wx, wy, wz = pixel_list[0]
            return [(x - wx, y - wy, z - wz) for x, y, z in pixel_list]

        lhand_px = to_pixel(lhand_lms)
        rhand_px = to_pixel(rhand_lms)
        norm_left  = normalize_wrist(lhand_px)
        norm_right = normalize_wrist(rhand_px)

        # ---- Draw skeleton ---------------------------------------------------
        draw = _vision.drawing_utils
        styles = _vision.drawing_styles

        if rhand_lms:
            draw.draw_landmarks(
                annotated, rhand_lms,
                _vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=draw.DrawingSpec(color=(255, 255, 255), thickness=1),
            )
        if lhand_lms:
            draw.draw_landmarks(
                annotated, lhand_lms,
                _vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=draw.DrawingSpec(color=(200, 200, 200), thickness=1),
            )

        return annotated, lm543, lhand_px, rhand_px, norm_left, norm_right

    def close(self):
        self._landmarker.close()


if __name__ == "__main__":
    tracker = HolisticTracker()
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Error: could not open webcam (index {cam_idx})")
        raise SystemExit(1)

    print("HolisticTracker running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, lm543, lhand, rhand, norm_l, norm_r = tracker.process_frame(frame)

        detected = []
        face_ok  = not np.isnan(lm543[0, 0])
        pose_ok  = not np.isnan(lm543[POSE_SLICE][0, 0])
        if face_ok:  detected.append("face")
        if pose_ok:  detected.append("pose")
        if lhand:    detected.append("lhand")
        if rhand:    detected.append("rhand")

        label = "  ".join(detected) if detected else "nothing detected"
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow("HolisticTracker", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
