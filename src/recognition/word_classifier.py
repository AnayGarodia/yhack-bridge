"""
Word-level ASL classifier using k-NN template matching.

Compares temporal-averaged landmarks against stored templates.
No neural network — just distance-based matching. Robust with small data.

Usage:
    clf = WordClassifier()
    clf.add_frame(lm543)
    sign, conf = clf.predict()
"""

import collections
import json
import os

import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "word_model.npz")
_LABEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "word_labels.json")

KEEP_SLICE  = slice(468, 543)    # hands + pose = 75 landmarks
N_FEATURES  = 75 * 3             # 225


class WordClassifier:
    """
    k-NN template matching classifier.

    Stores all training clip means. At inference, temporal-averages the
    last ~30 frames of hand+pose landmarks, normalizes, and finds the
    k=5 nearest training clips. Majority vote → prediction.
    """

    def __init__(self, model_path=None, confidence_threshold=0.35):
        self.confidence_threshold = confidence_threshold
        self.ready = False

        self._templates = None   # (N, 225) normalized clip means
        self._labels    = None   # (N,) int labels
        self._centroids = None   # (C, 225) per-class centroids
        self._feat_mean = None
        self._feat_std  = None
        self._idx_to_sign = {}

        self._buf  = collections.deque(maxlen=60)   # raw frames
        self._hist = collections.deque(maxlen=5)
        self._idle = 0
        self._call_n = 0
        self._raw  = (None, 0.0)
        self._last = (None, 0.0)

        model_path = model_path or _MODEL_PATH
        label_path = _LABEL_PATH

        if not os.path.exists(model_path):
            print(f"[WordClassifier] no model: {model_path}")
            print("[WordClassifier] Run: venv/bin/python src/recognition/sign_trainer.py")
            return

        try:
            data = np.load(model_path)
            self._templates = data["X_norm"]       # (N, 225)
            self._labels    = data["y"]             # (N,)
            self._centroids = data["centroids"]     # (C, 225)
            self._feat_mean = data["feat_mean"]     # (225,)
            self._feat_std  = data["feat_std"]      # (225,)
            n_classes = int(data["n_classes"])

            if os.path.exists(label_path):
                with open(label_path) as f:
                    d = json.load(f)
                self._idx_to_sign = {int(k): v for k, v in d.items()}
            else:
                self._idx_to_sign = {i: f"sign_{i}" for i in range(n_classes)}

            self.ready = True
            signs = [self._idx_to_sign[i] for i in range(n_classes)]
            print(f"[WordClassifier] k-NN with {len(self._templates)} templates, "
                  f"{n_classes} classes: {signs}")
        except Exception as exc:
            print(f"[WordClassifier] load failed: {exc}")

    # ------------------------------------------------------------------

    def add_frame(self, lm543):
        """Buffer one frame of (543, 3) landmarks."""
        frame = np.array(lm543, dtype=np.float32)
        np.nan_to_num(frame, copy=False, nan=0.0)
        frame = frame[KEEP_SLICE].reshape(-1)  # (225,)

        # Auto-reset if hands vanish
        has_data = np.abs(frame).sum() > 1.0
        if not has_data:
            self._idle += 1
            if self._idle > 15:
                self._buf.clear()
                self._hist.clear()
                self._raw = (None, 0.0)
                self._last = (None, 0.0)
            return
        self._idle = 0
        self._buf.append(frame)

    def predict(self):
        """Run k-NN matching on temporal-averaged recent frames."""
        if not self.ready or len(self._buf) < 8:
            self._raw = (None, 0.0)
            return (None, 0.0)

        # Temporal mean of last ~30 frames (matching recording length)
        frames = list(self._buf)[-30:]
        clip = np.stack(frames, axis=0)            # (T, 225)
        nonzero = np.abs(clip).sum(axis=1) > 1e-6
        if nonzero.any():
            query = clip[nonzero].mean(axis=0)     # (225,)
        else:
            self._raw = (None, 0.0)
            return (None, 0.0)

        # Normalize with training stats
        query_norm = (query - self._feat_mean) / self._feat_std

        # k-NN: distances to all templates
        dists = np.linalg.norm(self._templates - query_norm, axis=1)  # (N,)
        k = min(5, len(dists))
        nn_idx = np.argsort(dists)[:k]
        nn_labels = self._labels[nn_idx]
        nn_dists  = dists[nn_idx]

        # Weighted vote: closer neighbors count more
        n_classes = len(self._centroids)
        votes = np.zeros(n_classes)
        for label, dist in zip(nn_labels, nn_dists):
            votes[label] += 1.0 / (1.0 + dist)

        pred = int(votes.argmax())
        # Confidence: winning vote share (0-1)
        total_votes = votes.sum()
        conf = float(votes[pred] / total_votes) if total_votes > 0 else 0.0

        sign = self._idx_to_sign.get(pred, f"sign_{pred}")
        self._raw = (sign, conf)
        self._hist.append((sign, conf))

        if conf < self.confidence_threshold:
            self._last = (None, conf)
            return self._last

        # Majority-vote smoothing over recent predictions
        if len(self._hist) >= 3:
            signs = [s for s, _ in self._hist]
            winner = max(set(signs), key=signs.count)
            avg_conf = sum(c for s, c in self._hist if s == winner) / signs.count(winner)
            self._last = (winner, avg_conf)
        else:
            self._last = (sign, conf)

        return self._last

    def process_landmarks(self, lm543):
        """Convenience: add_frame + predict every 3 frames."""
        self.add_frame(lm543)
        self._call_n += 1
        if self._call_n % 3 == 0:
            return self.predict()
        return self._last

    @property
    def raw(self):
        return self._raw

    def reset(self):
        self._buf.clear()
        self._hist.clear()
        self._idle = 0
        self._call_n = 0
        self._raw  = (None, 0.0)
        self._last = (None, 0.0)


if __name__ == "__main__":
    import sys
    import time
    import cv2

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.recognition.holistic_tracker import HolisticTracker

    print("=== WordClassifier (k-NN) ===")
    clf = WordClassifier()
    if not clf.ready:
        print("No model. Run sign_trainer.py first.")
        sys.exit(1)

    tracker = HolisticTracker()
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open webcam {cam_idx}")
        sys.exit(1)

    print(f"Webcam {cam_idx}. 'q'=quit  'r'=reset\n")
    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        annotated, lm543, *_ = tracker.process_frame(frame)
        clf.add_frame(lm543)

        if frame_n % 3 == 0:
            t0 = time.perf_counter()
            sign, conf = clf.predict()
            dt = (time.perf_counter() - t0) * 1000
            raw_sign, raw_conf = clf.raw
            print(f"\r  raw={raw_sign or '—':15s} {raw_conf:.0%}  "
                  f"out={sign or '—':15s} {conf:.0%}  {dt:.1f}ms",
                  end="", flush=True)
            label = f"{raw_sign} ({raw_conf:.0%})" if raw_sign else "..."
            color = (0, 220, 0) if raw_conf > 0.6 else (0, 200, 255) if raw_conf > 0.3 else (100, 100, 200)
            cv2.putText(annotated, label, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("WordClassifier", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clf.reset()
            print("\n  [reset]")

    print()
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
