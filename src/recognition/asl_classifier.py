"""
Rule-based ASL fingerspelling classifier (A-Z).
Uses geometric features from 21 MediaPipe hand landmarks.
"""

import math

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_DIPS = [INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]


def _dist(a, b):
    """Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _dist2d(a, b):
    """Euclidean distance in x,y only."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class ASLClassifier:
    """Classifies ASL fingerspelling letters from MediaPipe hand landmarks."""

    def __init__(self):
        self._rules = self._build_rules()

    def classify(self, normalized_landmarks):
        """
        Classify a single hand's normalized landmarks into an ASL letter.

        Args:
            normalized_landmarks: list of 21 (x, y, z) tuples, wrist-relative.

        Returns:
            (letter, confidence) where letter is 'A'-'Z' and confidence is 0.0-1.0.
        """
        lm = normalized_landmarks
        features = self._extract_features(lm)

        best_letter = "?"
        best_score = -1.0

        for letter, rule_fn in self._rules.items():
            score = rule_fn(lm, features)
            score = max(0.0, min(1.0, score))
            if score > best_score:
                best_score = score
                best_letter = letter

        return best_letter, best_score

    def _extract_features(self, lm):
        """Extract geometric features used by multiple rules."""
        # Hand scale: distance from wrist to middle MCP
        hand_scale = _dist(lm[WRIST], lm[MIDDLE_MCP])
        if hand_scale < 1e-6:
            hand_scale = 1.0

        # Finger extended: tip farther from wrist than PIP
        def finger_extended(tip, pip):
            return _dist(lm[tip], lm[WRIST]) > _dist(lm[pip], lm[WRIST]) * 1.1

        fingers_extended = [
            finger_extended(FINGER_TIPS[i], FINGER_PIPS[i]) for i in range(4)
        ]

        # Finger curl ratio: tip-to-mcp distance / pip-to-mcp distance
        def curl_ratio(tip, pip, mcp):
            d_tip = _dist(lm[tip], lm[mcp])
            d_pip = _dist(lm[pip], lm[mcp])
            return d_tip / d_pip if d_pip > 1e-6 else 0.0

        curl_ratios = [
            curl_ratio(FINGER_TIPS[i], FINGER_PIPS[i], FINGER_MCPS[i])
            for i in range(4)
        ]

        # Thumb extended: tip farther from palm center than thumb MCP
        palm_center_x = sum(lm[m][0] for m in FINGER_MCPS) / 4
        palm_center_y = sum(lm[m][1] for m in FINGER_MCPS) / 4
        palm_center_z = sum(lm[m][2] for m in FINGER_MCPS) / 4
        palm_center = (palm_center_x, palm_center_y, palm_center_z)

        thumb_extended = _dist(lm[THUMB_TIP], palm_center) > _dist(lm[THUMB_MCP], palm_center) * 1.1

        # Thumb-to-fingertip distances (normalized by hand scale)
        thumb_to_tips = [
            _dist(lm[THUMB_TIP], lm[tip]) / hand_scale for tip in FINGER_TIPS
        ]

        # Adjacent fingertip spread (normalized)
        tip_spreads = [
            _dist(lm[FINGER_TIPS[i]], lm[FINGER_TIPS[i + 1]]) / hand_scale
            for i in range(3)
        ]

        # Fingertip to palm distances (normalized)
        tips_to_palm = [
            _dist(lm[tip], palm_center) / hand_scale for tip in FINGER_TIPS
        ]

        # Thumb tip to index MCP distance (normalized)
        thumb_to_index_mcp = _dist(lm[THUMB_TIP], lm[INDEX_MCP]) / hand_scale

        return {
            "hand_scale": hand_scale,
            "fingers_extended": fingers_extended,
            "curl_ratios": curl_ratios,
            "thumb_extended": thumb_extended,
            "thumb_to_tips": thumb_to_tips,
            "tip_spreads": tip_spreads,
            "tips_to_palm": tips_to_palm,
            "thumb_to_index_mcp": thumb_to_index_mcp,
            "palm_center": palm_center,
        }

    def _build_rules(self):
        """Build rule functions for each ASL letter."""
        return {
            "A": self._rule_A,
            "B": self._rule_B,
            "C": self._rule_C,
            "D": self._rule_D,
            "E": self._rule_E,
            "F": self._rule_F,
            "G": self._rule_G,
            "H": self._rule_H,
            "I": self._rule_I,
            "J": self._rule_J,
            "K": self._rule_K,
            "L": self._rule_L,
            "M": self._rule_M,
            "N": self._rule_N,
            "O": self._rule_O,
            "P": self._rule_P,
            "Q": self._rule_Q,
            "R": self._rule_R,
            "S": self._rule_S,
            "T": self._rule_T,
            "U": self._rule_U,
            "V": self._rule_V,
            "W": self._rule_W,
            "X": self._rule_X,
            "Y": self._rule_Y,
            "Z": self._rule_Z,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _all_curled(f, ext):
        """Score for all four fingers being curled."""
        return sum(1 for e in ext if not e) / 4.0

    @staticmethod
    def _count_extended(ext):
        return sum(1 for e in ext if e)

    # ── Letter rules ─────────────────────────────────────────────────────
    # Each returns a score 0.0–1.0.

    def _rule_A(self, lm, f):
        """A: Fist, thumb alongside (not over fingers). All fingers curled, thumb up beside index."""
        ext = f["fingers_extended"]
        score = 0.0
        # All four fingers curled
        score += 0.4 * self._all_curled(f, ext)
        # Thumb extended or alongside (not tucked under)
        if f["thumb_extended"]:
            score += 0.3
        # Thumb tip near index MCP (alongside fist)
        if f["thumb_to_index_mcp"] < 0.8:
            score += 0.3
        return score

    def _rule_B(self, lm, f):
        """B: All four fingers extended and together, thumb tucked across palm."""
        ext = f["fingers_extended"]
        score = 0.0
        # All four fingers extended
        score += 0.4 * (self._count_extended(ext) / 4.0)
        # Fingers together (small spread)
        avg_spread = sum(f["tip_spreads"]) / 3.0
        if avg_spread < 0.5:
            score += 0.3
        # Thumb tucked (not extended)
        if not f["thumb_extended"]:
            score += 0.3
        return score

    def _rule_C(self, lm, f):
        """C: Curved hand forming a C. Fingers partially curled, thumb opposed."""
        score = 0.0
        # Fingers partially extended (curl ratios between 1.0 and 2.0)
        mid_curl = sum(1 for cr in f["curl_ratios"] if 1.0 < cr < 2.5) / 4.0
        score += 0.4 * mid_curl
        # Thumb tip away from fingers but not fully extended
        if 0.5 < f["thumb_to_tips"][0] < 1.5:
            score += 0.3
        # Fingertips roughly same distance from palm (curved uniformly)
        if len(f["tips_to_palm"]) == 4:
            spread = max(f["tips_to_palm"]) - min(f["tips_to_palm"])
            if spread < 0.5:
                score += 0.3
        return score

    def _rule_D(self, lm, f):
        """D: Index extended, others curled, thumb touches middle finger."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index extended
        if ext[0]:
            score += 0.35
        # Middle, ring, pinky curled
        score += 0.35 * (sum(1 for e in ext[1:] if not e) / 3.0)
        # Thumb tip near middle fingertip
        if f["thumb_to_tips"][1] < 0.6:
            score += 0.3
        return score

    def _rule_E(self, lm, f):
        """E: All fingers curled down, thumb tucked across."""
        ext = f["fingers_extended"]
        score = 0.0
        # All fingers curled
        score += 0.4 * self._all_curled(f, ext)
        # Thumb tucked
        if not f["thumb_extended"]:
            score += 0.3
        # Fingertips close to palm
        avg_palm_dist = sum(f["tips_to_palm"]) / 4.0
        if avg_palm_dist < 0.8:
            score += 0.3
        return score

    def _rule_F(self, lm, f):
        """F: Index and thumb form circle, middle/ring/pinky extended."""
        ext = f["fingers_extended"]
        score = 0.0
        # Middle, ring, pinky extended
        score += 0.3 * (sum(1 for e in ext[1:] if e) / 3.0)
        # Index curled or touching thumb
        if not ext[0] or f["thumb_to_tips"][0] < 0.5:
            score += 0.3
        # Thumb tip close to index tip
        if f["thumb_to_tips"][0] < 0.5:
            score += 0.4
        return score

    def _rule_G(self, lm, f):
        """G: Index pointing sideways, thumb parallel. Hand oriented sideways."""
        ext = f["fingers_extended"]
        hs = f["hand_scale"]
        score = 0.0
        # Index extended
        if ext[0]:
            score += 0.25
        # Middle, ring, pinky curled
        score += 0.25 * (sum(1 for e in ext[1:] if not e) / 3.0)
        # Index pointing sideways (large x component relative to y)
        idx_dir_x = abs(lm[INDEX_TIP][0] - lm[INDEX_MCP][0])
        idx_dir_y = abs(lm[INDEX_TIP][1] - lm[INDEX_MCP][1])
        if idx_dir_x > idx_dir_y:
            score += 0.25
        # Thumb extended alongside
        if f["thumb_extended"]:
            score += 0.25
        return score

    def _rule_H(self, lm, f):
        """H: Index and middle pointing sideways."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index and middle extended
        if ext[0] and ext[1]:
            score += 0.3
        # Ring and pinky curled
        score += 0.2 * (sum(1 for e in ext[2:] if not e) / 2.0)
        # Fingers pointing sideways
        idx_dir_x = abs(lm[INDEX_TIP][0] - lm[INDEX_MCP][0])
        idx_dir_y = abs(lm[INDEX_TIP][1] - lm[INDEX_MCP][1])
        if idx_dir_x > idx_dir_y:
            score += 0.25
        mid_dir_x = abs(lm[MIDDLE_TIP][0] - lm[MIDDLE_MCP][0])
        mid_dir_y = abs(lm[MIDDLE_TIP][1] - lm[MIDDLE_MCP][1])
        if mid_dir_x > mid_dir_y:
            score += 0.25
        return score

    def _rule_I(self, lm, f):
        """I: Pinky extended, rest curled in fist."""
        ext = f["fingers_extended"]
        score = 0.0
        # Pinky extended
        if ext[3]:
            score += 0.4
        # Index, middle, ring curled
        score += 0.3 * (sum(1 for e in ext[:3] if not e) / 3.0)
        # Thumb tucked or alongside
        if not f["thumb_extended"]:
            score += 0.3
        return score

    def _rule_J(self, lm, f):
        """J: Same as I (static approximation — J involves motion)."""
        # J is I with a downward scoop motion; statically same as I but lower confidence
        return self._rule_I(lm, f) * 0.3

    def _rule_K(self, lm, f):
        """K: Index and middle up in V shape, thumb between them."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index and middle extended
        if ext[0] and ext[1]:
            score += 0.3
        # Ring and pinky curled
        score += 0.2 * (sum(1 for e in ext[2:] if not e) / 2.0)
        # Fingers spread (V shape)
        if f["tip_spreads"][0] > 0.4:
            score += 0.2
        # Thumb between index and middle (thumb tip near index/middle base)
        thumb_to_idx_pip = _dist(lm[THUMB_TIP], lm[INDEX_PIP]) / f["hand_scale"]
        if thumb_to_idx_pip < 0.6:
            score += 0.3
        return score

    def _rule_L(self, lm, f):
        """L: Index extended up, thumb extended to side, forming L shape."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index extended
        if ext[0]:
            score += 0.3
        # Middle, ring, pinky curled
        score += 0.2 * (sum(1 for e in ext[1:] if not e) / 3.0)
        # Thumb extended
        if f["thumb_extended"]:
            score += 0.25
        # Angle between thumb and index ~ 90 degrees
        thumb_vec = (lm[THUMB_TIP][0] - lm[THUMB_CMC][0], lm[THUMB_TIP][1] - lm[THUMB_CMC][1])
        index_vec = (lm[INDEX_TIP][0] - lm[INDEX_MCP][0], lm[INDEX_TIP][1] - lm[INDEX_MCP][1])
        dot = thumb_vec[0] * index_vec[0] + thumb_vec[1] * index_vec[1]
        mag_t = math.sqrt(thumb_vec[0] ** 2 + thumb_vec[1] ** 2) + 1e-6
        mag_i = math.sqrt(index_vec[0] ** 2 + index_vec[1] ** 2) + 1e-6
        cos_angle = dot / (mag_t * mag_i)
        # cos(90°) = 0, so closer to 0 is better
        if abs(cos_angle) < 0.5:
            score += 0.25
        return score

    def _rule_M(self, lm, f):
        """M: Three fingers (index, middle, ring) over thumb in fist."""
        ext = f["fingers_extended"]
        score = 0.0
        # All fingers curled
        score += 0.3 * self._all_curled(f, ext)
        # Thumb tucked under
        if not f["thumb_extended"]:
            score += 0.2
        # Thumb tip below index, middle, ring MCPs (tucked under three fingers)
        thumb_below = 0
        for mcp in [INDEX_MCP, MIDDLE_MCP, RING_MCP]:
            if lm[THUMB_TIP][1] > lm[mcp][1]:  # y increases downward
                thumb_below += 1
        score += 0.3 * (thumb_below / 3.0)
        # Distinguish from S/E: fingertips visible over thumb
        avg_tip_y = sum(lm[t][1] for t in FINGER_TIPS[:3]) / 3.0
        if avg_tip_y > lm[THUMB_TIP][1]:
            score += 0.2
        return score

    def _rule_N(self, lm, f):
        """N: Two fingers (index, middle) over thumb in fist."""
        ext = f["fingers_extended"]
        score = 0.0
        # All fingers curled
        score += 0.3 * self._all_curled(f, ext)
        # Thumb tucked
        if not f["thumb_extended"]:
            score += 0.2
        # Thumb between index and middle
        thumb_to_idx = _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / f["hand_scale"]
        thumb_to_mid = _dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / f["hand_scale"]
        if thumb_to_idx < 0.6 and thumb_to_mid < 0.6:
            score += 0.3
        # Ring and pinky more curled than index/middle
        if f["curl_ratios"][2] < f["curl_ratios"][0] and f["curl_ratios"][3] < f["curl_ratios"][0]:
            score += 0.2
        return score

    def _rule_O(self, lm, f):
        """O: All fingertips touch thumb tip, forming O shape."""
        score = 0.0
        # All fingertips close to thumb tip
        avg_thumb_dist = sum(f["thumb_to_tips"]) / 4.0
        if avg_thumb_dist < 0.6:
            score += 0.5
        elif avg_thumb_dist < 0.8:
            score += 0.3
        # Fingers partially curled (not fully extended)
        mid_curl = sum(1 for cr in f["curl_ratios"] if 0.8 < cr < 2.5) / 4.0
        score += 0.3 * mid_curl
        # Round shape: fingertips roughly equidistant from palm
        if len(f["tips_to_palm"]) == 4:
            spread = max(f["tips_to_palm"]) - min(f["tips_to_palm"])
            if spread < 0.4:
                score += 0.2
        return score

    def _rule_P(self, lm, f):
        """P: Like K but hand pointing down."""
        # Start with K score
        k_score = self._rule_K(lm, f)
        # Check if hand is pointing downward
        middle_tip_y = lm[MIDDLE_TIP][1]
        middle_mcp_y = lm[MIDDLE_MCP][1]
        # In image coords, y increases downward, so tip > mcp means pointing down
        if middle_tip_y > middle_mcp_y:
            return k_score * 0.9
        return k_score * 0.2

    def _rule_Q(self, lm, f):
        """Q: Like G but hand pointing down."""
        g_score = self._rule_G(lm, f)
        # Check if index is pointing downward
        if lm[INDEX_TIP][1] > lm[INDEX_MCP][1]:
            return g_score * 0.9
        return g_score * 0.2

    def _rule_R(self, lm, f):
        """R: Index and middle crossed/together and extended."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index and middle extended
        if ext[0] and ext[1]:
            score += 0.3
        # Ring and pinky curled
        score += 0.2 * (sum(1 for e in ext[2:] if not e) / 2.0)
        # Index and middle tips very close together (crossed)
        if f["tip_spreads"][0] < 0.25:
            score += 0.3
        # Thumb not extended
        if not f["thumb_extended"]:
            score += 0.2
        return score

    def _rule_S(self, lm, f):
        """S: Fist with thumb over fingers."""
        ext = f["fingers_extended"]
        score = 0.0
        # All fingers curled
        score += 0.3 * self._all_curled(f, ext)
        # Thumb over fingers (not extended outward, but crossing over)
        if not f["thumb_extended"]:
            score += 0.2
        # Thumb tip in front of curled fingers (near index/middle PIPs)
        thumb_to_idx_pip = _dist(lm[THUMB_TIP], lm[INDEX_PIP]) / f["hand_scale"]
        thumb_to_mid_pip = _dist(lm[THUMB_TIP], lm[MIDDLE_PIP]) / f["hand_scale"]
        if thumb_to_idx_pip < 0.6 or thumb_to_mid_pip < 0.6:
            score += 0.3
        # Distinguish from A: thumb more centered over fingers
        if f["thumb_to_index_mcp"] > 0.4:
            score += 0.2
        return score

    def _rule_T(self, lm, f):
        """T: Thumb between index and middle, tucked in fist."""
        ext = f["fingers_extended"]
        score = 0.0
        # All fingers curled
        score += 0.3 * self._all_curled(f, ext)
        # Thumb tucked between index and middle
        thumb_to_idx = _dist(lm[THUMB_TIP], lm[INDEX_PIP]) / f["hand_scale"]
        thumb_to_mid = _dist(lm[THUMB_TIP], lm[MIDDLE_PIP]) / f["hand_scale"]
        if thumb_to_idx < 0.5 and thumb_to_mid < 0.5:
            score += 0.4
        # Thumb not extended
        if not f["thumb_extended"]:
            score += 0.3
        return score

    def _rule_U(self, lm, f):
        """U: Index and middle extended together (close), rest curled."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index and middle extended
        if ext[0] and ext[1]:
            score += 0.3
        # Ring and pinky curled
        score += 0.2 * (sum(1 for e in ext[2:] if not e) / 2.0)
        # Index and middle together (small spread)
        if f["tip_spreads"][0] < 0.4:
            score += 0.3
        # Pointing upward (not sideways like H)
        idx_dir_y = abs(lm[INDEX_TIP][1] - lm[INDEX_MCP][1])
        idx_dir_x = abs(lm[INDEX_TIP][0] - lm[INDEX_MCP][0])
        if idx_dir_y > idx_dir_x:
            score += 0.2
        return score

    def _rule_V(self, lm, f):
        """V: Index and middle extended and spread (peace sign)."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index and middle extended
        if ext[0] and ext[1]:
            score += 0.3
        # Ring and pinky curled
        score += 0.2 * (sum(1 for e in ext[2:] if not e) / 2.0)
        # Index and middle spread apart
        if f["tip_spreads"][0] > 0.4:
            score += 0.3
        # Thumb not extended
        if not f["thumb_extended"]:
            score += 0.2
        return score

    def _rule_W(self, lm, f):
        """W: Index, middle, ring extended and spread, pinky curled."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index, middle, ring extended
        score += 0.3 * (sum(1 for e in ext[:3] if e) / 3.0)
        # Pinky curled
        if not ext[3]:
            score += 0.2
        # Fingers spread
        avg_spread = sum(f["tip_spreads"]) / 3.0
        if avg_spread > 0.35:
            score += 0.3
        # Thumb not extended
        if not f["thumb_extended"]:
            score += 0.2
        return score

    def _rule_X(self, lm, f):
        """X: Index bent at hook, rest curled."""
        ext = f["fingers_extended"]
        score = 0.0
        # Index not fully extended but DIP bent (hook shape)
        # Curl ratio between 1.0 and 1.8 (partially curled)
        if 0.8 < f["curl_ratios"][0] < 2.0:
            score += 0.3
        # Index tip closer to palm than a fully extended finger but not fully curled
        if 0.6 < f["tips_to_palm"][0] < 1.2:
            score += 0.2
        # Middle, ring, pinky curled
        score += 0.3 * (sum(1 for e in ext[1:] if not e) / 3.0)
        # Thumb not extended
        if not f["thumb_extended"]:
            score += 0.2
        return score

    def _rule_Y(self, lm, f):
        """Y: Thumb and pinky extended, rest curled (hang loose)."""
        ext = f["fingers_extended"]
        score = 0.0
        # Pinky extended
        if ext[3]:
            score += 0.3
        # Index, middle, ring curled
        score += 0.25 * (sum(1 for e in ext[:3] if not e) / 3.0)
        # Thumb extended
        if f["thumb_extended"]:
            score += 0.3
        # Thumb and pinky spread apart
        thumb_pinky_dist = _dist(lm[THUMB_TIP], lm[PINKY_TIP]) / f["hand_scale"]
        if thumb_pinky_dist > 1.2:
            score += 0.15
        return score

    def _rule_Z(self, lm, f):
        """Z: Index extended (static approximation — Z involves tracing motion)."""
        # Z is traced with index finger; statically looks like index pointing
        ext = f["fingers_extended"]
        score = 0.0
        if ext[0]:
            score += 0.2
        score += 0.1 * (sum(1 for e in ext[1:] if not e) / 3.0)
        # Very low confidence since we can't detect the tracing motion
        return score * 0.3


if __name__ == "__main__":
    import cv2
    from hand_tracker import HandTracker

    tracker = HandTracker(max_hands=1)
    classifier = ASLClassifier()
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: could not open webcam")
        raise SystemExit(1)

    print("ASL Fingerspelling Classifier — press 'q' to quit")

    # Smoothing: keep last N predictions
    history = []
    HISTORY_LEN = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, landmarks, normalized = tracker.process_frame(frame)

        if normalized:
            letter, confidence = classifier.classify(normalized[0])

            history.append((letter, confidence))
            if len(history) > HISTORY_LEN:
                history.pop(0)

            # Pick most common letter in history (weighted by confidence)
            from collections import Counter
            weighted = Counter()
            for l, c in history:
                weighted[l] += c
            display_letter = weighted.most_common(1)[0][0]
            display_conf = confidence

            color = (0, 255, 0) if display_conf > 0.6 else (0, 255, 255) if display_conf > 0.4 else (0, 0, 255)

            cv2.putText(
                annotated,
                f"{display_letter}  ({display_conf:.0%})",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                4,
            )
        else:
            cv2.putText(
                annotated,
                "No hand detected",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        cv2.imshow("ASL Fingerspelling", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
