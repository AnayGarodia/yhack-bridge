"""
SignAnimator — generates ASL sign illustrations with a fixed high-quality character.

Architecture:
  1. Beautiful hand-crafted base character SVG (always identical)
  2. AI provides ONLY hand configuration as JSON (position, finger states)
  3. Python renders the hands programmatically from the config
  4. Result: consistent character, correct ASL hand shapes
"""

import json
import logging
import math
import os
import re
import threading

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, "data", "sign_animations.json")

# ── AI prompt: only asks for hand configuration JSON, not full SVG ───────────
_SYSTEM_PROMPT = """\
You are an ASL (American Sign Language) expert. Given a sign name, output a JSON object describing the hand positions and finger states for that sign.

Output format (JSON only, no markdown):
{
  "right_hand": {
    "x": 300, "y": 150,
    "thumb": "extended",
    "index": "extended",
    "middle": "extended",
    "ring": "extended",
    "pinky": "extended",
    "palm_facing": "outward",
    "wrist_angle": 0
  },
  "left_hand": {
    "x": 120, "y": 340,
    "thumb": "curled",
    "index": "curled",
    "middle": "curled",
    "ring": "curled",
    "pinky": "curled",
    "palm_facing": "inward",
    "wrist_angle": 0
  },
  "description": "Brief description of the sign motion"
}

Coordinate system: 400x500 canvas. Character center at x=200. Head at y=160. Shoulders at y=240. Waist at y=370.
- Face level: y=120-180
- Chest level: y=230-300
- Waist level: y=330-370
- Side resting: x=100-120 (left) or x=280-300 (right), y=340-370

Finger states: "extended" (straight out), "curled" (closed into fist), "bent" (partially bent)
Palm facing: "outward" (toward viewer), "inward" (toward signer), "sideways", "down", "up"
Wrist angle: degrees of rotation, 0=neutral, positive=clockwise

Be anatomically accurate for each ASL sign. Output ONLY the JSON."""

# ── Hand-crafted base character SVG ──────────────────────────────────────────
_BASE_SVG_START = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
<rect width="400" height="500" fill="#141414" rx="16"/>
<defs>
  <linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#1a1a2e"/>
    <stop offset="100%" stop-color="#16213e"/>
  </linearGradient>
</defs>
<rect width="400" height="500" fill="url(#bg-grad)" rx="16"/>
"""

_BODY_SVG = """
<!-- Torso -->
<path d="M140 242 Q138 300 148 370 L252 370 Q262 300 260 242 Z" fill="#3C78B4" stroke="#2D2319" stroke-width="2" stroke-linejoin="round"/>
<path d="M148 242 L252 242 L250 254 L150 254 Z" fill="#2D5F8E"/>
<path d="M185 242 Q200 257 215 242" stroke="#2D5F8E" stroke-width="2" fill="none"/>
<!-- Neck -->
<rect x="188" y="218" width="24" height="24" rx="5" fill="#E1B48C"/>
<!-- Head -->
<ellipse cx="200" cy="170" rx="48" ry="52" fill="#E1B48C" stroke="#2D2319" stroke-width="2"/>
<!-- Hair (on top only — NOT covering face) -->
<path d="M152 155 Q152 128 170 115 Q185 108 200 106 Q215 108 230 115 Q248 128 248 155 L248 148 Q245 125 230 113 Q215 106 200 104 Q185 106 170 113 Q155 125 152 148 Z" fill="#4A3728"/>
<path d="M155 155 Q155 132 172 120 Q188 112 200 110 Q212 112 228 120 Q245 132 245 155" fill="#352D28" stroke="#2D2319" stroke-width="1"/>
<!-- Fringe/bangs -->
<path d="M160 150 Q170 140 180 148" fill="#352D28"/>
<path d="M175 146 Q185 136 195 145" fill="#3D3028"/>
<path d="M190 144 Q200 134 210 143" fill="#352D28"/>
<path d="M205 145 Q215 135 225 146" fill="#3D3028"/>
<path d="M220 148 Q230 138 240 150" fill="#352D28"/>
<!-- Ears -->
<ellipse cx="153" cy="172" rx="7" ry="11" fill="#D4A07A" stroke="#2D2319" stroke-width="1"/>
<ellipse cx="247" cy="172" rx="7" ry="11" fill="#D4A07A" stroke="#2D2319" stroke-width="1"/>
<path d="M150 168 Q148 172 150 176" stroke="#C9956B" stroke-width="1" fill="none"/>
<path d="M250 168 Q252 172 250 176" stroke="#C9956B" stroke-width="1" fill="none"/>
<!-- Eyes -->
<ellipse cx="183" cy="168" rx="9" ry="10" fill="white" stroke="#2D2319" stroke-width="1.5"/>
<ellipse cx="217" cy="168" rx="9" ry="10" fill="white" stroke="#2D2319" stroke-width="1.5"/>
<circle cx="185" cy="169" r="5" fill="#3E2F1C"/>
<circle cx="219" cy="169" r="5" fill="#3E2F1C"/>
<circle cx="185" cy="169" r="2.5" fill="#111"/>
<circle cx="219" cy="169" r="2.5" fill="#111"/>
<circle cx="183" cy="167" r="1.8" fill="white" opacity="0.9"/>
<circle cx="217" cy="167" r="1.8" fill="white" opacity="0.9"/>
<!-- Eyelashes -->
<path d="M175 161 Q178 159 180 161" stroke="#2D2319" stroke-width="1" fill="none"/>
<path d="M220 161 Q222 159 225 161" stroke="#2D2319" stroke-width="1" fill="none"/>
<!-- Eyebrows -->
<path d="M173 155 Q183 150 193 155" stroke="#3D3028" stroke-width="2.5" fill="none" stroke-linecap="round"/>
<path d="M207 155 Q217 150 227 155" stroke="#3D3028" stroke-width="2.5" fill="none" stroke-linecap="round"/>
<!-- Nose -->
<path d="M196 177 Q200 182 204 177" stroke="#C9956B" stroke-width="1.5" fill="none" stroke-linecap="round"/>
<circle cx="195" cy="178" r="1.2" fill="#D4A07A"/>
<circle cx="205" cy="178" r="1.2" fill="#D4A07A"/>
<!-- Mouth -->
<path d="M190 190 Q200 197 210 190" stroke="#B5716B" stroke-width="2.2" fill="none" stroke-linecap="round"/>
<path d="M193 190 Q200 194 207 190" fill="#C4706B" opacity="0.3"/>
<!-- Blush -->
<ellipse cx="168" cy="183" rx="10" ry="6" fill="#E8A090" opacity="0.2"/>
<ellipse cx="232" cy="183" rx="10" ry="6" fill="#E8A090" opacity="0.2"/>
"""

_SVG_END = "</svg>"


def _render_arm(shoulder_x, shoulder_y, hand_x, hand_y):
    """Render a smooth arm from shoulder to hand position."""
    # Compute elbow position (bent outward)
    mid_x = (shoulder_x + hand_x) / 2
    mid_y = (shoulder_y + hand_y) / 2
    dx = hand_x - shoulder_x
    dy = hand_y - shoulder_y
    # Elbow bends outward from the body center
    outward = 30 if hand_x < 200 else -30
    elbow_x = mid_x + outward
    elbow_y = mid_y - 15

    svg = ""
    # Arm shadow
    svg += f'<path d="M{shoulder_x} {shoulder_y} Q{elbow_x} {elbow_y} {hand_x} {hand_y}" '
    svg += f'stroke="#C9956B" stroke-width="22" stroke-linecap="round" fill="none"/>\n'
    # Arm main
    svg += f'<path d="M{shoulder_x} {shoulder_y} Q{elbow_x} {elbow_y} {hand_x} {hand_y}" '
    svg += f'stroke="#E1B48C" stroke-width="20" stroke-linecap="round" fill="none"/>\n'
    # Sleeve
    sx = shoulder_x + (5 if hand_x > 200 else -5)
    svg += f'<ellipse cx="{shoulder_x}" cy="{shoulder_y}" rx="15" ry="12" fill="#3C78B4" stroke="#2D2319" stroke-width="1.5"/>\n'
    return svg


def _render_hand(x, y, fingers, palm_facing="outward"):
    """Render a detailed hand at position (x, y) with finger states."""
    svg = ""
    face_out = palm_facing in ("outward", "forward")

    # Count extended fingers to decide fist vs open hand shape
    n_extended = sum(1 for f in ["index", "middle", "ring", "pinky"] if fingers.get(f) == "extended")
    is_fist = n_extended == 0

    if is_fist:
        # ── FIST: rounded compact shape with knuckle bumps ──
        fw, fh = 20, 24
        # Main fist body
        svg += f'<ellipse cx="{x}" cy="{y}" rx="{fw}" ry="{fh}" fill="#E1B48C" stroke="#C9956B" stroke-width="1.5"/>\n'
        # Knuckle bumps across top
        for i, kx in enumerate([-10, -3, 4, 11]):
            svg += f'<ellipse cx="{x + kx}" cy="{y - fh + 5}" rx="5" ry="4" fill="#DBA67C" stroke="#C9956B" stroke-width="0.6"/>\n'
        # Finger fold lines
        for ky in [-6, 0, 6]:
            svg += f'<path d="M{x - fw + 6} {y + ky} Q{x} {y + ky + 2} {x + fw - 6} {y + ky}" stroke="#C9956B" stroke-width="0.5" fill="none" opacity="0.4"/>\n'
        # Thumb wrapping over front
        ts = fingers.get("thumb", "curled")
        if ts == "extended":
            svg += f'<path d="M{x - fw} {y + 2} Q{x - fw - 10} {y - 12} {x - fw - 4} {y - 22}" stroke="#E1B48C" stroke-width="11" stroke-linecap="round" fill="none"/>\n'
            svg += f'<path d="M{x - fw} {y + 2} Q{x - fw - 10} {y - 12} {x - fw - 4} {y - 22}" stroke="#C9956B" stroke-width="1" fill="none"/>\n'
        else:
            svg += f'<path d="M{x - fw + 2} {y + 6} Q{x - fw - 4} {y} {x - fw + 2} {y - 8}" stroke="#DBA67C" stroke-width="10" stroke-linecap="round" fill="none"/>\n'
            svg += f'<ellipse cx="{x - fw + 1}" cy="{y - 8}" rx="4.5" ry="3.5" fill="#E8C8A8" stroke="#C9956B" stroke-width="0.5"/>\n'
        return svg

    # ── OPEN/PARTIAL HAND: palm + individual fingers ──
    palm_w = 20 if face_out else 15
    palm_h = 22 if face_out else 18

    # Palm
    svg += f'<rect x="{x - palm_w}" y="{y - palm_h + 4}" width="{palm_w * 2}" height="{palm_h * 2}" rx="6" fill="#E1B48C" stroke="#C9956B" stroke-width="1.2"/>\n'
    if face_out:
        svg += f'<path d="M{x - palm_w + 5} {y + 2} Q{x} {y + 10} {x + palm_w - 5} {y + 2}" stroke="#D4A07A" stroke-width="0.6" fill="none" opacity="0.4"/>\n'

    # Finger specs: (name, x_position_ratio, length, splay_degrees)
    finger_specs = [
        ("index",  -0.55, 30, -8),
        ("middle", -0.18, 33,  0),
        ("ring",    0.18, 30,  6),
        ("pinky",   0.55, 25, 13),
    ]

    for fname, xr, length, splay in finger_specs:
        state = fingers.get(fname, "curled")
        bx = x + xr * (palm_w - 3)
        by = y - palm_h + 6
        rad = math.radians(splay)

        if state == "extended":
            s1, s2, s3 = length * 0.38, length * 0.35, length * 0.27
            j1x = bx + s1 * math.sin(rad)
            j1y = by - s1 * math.cos(rad)
            j2x = j1x + s2 * math.sin(rad)
            j2y = j1y - s2 * math.cos(rad)
            tx = j2x + s3 * math.sin(rad)
            ty = j2y - s3 * math.cos(rad)
            # Finger segments (tapered)
            svg += f'<path d="M{bx:.1f} {by:.1f} L{j1x:.1f} {j1y:.1f} L{j2x:.1f} {j2y:.1f} L{tx:.1f} {ty:.1f}" stroke="#E1B48C" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" fill="none"/>\n'
            svg += f'<path d="M{bx:.1f} {by:.1f} L{j1x:.1f} {j1y:.1f} L{j2x:.1f} {j2y:.1f} L{tx:.1f} {ty:.1f}" stroke="#C9956B" stroke-width="1" fill="none" stroke-linejoin="round"/>\n'
            # Knuckle creases
            svg += f'<circle cx="{j1x:.1f}" cy="{j1y:.1f}" r="1" fill="#C9956B" opacity="0.4"/>\n'
            # Nail
            svg += f'<ellipse cx="{tx:.1f}" cy="{ty:.1f}" rx="3.5" ry="2.5" fill="#E8C8A8" stroke="#C9956B" stroke-width="0.5"/>\n'

        elif state == "bent":
            s1 = length * 0.4
            j1x = bx + s1 * math.sin(rad)
            j1y = by - s1 * math.cos(rad)
            cx2 = j1x + 5 * math.sin(rad + 0.5)
            cy2 = j1y
            tx = j1x + 3
            ty = j1y + 10
            svg += f'<path d="M{bx:.1f} {by:.1f} L{j1x:.1f} {j1y:.1f} Q{cx2:.1f} {cy2:.1f} {tx:.1f} {ty:.1f}" stroke="#E1B48C" stroke-width="8" stroke-linecap="round" fill="none"/>\n'
            svg += f'<path d="M{bx:.1f} {by:.1f} L{j1x:.1f} {j1y:.1f} Q{cx2:.1f} {cy2:.1f} {tx:.1f} {ty:.1f}" stroke="#C9956B" stroke-width="0.7" fill="none"/>\n'

        else:  # curled into palm
            svg += f'<ellipse cx="{bx}" cy="{by - 1}" rx="4.5" ry="3.5" fill="#DBA67C" stroke="#C9956B" stroke-width="0.5"/>\n'

    # Thumb
    ts = fingers.get("thumb", "curled")
    tbx = x - palm_w + 1
    tby = y

    if ts == "extended":
        svg += f'<path d="M{tbx} {tby} Q{tbx - 12} {tby - 10} {tbx - 14} {tby - 22}" stroke="#E1B48C" stroke-width="10" stroke-linecap="round" fill="none"/>\n'
        svg += f'<path d="M{tbx} {tby} Q{tbx - 12} {tby - 10} {tbx - 14} {tby - 22}" stroke="#C9956B" stroke-width="0.8" fill="none"/>\n'
        svg += f'<ellipse cx="{tbx - 14}" cy="{tby - 22}" rx="3.5" ry="2.5" fill="#E8C8A8" stroke="#C9956B" stroke-width="0.5"/>\n'
    elif ts == "bent":
        svg += f'<path d="M{tbx} {tby} Q{tbx - 8} {tby - 6} {tbx - 4} {tby + 5}" stroke="#E1B48C" stroke-width="9" stroke-linecap="round" fill="none"/>\n'
    else:
        svg += f'<path d="M{tbx} {tby} Q{tbx - 5} {tby - 3} {tbx + 1} {tby - 6}" stroke="#DBA67C" stroke-width="7" stroke-linecap="round" fill="none"/>\n'

    return svg


def _build_sign_svg(hand_config: dict, label: str = "") -> str:
    """Build a complete SVG from hand configuration JSON."""
    svg = _BASE_SVG_START + _BODY_SVG

    rh = hand_config.get("right_hand", {})
    lh = hand_config.get("left_hand", {})

    rx, ry = rh.get("x", 285), rh.get("y", 360)
    lx, ly = lh.get("x", 115), lh.get("y", 360)

    r_fingers = {k: rh.get(k, "curled") for k in ["thumb", "index", "middle", "ring", "pinky"]}
    l_fingers = {k: lh.get(k, "curled") for k in ["thumb", "index", "middle", "ring", "pinky"]}

    # Draw arms (back arm first based on position)
    if rx > lx:
        svg += _render_arm(130, 248, lx, ly)
        svg += _render_hand(lx, ly, l_fingers, lh.get("palm_facing", "inward"))
        svg += _render_arm(270, 248, rx, ry)
        svg += _render_hand(rx, ry, r_fingers, rh.get("palm_facing", "outward"))
    else:
        svg += _render_arm(270, 248, rx, ry)
        svg += _render_hand(rx, ry, r_fingers, rh.get("palm_facing", "outward"))
        svg += _render_arm(130, 248, lx, ly)
        svg += _render_hand(lx, ly, l_fingers, lh.get("palm_facing", "inward"))

    # Sign label + description at bottom
    if label:
        desc = hand_config.get("description", "")
        svg += f'<rect x="0" y="448" width="400" height="52" fill="rgba(0,0,0,0.6)" rx="0"/>\n'
        svg += f'<text x="200" y="470" text-anchor="middle" fill="white" font-family="Arial,sans-serif" font-size="18" font-weight="bold">{label}</text>\n'
        if desc:
            # Truncate long descriptions
            short = desc[:50] + ("..." if len(desc) > 50 else "")
            svg += f'<text x="200" y="490" text-anchor="middle" fill="#999" font-family="Arial,sans-serif" font-size="11">{short}</text>\n'

    svg += _SVG_END
    return svg


# ── Default idle pose ────────────────────────────────────────────────────────
_IDLE_CONFIG = {
    "right_hand": {"x": 285, "y": 365, "thumb": "curled", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
    "left_hand": {"x": 115, "y": 365, "thumb": "curled", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
}
_IDLE_SVG = _build_sign_svg(_IDLE_CONFIG)

# Open hand (all fingers extended, palm out)
_OPEN_HAND = {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended", "palm_facing": "outward"}
# Flat hand (fingers together, palm out)
_FLAT_HAND = {"thumb": "bent", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended", "palm_facing": "outward"}
# Fist (all curled)
_FIST = {"thumb": "curled", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}
# S-fist (thumb over fingers)
_S_FIST = {"thumb": "bent", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}
# Index point
_POINT = {"thumb": "curled", "index": "extended", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "outward"}
# Resting hand at side (relaxed, not tight fist)
_REST = {"x": 280, "y": 355, "thumb": "bent", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}
_REST_L = {"x": 120, "y": 355, "thumb": "bent", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}

# ── Additional hand shapes ────────────────────────────────────────────────────
_BENT_HAND = {"thumb": "bent", "index": "bent", "middle": "bent", "ring": "bent", "pinky": "bent", "palm_facing": "inward"}
_C_HAND = {"thumb": "extended", "index": "bent", "middle": "bent", "ring": "bent", "pinky": "bent", "palm_facing": "sideways"}
_W_HAND = {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "curled", "palm_facing": "outward"}
_H_HAND = {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "sideways"}
_ILY = {"thumb": "extended", "index": "extended", "middle": "curled", "ring": "curled", "pinky": "extended", "palm_facing": "outward"}
_CLAW = {"thumb": "extended", "index": "bent", "middle": "bent", "ring": "bent", "pinky": "bent", "palm_facing": "outward"}

# Helper: shorthand for frame dicts
def _f(rx, ry, rh, lx=None, ly=None, lh=None, desc=""):
    r = {"x": rx, "y": ry, **rh}
    l = {"x": lx, "y": ly, **(lh or {})} if lx else _REST_L
    return {"right_hand": r, "left_hand": l, "description": desc}

def hand_config_to_landmarks(hand: dict) -> list[list[float]]:
    """Convert a hand config dict (finger states + x,y) to 21 MediaPipe-style 3D landmarks.

    Returns list of 21 [x, y, z] points normalized ~0-1 range.
    """
    cx = hand.get("x", 200) / 400.0  # normalize to 0-1
    cy = hand.get("y", 300) / 500.0

    # Wrist at hand center
    landmarks = [[cx, cy, 0.5]]  # index 0: wrist

    finger_defs = [
        ("thumb",  [-0.08, -0.01], 0.06, -50),
        ("index",  [-0.03, -0.06], 0.08, -10),
        ("middle", [-0.01, -0.06], 0.09,   0),
        ("ring",   [ 0.02, -0.06], 0.08,   8),
        ("pinky",  [ 0.04, -0.06], 0.07,  15),
    ]

    for fname, base_off, length, angle_deg in finger_defs:
        state = hand.get(fname, "curled")
        bx = cx + base_off[0]
        by = cy + base_off[1]
        rad = math.radians(angle_deg)

        if state == "extended":
            segs = [0.35, 0.3, 0.2, 0.15]
        elif state == "bent":
            segs = [0.35, 0.2, 0.05, -0.05]
        else:  # curled
            segs = [0.15, 0.02, -0.05, -0.08]

        px, py = bx, by
        for i, seg_frac in enumerate(segs):
            seg_len = length * seg_frac if state != "curled" else length * abs(seg_frac)
            dx = seg_len * math.sin(rad)
            dy = -seg_len * math.cos(rad)
            if state == "curled" and i >= 2:
                dy = abs(dy) * 0.5  # curl back toward palm
            px = px + dx
            py = py + dy
            landmarks.append([round(px, 4), round(py, 4), round(0.5 + i * 0.02, 4)])

    return landmarks


def _interpolate_frames(frames: list[dict]) -> list[dict]:
    """Double the frame count by inserting midpoints between each pair of keyframes."""
    if len(frames) <= 1:
        return frames
    result = []
    for i in range(len(frames) - 1):
        result.append(frames[i])
        # Create midpoint frame
        a = frames[i]
        b = frames[i + 1]
        mid = {}
        for hand_key in ("right_hand", "left_hand"):
            ah = a.get(hand_key, {})
            bh = b.get(hand_key, {})
            mh = {}
            for k in ah:
                if k in ("x", "y") and k in bh:
                    mh[k] = int((ah[k] + bh[k]) / 2)
                else:
                    mh[k] = ah[k]
            mid[hand_key] = mh
        mid["description"] = ""
        result.append(mid)
    result.append(frames[-1])
    return result

# ── Hardcoded ASL sign definitions — 4-6 frames per sign for smooth motion ──
_HARDCODED_SIGNS = {
    "HELLO": [
        _f(275, 340, _S_FIST, desc="Fist at rest"),
        _f(275, 300, _OPEN_HAND, desc="Hand opens rising"),
        _f(278, 250, _OPEN_HAND, desc="Rising quickly"),
        _f(282, 200, _OPEN_HAND, desc="Approaching forehead"),
        _f(288, 160, _OPEN_HAND, desc="Near forehead"),
        _f(295, 145, _OPEN_HAND, desc="At forehead, palm out"),
        _f(310, 140, _OPEN_HAND, desc="Begin wave"),
        _f(325, 142, _OPEN_HAND, desc="Wave outward"),
        _f(340, 148, _OPEN_HAND, desc="Wave wide"),
        _f(348, 155, _OPEN_HAND, desc="Wave peak"),
        _f(338, 150, _OPEN_HAND, desc="Wave back"),
        _f(325, 145, _OPEN_HAND, desc="Second wave"),
        _f(340, 152, _OPEN_HAND, desc="Wave out again"),
        _f(332, 158, _OPEN_HAND, desc="Settle"),
    ],
    "THANK-YOU": [
        _f(235, 320, _FLAT_HAND, desc="Hand at rest"),
        _f(232, 290, _FLAT_HAND, desc="Hand rises"),
        _f(228, 250, _FLAT_HAND, desc="Approaching chin"),
        _f(224, 215, _FLAT_HAND, desc="Near chin"),
        _f(220, 192, _FLAT_HAND, desc="Touching chin"),
        _f(225, 195, _FLAT_HAND, desc="Press chin"),
        _f(235, 208, _FLAT_HAND, desc="Begin forward motion"),
        _f(250, 225, _FLAT_HAND, desc="Moving outward"),
        _f(270, 245, _FLAT_HAND, desc="Extending forward"),
        _f(290, 262, _FLAT_HAND, desc="Almost full extension"),
        _f(305, 275, _FLAT_HAND, desc="Extended — thank you"),
        _f(310, 280, _FLAT_HAND, desc="Hold"),
    ],
    "YES": [
        _f(278, 280, _S_FIST, desc="Fist rises from rest"),
        _f(278, 250, _S_FIST, desc="Rising"),
        _f(278, 210, _S_FIST, desc="Fist up"),
        _f(278, 195, _S_FIST, desc="Top position"),
        _f(278, 220, _S_FIST, desc="Nod down"),
        _f(278, 245, _S_FIST, desc="Down"),
        _f(278, 210, _S_FIST, desc="Back up"),
        _f(278, 195, _S_FIST, desc="Top again"),
        _f(278, 225, _S_FIST, desc="Nod down again"),
        _f(278, 248, _S_FIST, desc="Bottom"),
        _f(278, 215, _S_FIST, desc="Return up"),
    ],
    "NO": [
        _f(278, 280, _W_HAND, desc="Hand rises"),
        _f(278, 240, _W_HAND, desc="Fingers open"),
        _f(278, 210, _W_HAND, desc="Index+middle extended"),
        _f(278, 200, _W_HAND, desc="Full extension"),
        _f(278, 200, _H_HAND, desc="Begin snap to thumb"),
        _f(278, 198, _S_FIST, desc="Fingers snap shut"),
        _f(278, 202, _W_HAND, desc="Open again"),
        _f(278, 200, _S_FIST, desc="Snap shut again — NO"),
        _f(278, 205, _S_FIST, desc="Hold"),
    ],
    "PLEASE": [
        _f(222, 320, _FLAT_HAND, desc="Hand at lower chest"),
        _f(215, 300, _FLAT_HAND, desc="Begin circle upward"),
        _f(208, 278, _FLAT_HAND, desc="Circle up-left"),
        _f(215, 262, _FLAT_HAND, desc="Circle top-left"),
        _f(228, 258, _FLAT_HAND, desc="Circle top-right"),
        _f(242, 270, _FLAT_HAND, desc="Circle right"),
        _f(248, 290, _FLAT_HAND, desc="Circle bottom-right"),
        _f(238, 310, _FLAT_HAND, desc="Circle bottom"),
        _f(222, 318, _FLAT_HAND, desc="Complete circle — please"),
    ],
    "HELP": [
        _f(200, 350, _S_FIST, 200, 320, _FLAT_HAND, desc="Fist on open palm"),
        _f(200, 335, _S_FIST, 200, 305, _FLAT_HAND, desc="Begin rising"),
        _f(200, 315, _S_FIST, 200, 285, _FLAT_HAND, desc="Rising together"),
        _f(200, 295, _S_FIST, 200, 268, _FLAT_HAND, desc="Higher"),
        _f(200, 275, _S_FIST, 200, 250, _FLAT_HAND, desc="Near top"),
        _f(200, 260, _S_FIST, 200, 238, _FLAT_HAND, desc="Reach top — help"),
        _f(200, 265, _S_FIST, 200, 242, _FLAT_HAND, desc="Hold"),
    ],
    "WATER": [
        _f(232, 280, _W_HAND, desc="W-hand rises"),
        _f(230, 240, _W_HAND, desc="Approaching chin"),
        _f(228, 205, _W_HAND, desc="Near chin"),
        _f(225, 192, _W_HAND, desc="W taps chin"),
        _f(228, 200, _W_HAND, desc="Lift"),
        _f(225, 190, _W_HAND, desc="Tap again"),
        _f(228, 198, _W_HAND, desc="Lift"),
        _f(225, 188, _W_HAND, desc="Third tap — water"),
        _f(230, 200, _W_HAND, desc="Release"),
    ],
    "NAME": [
        _f(260, 270, _H_HAND, 180, 290, _H_HAND, desc="H-hands approach each other"),
        _f(250, 250, _H_HAND, 190, 260, _H_HAND, desc="H-fingers meet"),
        _f(255, 240, _H_HAND, 185, 250, _H_HAND, desc="Tap together"),
        _f(250, 250, _H_HAND, 190, 260, _H_HAND, desc="Tap again — name"),
    ],
    "MY": [
        _f(240, 310, _FLAT_HAND, desc="Hand approaches chest"),
        _f(220, 280, _FLAT_HAND, desc="Flat hand touches chest"),
        _f(215, 270, _FLAT_HAND, desc="Press on chest — my"),
        _f(220, 280, _FLAT_HAND, desc="Hold"),
    ],
    "YOU": [
        _f(250, 280, _POINT, desc="Hand rises, index extends"),
        _f(280, 250, _POINT, desc="Point forward"),
        _f(310, 235, _POINT, desc="Point fully extended — you"),
        _f(300, 240, _POINT, desc="Hold"),
    ],
    "HAPPY": [
        _f(230, 310, _FLAT_HAND, 170, 310, _FLAT_HAND, desc="Both hands at belly"),
        _f(230, 295, _FLAT_HAND, 170, 295, _FLAT_HAND, desc="Rising"),
        _f(232, 275, _FLAT_HAND, 168, 275, _FLAT_HAND, desc="Brush up on chest"),
        _f(234, 258, _FLAT_HAND, 166, 258, _FLAT_HAND, desc="Sweeping higher"),
        _f(232, 242, _FLAT_HAND, 168, 242, _FLAT_HAND, desc="Near top"),
        _f(230, 235, _FLAT_HAND, 170, 235, _FLAT_HAND, desc="Top — happy"),
        _f(232, 270, _FLAT_HAND, 168, 270, _FLAT_HAND, desc="Drop back down"),
        _f(234, 255, _FLAT_HAND, 166, 255, _FLAT_HAND, desc="Second brush up"),
        _f(230, 238, _FLAT_HAND, 170, 238, _FLAT_HAND, desc="Top again — happy!"),
    ],
    "SORRY": [
        _f(220, 310, _S_FIST, desc="Fist at chest"),
        _f(225, 280, _S_FIST, desc="Circle up-right"),
        _f(240, 270, _S_FIST, desc="Circle continues right"),
        _f(235, 295, _S_FIST, desc="Circle down"),
        _f(220, 310, _S_FIST, desc="Complete circle — sorry"),
    ],
    "GOOD": [
        _f(240, 250, _FLAT_HAND, desc="Hand rises to chin"),
        _f(230, 195, _FLAT_HAND, desc="Flat hand at chin"),
        _f(240, 240, _FLAT_HAND, 180, 290, _FLAT_HAND, desc="Hand drops toward palm"),
        _f(250, 280, _FLAT_HAND, 180, 300, _FLAT_HAND, desc="Hand lands on palm — good"),
    ],
    "BAD": [
        _f(230, 250, _FLAT_HAND, desc="Hand rises to chin"),
        _f(225, 195, _FLAT_HAND, desc="Flat hand at chin, palm in"),
        _f(235, 230, _FLAT_HAND, desc="Hand turns and drops"),
        _f(250, 290, _FLAT_HAND, desc="Hand flips down — bad"),
    ],
    "I": [
        _f(240, 300, _POINT, desc="Hand at side"),
        _f(225, 280, _POINT, desc="Point toward self"),
        _f(215, 270, _POINT, desc="Point touches chest — I"),
        _f(220, 275, _POINT, desc="Hold"),
    ],
    "WANT": [
        _f(280, 290, _OPEN_HAND, 140, 290, _OPEN_HAND, desc="Both hands open in front"),
        _f(270, 270, _OPEN_HAND, 150, 270, _OPEN_HAND, desc="Hands begin pulling in"),
        _f(255, 275, _CLAW, 160, 275, _CLAW, desc="Fingers curl as hands pull toward body"),
        _f(240, 285, _BENT_HAND, 170, 285, _BENT_HAND, desc="Hands clawed inward — want"),
    ],
    "EAT": [
        _f(250, 260, _S_FIST, desc="Hand rises toward mouth"),
        _f(235, 200, _S_FIST, desc="Bunched fingers at mouth"),
        _f(240, 210, _S_FIST, desc="Pull away slightly"),
        _f(235, 195, _S_FIST, desc="Tap mouth again — eat"),
        _f(240, 210, _S_FIST, desc="Pull away"),
    ],
    "DRINK": [
        _f(245, 260, _C_HAND, desc="C-hand rises toward mouth"),
        _f(235, 210, _C_HAND, desc="C-shape at mouth"),
        _f(230, 195, _C_HAND, desc="Tip hand up — drinking"),
        _f(235, 210, _C_HAND, desc="Return — drink"),
    ],
    "FRIEND": [
        _f(250, 290, _POINT, 170, 290, _POINT, desc="Both index fingers approach"),
        _f(240, 265, _POINT, 180, 270, _POINT, desc="Index fingers hook together"),
        _f(235, 260, _POINT, 185, 265, _POINT, desc="Locked — friend"),
        _f(240, 265, _POINT, 180, 270, _POINT, desc="Hold"),
    ],
    "FAMILY": [
        _f(250, 260, _OPEN_HAND, 150, 260, _OPEN_HAND, desc="Both F-hands in front"),
        _f(260, 250, _OPEN_HAND, 140, 250, _OPEN_HAND, desc="Hands circle outward"),
        _f(255, 275, _OPEN_HAND, 145, 275, _OPEN_HAND, desc="Circle down"),
        _f(245, 270, _OPEN_HAND, 155, 270, _OPEN_HAND, desc="Circle back — family"),
    ],
    "HOME": [
        _f(235, 230, _S_FIST, desc="Hand approaches chin"),
        _f(225, 195, _S_FIST, desc="Bunched fingers at chin"),
        _f(235, 185, _S_FIST, desc="Move toward cheek"),
        _f(245, 170, _S_FIST, desc="Touch cheek — home"),
    ],
    "SCHOOL": [
        _f(260, 290, _FLAT_HAND, 160, 300, _FLAT_HAND, desc="Hands apart"),
        _f(240, 270, _FLAT_HAND, 180, 280, _FLAT_HAND, desc="Hands approach — clap"),
        _f(260, 285, _FLAT_HAND, 160, 295, _FLAT_HAND, desc="Hands separate"),
        _f(240, 265, _FLAT_HAND, 180, 275, _FLAT_HAND, desc="Clap again — school"),
    ],
    "WORK": [
        _f(255, 300, _S_FIST, 170, 310, _S_FIST, desc="Fists approach"),
        _f(245, 280, _S_FIST, 175, 300, _S_FIST, desc="Right fist taps left"),
        _f(255, 290, _S_FIST, 170, 305, _S_FIST, desc="Lift right fist"),
        _f(245, 275, _S_FIST, 175, 300, _S_FIST, desc="Tap again — work"),
    ],
    "LOVE": [
        _f(240, 340, _S_FIST, desc="Fist at rest"),
        _f(238, 310, _S_FIST, desc="Rising"),
        _f(235, 285, _ILY, desc="Open to ILY shape"),
        _f(230, 268, _ILY, desc="ILY at chest level"),
        _f(228, 255, _ILY, desc="Rise higher"),
        _f(225, 248, _ILY, desc="Hold up — I Love You"),
        _f(228, 252, _ILY, desc="Gentle pulse"),
        _f(225, 245, _ILY, desc="Hold"),
    ],
    "FOOD": [
        _f(245, 250, _S_FIST, desc="Hand rises"),
        _f(235, 200, _S_FIST, desc="Bunched fingers at mouth"),
        _f(240, 215, _S_FIST, desc="Pull away"),
        _f(235, 195, _S_FIST, desc="Tap again — food"),
    ],
    "UNDERSTAND": [
        _f(260, 210, _S_FIST, desc="Fist near forehead"),
        _f(255, 175, _S_FIST, desc="Fist at forehead"),
        _f(260, 165, _S_FIST, desc="Begin flicking up"),
        _f(260, 155, _POINT, desc="Index flicks up — understand!"),
    ],
    "MORE": [
        _f(255, 290, _S_FIST, 165, 290, _S_FIST, desc="Bunched hands apart"),
        _f(240, 275, _S_FIST, 180, 275, _S_FIST, desc="Fingertips approach"),
        _f(250, 285, _S_FIST, 170, 285, _S_FIST, desc="Separate slightly"),
        _f(240, 272, _S_FIST, 180, 272, _S_FIST, desc="Tap again — more"),
    ],
    "STOP": [
        _f(260, 300, _FLAT_HAND, 170, 310, _FLAT_HAND, desc="Flat hand rises"),
        _f(250, 270, _FLAT_HAND, 180, 290, _FLAT_HAND, desc="Right hand chops down"),
        _f(230, 290, _FLAT_HAND, 180, 290, _FLAT_HAND, desc="Hand strikes palm — stop"),
        _f(235, 285, _FLAT_HAND, 180, 290, _FLAT_HAND, desc="Hold"),
    ],
    # ── New signs ──
    "WHAT": [
        _f(280, 280, _OPEN_HAND, 140, 290, _FLAT_HAND, desc="Open hand in front"),
        _f(250, 275, _OPEN_HAND, 160, 285, _FLAT_HAND, desc="Brush down across palm"),
        _f(220, 285, _OPEN_HAND, 175, 285, _FLAT_HAND, desc="Reach other side"),
        _f(240, 280, _OPEN_HAND, 165, 285, _FLAT_HAND, desc="Complete — what?"),
    ],
    "WHERE": [
        _f(270, 250, _POINT, desc="Index finger up"),
        _f(280, 230, _POINT, desc="Shake finger right"),
        _f(260, 230, _POINT, desc="Shake finger left"),
        _f(275, 230, _POINT, desc="Shake right again — where?"),
    ],
    "WHY": [
        _f(260, 200, _OPEN_HAND, desc="Hand at forehead"),
        _f(255, 175, _OPEN_HAND, desc="Touch forehead"),
        _f(270, 210, {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"}, desc="Pull away into Y-shape"),
        _f(285, 230, {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"}, desc="Complete — why?"),
    ],
    "HOW": [
        _f(250, 280, _S_FIST, 170, 280, _S_FIST, desc="Both fists together, knuckles up"),
        _f(260, 260, _OPEN_HAND, 160, 260, _OPEN_HAND, desc="Roll open outward"),
        _f(270, 250, _OPEN_HAND, 150, 250, _OPEN_HAND, desc="Fingers spread — how?"),
        _f(265, 255, _OPEN_HAND, 155, 255, _OPEN_HAND, desc="Hold"),
    ],
    "WHO": [
        _f(230, 210, _POINT, desc="Index at lips"),
        _f(225, 195, _POINT, desc="Touch lips"),
        _f(230, 200, _POINT, desc="Circle at lips"),
        _f(228, 195, _POINT, desc="Complete circle — who?"),
    ],
    "CAN": [
        _f(260, 290, _S_FIST, 160, 290, _S_FIST, desc="Both S-fists up"),
        _f(255, 310, _S_FIST, 165, 310, _S_FIST, desc="Both push down together"),
        _f(258, 305, _S_FIST, 162, 305, _S_FIST, desc="Settle — can"),
        _f(255, 310, _S_FIST, 165, 310, _S_FIST, desc="Hold"),
    ],
    "GO": [
        _f(240, 280, _POINT, 170, 280, _POINT, desc="Both index fingers point"),
        _f(260, 260, _POINT, 180, 260, _POINT, desc="Both arc forward"),
        _f(290, 250, _POINT, 200, 250, _POINT, desc="Point outward — go"),
        _f(300, 255, _POINT, 205, 255, _POINT, desc="Complete"),
    ],
    "COME": [
        _f(310, 260, _POINT, desc="Point outward"),
        _f(290, 265, _POINT, desc="Beckon inward"),
        _f(260, 275, _BENT_HAND, desc="Curl fingers toward self"),
        _f(240, 280, _BENT_HAND, desc="Pull in — come"),
    ],
    "LIKE": [
        _f(225, 280, {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}, desc="Hand at chest"),
        _f(240, 270, {"thumb": "extended", "index": "bent", "middle": "bent", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}, desc="Pull away, fingers close"),
        _f(260, 275, {"thumb": "extended", "index": "bent", "middle": "bent", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}, desc="Pinch — like"),
        _f(265, 280, _S_FIST, desc="Complete"),
    ],
    "KNOW": [
        _f(260, 210, _FLAT_HAND, desc="Hand approaches forehead"),
        _f(255, 175, _FLAT_HAND, desc="Flat hand taps forehead"),
        _f(260, 185, _FLAT_HAND, desc="Tap again — know"),
        _f(258, 175, _FLAT_HAND, desc="Hold"),
    ],
    "SEE": [
        _f(240, 200, _POINT, desc="Index finger near eyes"),
        _f(250, 190, _POINT, desc="Point from eyes"),
        _f(280, 220, _POINT, desc="Point outward — see"),
        _f(290, 230, _POINT, desc="Complete"),
    ],
    "HEAR": [
        _f(250, 210, _POINT, desc="Index near ear"),
        _f(255, 185, _POINT, desc="Point at ear — hear"),
        _f(260, 190, _POINT, desc="Tap ear"),
        _f(255, 185, _POINT, desc="Hold"),
    ],
    "THINK": [
        _f(250, 210, _POINT, desc="Hand approaches forehead"),
        _f(240, 175, _POINT, desc="Index finger touches forehead — think"),
        _f(245, 180, _POINT, desc="Tap forehead"),
        _f(240, 175, _POINT, desc="Hold"),
    ],
    "LOOK": [
        _f(245, 200, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"}, desc="V-hand at eyes"),
        _f(250, 190, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"}, desc="Turn outward from eyes"),
        _f(280, 220, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"}, desc="Point V outward — look"),
        _f(290, 230, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"}, desc="Complete"),
    ],
    "GIVE": [
        _f(230, 280, _FLAT_HAND, desc="Hand at body"),
        _f(250, 265, _FLAT_HAND, desc="Push outward"),
        _f(280, 255, _OPEN_HAND, desc="Hand opens forward"),
        _f(300, 260, _OPEN_HAND, desc="Release — give"),
    ],
    "FINISH": [
        _f(250, 260, _OPEN_HAND, 170, 260, _OPEN_HAND, desc="Both open hands up"),
        _f(260, 270, _OPEN_HAND, 160, 270, _OPEN_HAND, desc="Shake outward"),
        _f(275, 265, _OPEN_HAND, 145, 265, _OPEN_HAND, desc="Fingers splay out — finish"),
        _f(280, 270, _OPEN_HAND, 140, 270, _OPEN_HAND, desc="Complete"),
    ],
    "AGAIN": [
        _f(260, 300, _BENT_HAND, 170, 290, _FLAT_HAND, desc="Bent hand above open palm"),
        _f(245, 280, _BENT_HAND, 175, 290, _FLAT_HAND, desc="Arc up"),
        _f(240, 290, _BENT_HAND, 175, 290, _FLAT_HAND, desc="Touch palm — again"),
        _f(255, 275, _BENT_HAND, 175, 290, _FLAT_HAND, desc="Arc up for repeat"),
    ],
    "DIFFERENT": [
        _f(240, 270, _POINT, 180, 270, _POINT, desc="Both index fingers crossed"),
        _f(270, 260, _POINT, 150, 260, _POINT, desc="Pull apart"),
        _f(300, 255, _POINT, 120, 255, _POINT, desc="Spread wide — different"),
        _f(295, 260, _POINT, 125, 260, _POINT, desc="Hold"),
    ],
    # ── Batch: 80+ more common signs ──
    "NEED": [
        _f(270, 280, _BENT_HAND, desc="Hand bends down"),
        _f(270, 300, _BENT_HAND, desc="Nod wrist down"),
        _f(270, 280, _BENT_HAND, desc="Nod up — need"),
        _f(270, 295, _BENT_HAND, desc="Nod down again"),
    ],
    "HAVE": [
        _f(250, 290, _BENT_HAND, desc="Bent hand approaches chest"),
        _f(230, 275, _BENT_HAND, desc="Touch chest"),
        _f(225, 270, _BENT_HAND, desc="Press — have"),
        _f(230, 275, _BENT_HAND, desc="Hold"),
    ],
    "NOT": [
        _f(240, 210, _OPEN_HAND, desc="Hand at chin"),
        _f(230, 195, _OPEN_HAND, desc="Thumb under chin"),
        _f(260, 230, _OPEN_HAND, desc="Flick forward — not"),
        _f(275, 245, _OPEN_HAND, desc="Complete"),
    ],
    "MAKE": [
        _f(250, 280, _S_FIST, 170, 290, _S_FIST, desc="Fists stack"),
        _f(250, 275, _S_FIST, 170, 285, _S_FIST, desc="Twist right on left"),
        _f(255, 270, _S_FIST, 170, 285, _S_FIST, desc="Twist — make"),
        _f(250, 275, _S_FIST, 170, 285, _S_FIST, desc="Hold"),
    ],
    "SAY": [
        _f(230, 210, _POINT, desc="Index at chin"),
        _f(225, 195, _POINT, desc="Touch chin"),
        _f(250, 220, _POINT, desc="Arc forward from chin"),
        _f(275, 240, _POINT, desc="Point out — say"),
    ],
    "TIME": [
        _f(260, 280, _POINT, 170, 290, _S_FIST, desc="Point at wrist"),
        _f(255, 275, _POINT, 170, 290, _S_FIST, desc="Tap wrist"),
        _f(260, 280, _POINT, 170, 290, _S_FIST, desc="Tap again — time"),
        _f(258, 278, _POINT, 170, 290, _S_FIST, desc="Hold"),
    ],
    "PEOPLE": [
        _f(260, 270, _POINT, 160, 270, _POINT, desc="Both P-hands"),
        _f(270, 265, _POINT, 150, 265, _POINT, desc="Circle alternating"),
        _f(260, 275, _POINT, 160, 275, _POINT, desc="Circle back"),
        _f(268, 268, _POINT, 152, 268, _POINT, desc="Complete — people"),
    ],
    "MOTHER": [
        _f(230, 210, _OPEN_HAND, desc="Open hand approaches chin"),
        _f(220, 195, _OPEN_HAND, desc="Thumb taps chin"),
        _f(225, 200, _OPEN_HAND, desc="Tap again — mother"),
        _f(220, 195, _OPEN_HAND, desc="Hold"),
    ],
    "FATHER": [
        _f(230, 180, _OPEN_HAND, desc="Open hand approaches forehead"),
        _f(220, 165, _OPEN_HAND, desc="Thumb taps forehead"),
        _f(225, 170, _OPEN_HAND, desc="Tap again — father"),
        _f(220, 165, _OPEN_HAND, desc="Hold"),
    ],
    "MAN": [
        _f(230, 185, _OPEN_HAND, desc="Hand at forehead"),
        _f(225, 170, _OPEN_HAND, desc="Thumb on forehead"),
        _f(230, 250, _FLAT_HAND, desc="Hand drops to chest"),
        _f(228, 260, _FLAT_HAND, desc="Touch chest — man"),
    ],
    "WOMAN": [
        _f(230, 200, _OPEN_HAND, desc="Hand at chin"),
        _f(225, 190, _OPEN_HAND, desc="Thumb on chin"),
        _f(230, 260, _FLAT_HAND, desc="Hand drops to chest"),
        _f(228, 270, _FLAT_HAND, desc="Touch chest — woman"),
    ],
    "CHILD": [
        _f(280, 310, _FLAT_HAND, desc="Flat hand low"),
        _f(280, 290, _FLAT_HAND, desc="Pat down — child height"),
        _f(280, 305, _FLAT_HAND, desc="Pat again"),
        _f(280, 290, _FLAT_HAND, desc="Pat — child"),
    ],
    "BOY": [
        _f(230, 180, _S_FIST, desc="Fist at forehead"),
        _f(225, 170, _S_FIST, desc="Open-close at forehead"),
        _f(230, 175, _FLAT_HAND, desc="Open"),
        _f(225, 170, _S_FIST, desc="Close — boy"),
    ],
    "GIRL": [
        _f(225, 200, _POINT, desc="Thumb on cheek"),
        _f(220, 195, _POINT, desc="Trace down jaw"),
        _f(225, 205, _POINT, desc="To chin — girl"),
        _f(223, 200, _POINT, desc="Hold"),
    ],
    "BIG": [
        _f(230, 270, _OPEN_HAND, 190, 270, _OPEN_HAND, desc="Both hands together"),
        _f(270, 260, _OPEN_HAND, 150, 260, _OPEN_HAND, desc="Spread apart"),
        _f(310, 255, _OPEN_HAND, 110, 255, _OPEN_HAND, desc="Wide — big"),
        _f(305, 258, _OPEN_HAND, 115, 258, _OPEN_HAND, desc="Hold"),
    ],
    "SMALL": [
        _f(280, 260, _FLAT_HAND, 140, 260, _FLAT_HAND, desc="Both hands apart"),
        _f(260, 265, _FLAT_HAND, 160, 265, _FLAT_HAND, desc="Move together"),
        _f(240, 268, _FLAT_HAND, 180, 268, _FLAT_HAND, desc="Close — small"),
        _f(245, 267, _FLAT_HAND, 175, 267, _FLAT_HAND, desc="Hold"),
    ],
    "HOT": [
        _f(230, 200, _CLAW, desc="Claw at mouth"),
        _f(225, 195, _CLAW, desc="Touch mouth"),
        _f(260, 230, _CLAW, desc="Pull away and turn"),
        _f(280, 250, _OPEN_HAND, desc="Open hand outward — hot"),
    ],
    "COLD": [
        _f(250, 270, _S_FIST, 170, 270, _S_FIST, desc="Both fists up"),
        _f(245, 275, _S_FIST, 175, 275, _S_FIST, desc="Shake/shiver left"),
        _f(255, 275, _S_FIST, 165, 275, _S_FIST, desc="Shake right"),
        _f(245, 278, _S_FIST, 175, 278, _S_FIST, desc="Shiver — cold"),
    ],
    "NEW": [
        _f(260, 290, _BENT_HAND, 170, 280, _FLAT_HAND, desc="Bent hand scoops on palm"),
        _f(250, 275, _BENT_HAND, 175, 280, _FLAT_HAND, desc="Scoop across"),
        _f(240, 270, _BENT_HAND, 180, 280, _FLAT_HAND, desc="Complete scoop — new"),
        _f(245, 273, _BENT_HAND, 178, 280, _FLAT_HAND, desc="Hold"),
    ],
    "OLD": [
        _f(230, 200, _C_HAND, desc="C-hand at chin"),
        _f(225, 195, _C_HAND, desc="Grab chin"),
        _f(230, 250, _S_FIST, desc="Pull down closing fist"),
        _f(235, 280, _S_FIST, desc="Down — old"),
    ],
    "TOMORROW": [
        _f(250, 195, _OPEN_HAND, desc="Thumb on cheek"),
        _f(260, 190, _OPEN_HAND, desc="Arc forward"),
        _f(280, 200, _OPEN_HAND, desc="Forward — tomorrow"),
        _f(285, 205, _OPEN_HAND, desc="Hold"),
    ],
    "YESTERDAY": [
        _f(250, 195, _OPEN_HAND, desc="Thumb on cheek"),
        _f(240, 190, _OPEN_HAND, desc="Arc backward"),
        _f(230, 185, _OPEN_HAND, desc="Back — yesterday"),
        _f(232, 188, _OPEN_HAND, desc="Hold"),
    ],
    "TODAY": [
        _f(260, 270, _FLAT_HAND, 160, 270, _FLAT_HAND, desc="Both flat hands"),
        _f(255, 285, _FLAT_HAND, 165, 285, _FLAT_HAND, desc="Drop down together"),
        _f(258, 280, _FLAT_HAND, 162, 280, _FLAT_HAND, desc="Today"),
        _f(255, 283, _FLAT_HAND, 165, 283, _FLAT_HAND, desc="Hold"),
    ],
    "NIGHT": [
        _f(280, 260, _BENT_HAND, 160, 280, _FLAT_HAND, desc="Bent hand over flat"),
        _f(270, 270, _BENT_HAND, 165, 280, _FLAT_HAND, desc="Arc down"),
        _f(260, 280, _BENT_HAND, 170, 280, _FLAT_HAND, desc="Cover — night"),
        _f(262, 278, _BENT_HAND, 168, 280, _FLAT_HAND, desc="Hold"),
    ],
    "MORNING": [
        _f(170, 310, _FLAT_HAND, 160, 300, _FLAT_HAND, desc="Flat hand low"),
        _f(180, 280, _FLAT_HAND, 160, 290, _FLAT_HAND, desc="Rise like sun"),
        _f(200, 260, _FLAT_HAND, 160, 280, _FLAT_HAND, desc="Rising"),
        _f(210, 250, _OPEN_HAND, 160, 275, _FLAT_HAND, desc="Sun up — morning"),
    ],
    "SAME": [
        _f(260, 270, _POINT, 160, 270, _POINT, desc="Both index fingers"),
        _f(240, 268, _POINT, 180, 268, _POINT, desc="Bring together"),
        _f(215, 270, _POINT, 205, 270, _POINT, desc="Touch — same"),
        _f(218, 269, _POINT, 202, 269, _POINT, desc="Hold"),
    ],
    "BUT": [
        _f(240, 270, _POINT, 180, 270, _POINT, desc="Index fingers crossed"),
        _f(270, 265, _POINT, 150, 265, _POINT, desc="Pull apart"),
        _f(290, 260, _POINT, 130, 260, _POINT, desc="Separate — but"),
        _f(285, 262, _POINT, 135, 262, _POINT, desc="Hold"),
    ],
    "AND": [
        _f(250, 270, _OPEN_HAND, desc="Open hand"),
        _f(240, 270, _BENT_HAND, desc="Close while moving left"),
        _f(230, 270, _S_FIST, desc="Fist — and"),
        _f(232, 272, _S_FIST, desc="Hold"),
    ],
    "WITH": [
        _f(260, 280, _S_FIST, 160, 280, _S_FIST, desc="Both fists apart"),
        _f(240, 275, _S_FIST, 180, 275, _S_FIST, desc="Bring together"),
        _f(220, 272, _S_FIST, 200, 272, _S_FIST, desc="Together — with"),
        _f(222, 274, _S_FIST, 198, 274, _S_FIST, desc="Hold"),
    ],
    "FOR": [
        _f(250, 195, _POINT, desc="Index at forehead"),
        _f(245, 180, _POINT, desc="Touch forehead"),
        _f(270, 210, _POINT, desc="Twist outward"),
        _f(290, 230, _POINT, desc="Point forward — for"),
    ],
    "BECAUSE": [
        _f(250, 195, _POINT, desc="Index at forehead"),
        _f(245, 180, _POINT, desc="Touch forehead"),
        _f(260, 195, _OPEN_HAND, desc="Pull away opening hand"),
        _f(275, 210, _OPEN_HAND, desc="Open — because"),
    ],
    "IF": [
        _f(260, 270, _POINT, 160, 270, _POINT, desc="Both F-hands"),
        _f(255, 280, _POINT, 165, 280, _POINT, desc="Alternate up/down"),
        _f(260, 270, _POINT, 160, 270, _POINT, desc="Alternate — if"),
        _f(257, 275, _POINT, 163, 275, _POINT, desc="Hold"),
    ],
    "WHEN": [
        _f(270, 260, _POINT, 170, 280, _POINT, desc="Both index fingers"),
        _f(265, 265, _POINT, 175, 275, _POINT, desc="Right circles around left"),
        _f(260, 275, _POINT, 175, 270, _POINT, desc="Circle — when"),
        _f(265, 270, _POINT, 175, 272, _POINT, desc="Hold"),
    ],
    "ALL": [
        _f(280, 270, _OPEN_HAND, 150, 280, _FLAT_HAND, desc="Right hand circles over left"),
        _f(260, 260, _OPEN_HAND, 160, 280, _FLAT_HAND, desc="Circle around"),
        _f(240, 270, _OPEN_HAND, 170, 280, _FLAT_HAND, desc="Circle back"),
        _f(260, 280, _OPEN_HAND, 160, 280, _FLAT_HAND, desc="Land on palm — all"),
    ],
    "MANY": [
        _f(250, 270, _S_FIST, 170, 270, _S_FIST, desc="Both fists"),
        _f(260, 265, _OPEN_HAND, 160, 265, _OPEN_HAND, desc="Open up quickly"),
        _f(270, 260, _OPEN_HAND, 150, 260, _OPEN_HAND, desc="Spread — many"),
        _f(265, 262, _OPEN_HAND, 155, 262, _OPEN_HAND, desc="Hold"),
    ],
    "FAST": [
        _f(260, 270, _POINT, 160, 270, _POINT, desc="Both index fingers"),
        _f(250, 268, _S_FIST, 170, 268, _S_FIST, desc="Pull and close quickly"),
        _f(240, 270, _S_FIST, 180, 270, _S_FIST, desc="Snap — fast"),
        _f(242, 269, _S_FIST, 178, 269, _S_FIST, desc="Hold"),
    ],
    "SLOW": [
        _f(280, 280, _FLAT_HAND, 160, 290, _FLAT_HAND, desc="Right hand on left"),
        _f(275, 275, _FLAT_HAND, 160, 290, _FLAT_HAND, desc="Drag up slowly"),
        _f(270, 270, _FLAT_HAND, 160, 290, _FLAT_HAND, desc="Slow drag"),
        _f(265, 265, _FLAT_HAND, 160, 290, _FLAT_HAND, desc="Complete — slow"),
    ],
    "WAIT": [
        _f(270, 270, _OPEN_HAND, 150, 270, _OPEN_HAND, desc="Both open hands up"),
        _f(268, 268, _OPEN_HAND, 152, 268, _OPEN_HAND, desc="Wiggle fingers"),
        _f(272, 272, _OPEN_HAND, 148, 272, _OPEN_HAND, desc="Wiggle"),
        _f(270, 270, _OPEN_HAND, 150, 270, _OPEN_HAND, desc="Hold — wait"),
    ],
    "OPEN": [
        _f(240, 280, _FLAT_HAND, 180, 280, _FLAT_HAND, desc="Both flat hands together"),
        _f(260, 275, _FLAT_HAND, 160, 275, _FLAT_HAND, desc="Open apart"),
        _f(290, 270, _OPEN_HAND, 130, 270, _OPEN_HAND, desc="Wide open — open"),
        _f(285, 272, _OPEN_HAND, 135, 272, _OPEN_HAND, desc="Hold"),
    ],
    "CLOSE": [
        _f(290, 270, _FLAT_HAND, 130, 270, _FLAT_HAND, desc="Both hands apart"),
        _f(260, 275, _FLAT_HAND, 160, 275, _FLAT_HAND, desc="Come together"),
        _f(220, 278, _FLAT_HAND, 200, 278, _FLAT_HAND, desc="Close together — close"),
        _f(222, 277, _FLAT_HAND, 198, 277, _FLAT_HAND, desc="Hold"),
    ],
    "LEARN": [
        _f(280, 290, _OPEN_HAND, 160, 290, _FLAT_HAND, desc="Pick up from palm"),
        _f(270, 270, _CLAW, 165, 290, _FLAT_HAND, desc="Grab knowledge"),
        _f(260, 210, _S_FIST, 165, 290, _FLAT_HAND, desc="Bring to forehead"),
        _f(255, 185, _S_FIST, 165, 290, _FLAT_HAND, desc="Place at head — learn"),
    ],
    "TEACH": [
        _f(260, 200, _S_FIST, 160, 200, _S_FIST, desc="Both hands at forehead"),
        _f(265, 210, _S_FIST, 155, 210, _S_FIST, desc="Pull from head"),
        _f(275, 240, _OPEN_HAND, 145, 240, _OPEN_HAND, desc="Push outward opening"),
        _f(290, 260, _OPEN_HAND, 130, 260, _OPEN_HAND, desc="Give knowledge — teach"),
    ],
    "TALK": [
        _f(230, 210, _POINT, desc="Index at mouth"),
        _f(235, 195, _POINT, desc="Tap mouth"),
        _f(230, 205, _POINT, desc="Tap again"),
        _f(235, 195, _POINT, desc="Tap — talk"),
    ],
    "WALK": [
        _f(260, 310, _FLAT_HAND, 160, 310, _FLAT_HAND, desc="Both flat hands down"),
        _f(255, 305, _FLAT_HAND, 170, 305, _FLAT_HAND, desc="Alternate walking motion"),
        _f(265, 305, _FLAT_HAND, 155, 305, _FLAT_HAND, desc="Alternate"),
        _f(255, 305, _FLAT_HAND, 165, 305, _FLAT_HAND, desc="Walk"),
    ],
    "RUN": [
        _f(260, 290, _POINT, 160, 290, _POINT, desc="Both index out"),
        _f(250, 280, _POINT, 175, 285, _POINT, desc="Quick alternating"),
        _f(265, 275, _POINT, 155, 280, _POINT, desc="Fast motion"),
        _f(255, 278, _POINT, 170, 283, _POINT, desc="Run"),
    ],
    "PLAY": [
        _f(260, 270, _ILY, 160, 270, _ILY, desc="Both Y-hands"),
        _f(265, 265, _ILY, 155, 265, _ILY, desc="Shake/twist"),
        _f(255, 268, _ILY, 165, 268, _ILY, desc="Shake other way"),
        _f(260, 266, _ILY, 160, 266, _ILY, desc="Play"),
    ],
    "READ": [
        _f(270, 270, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         160, 280, _FLAT_HAND, desc="V-hand scans across palm"),
        _f(260, 275, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         165, 280, _FLAT_HAND, desc="Scan left"),
        _f(250, 270, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         170, 280, _FLAT_HAND, desc="Scan further — read"),
        _f(255, 272, {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         168, 280, _FLAT_HAND, desc="Hold"),
    ],
    "WRITE": [
        _f(260, 280, _S_FIST, 160, 290, _FLAT_HAND, desc="Writing hand on palm"),
        _f(250, 275, _S_FIST, 165, 290, _FLAT_HAND, desc="Write across left"),
        _f(240, 278, _S_FIST, 170, 290, _FLAT_HAND, desc="Continue writing"),
        _f(250, 276, _S_FIST, 165, 290, _FLAT_HAND, desc="Write"),
    ],
    "SLEEP": [
        _f(240, 210, _OPEN_HAND, desc="Open hand at face"),
        _f(235, 200, _OPEN_HAND, desc="Hand over face"),
        _f(230, 200, _BENT_HAND, desc="Close hand — eyes close"),
        _f(225, 205, _S_FIST, desc="Hand closes — sleep"),
    ],
    "SICK": [
        _f(250, 185, _OPEN_HAND, 180, 280, _OPEN_HAND, desc="Right at forehead, left at stomach"),
        _f(245, 180, _OPEN_HAND, 185, 285, _OPEN_HAND, desc="Tap forehead"),
        _f(250, 185, _OPEN_HAND, 180, 280, _OPEN_HAND, desc="Both touch — sick"),
        _f(248, 183, _OPEN_HAND, 182, 282, _OPEN_HAND, desc="Hold"),
    ],
    "TIRED": [
        _f(240, 270, _BENT_HAND, 180, 270, _BENT_HAND, desc="Both bent hands at chest"),
        _f(235, 280, _BENT_HAND, 185, 280, _BENT_HAND, desc="Hands droop down"),
        _f(230, 290, _BENT_HAND, 190, 290, _BENT_HAND, desc="Fall further"),
        _f(228, 295, _BENT_HAND, 192, 295, _BENT_HAND, desc="Tired"),
    ],
    "HUNGRY": [
        _f(230, 250, _C_HAND, desc="C-hand at throat"),
        _f(225, 270, _C_HAND, desc="Drag down chest"),
        _f(220, 300, _C_HAND, desc="Pull down — hungry"),
        _f(222, 305, _C_HAND, desc="Hold"),
    ],
    "THIRSTY": [
        _f(240, 210, _POINT, desc="Index at throat"),
        _f(235, 200, _POINT, desc="Touch throat"),
        _f(238, 220, _POINT, desc="Drag down"),
        _f(235, 240, _POINT, desc="Down — thirsty"),
    ],
    "BEAUTIFUL": [
        _f(230, 200, _OPEN_HAND, desc="Open hand at face"),
        _f(240, 195, _OPEN_HAND, desc="Circle around face"),
        _f(250, 200, _OPEN_HAND, desc="Circle continues"),
        _f(235, 200, _S_FIST, desc="Close at chin — beautiful"),
    ],
    "NICE": [
        _f(260, 280, _FLAT_HAND, 160, 290, _FLAT_HAND, desc="Right slides over left"),
        _f(250, 275, _FLAT_HAND, 165, 290, _FLAT_HAND, desc="Slide across"),
        _f(240, 278, _FLAT_HAND, 170, 290, _FLAT_HAND, desc="Complete — nice"),
        _f(245, 277, _FLAT_HAND, 168, 290, _FLAT_HAND, desc="Hold"),
    ],
    "WRONG": [
        _f(225, 200, _ILY, desc="Y-hand at chin"),
        _f(220, 195, _ILY, desc="Tap chin"),
        _f(225, 200, _ILY, desc="Tap again — wrong"),
        _f(222, 198, _ILY, desc="Hold"),
    ],
    "RIGHT": [
        _f(260, 270, _POINT, 160, 270, _POINT, desc="Both index fingers"),
        _f(250, 268, _POINT, 170, 268, _POINT, desc="Right on top of left"),
        _f(245, 270, _POINT, 175, 270, _POINT, desc="Tap — right/correct"),
        _f(248, 269, _POINT, 172, 269, _POINT, desc="Hold"),
    ],
    "REMEMBER": [
        _f(255, 185, _OPEN_HAND, desc="Thumb at forehead"),
        _f(250, 175, _OPEN_HAND, desc="Touch forehead"),
        _f(252, 180, _S_FIST, desc="Pull down closing"),
        _f(255, 200, _S_FIST, desc="To fist — remember"),
    ],
    "FORGET": [
        _f(250, 180, _FLAT_HAND, desc="Flat hand at forehead"),
        _f(255, 175, _FLAT_HAND, desc="Wipe across forehead"),
        _f(270, 180, _OPEN_HAND, desc="Hand opens away"),
        _f(285, 190, _OPEN_HAND, desc="Gone — forget"),
    ],
    "TRY": [
        _f(260, 280, _S_FIST, 160, 280, _S_FIST, desc="Both fists"),
        _f(265, 290, _S_FIST, 155, 290, _S_FIST, desc="Push forward"),
        _f(275, 300, _S_FIST, 145, 300, _S_FIST, desc="Push out — try"),
        _f(272, 298, _S_FIST, 148, 298, _S_FIST, desc="Hold"),
    ],
    "START": [
        _f(250, 270, _POINT, 170, 280, _FLAT_HAND, desc="Index in V of left hand"),
        _f(255, 268, _POINT, 170, 278, _FLAT_HAND, desc="Twist"),
        _f(260, 270, _POINT, 170, 280, _FLAT_HAND, desc="Twist — start"),
        _f(258, 269, _POINT, 170, 279, _FLAT_HAND, desc="Hold"),
    ],
    "PRACTICE": [
        _f(260, 280, _POINT, 170, 290, _S_FIST, desc="Index brushes fist"),
        _f(255, 275, _POINT, 170, 290, _S_FIST, desc="Brush left"),
        _f(265, 275, _POINT, 170, 290, _S_FIST, desc="Brush right"),
        _f(255, 278, _POINT, 170, 290, _S_FIST, desc="Practice"),
    ],
    "MEET": [
        _f(280, 270, _POINT, 140, 270, _POINT, desc="Both index fingers approach"),
        _f(260, 268, _POINT, 160, 268, _POINT, desc="Coming together"),
        _f(220, 270, _POINT, 200, 270, _POINT, desc="Meet in middle"),
        _f(222, 269, _POINT, 198, 269, _POINT, desc="Hold — meet"),
    ],
    "PROBLEM": [
        _f(255, 185, _BENT_HAND, desc="Bent hand at forehead"),
        _f(250, 180, _BENT_HAND, desc="Touch forehead"),
        _f(255, 185, _BENT_HAND, desc="Twist"),
        _f(260, 180, _BENT_HAND, desc="Twist — problem"),
    ],
    "IMPORTANT": [
        _f(250, 280, _POINT, 170, 290, _FLAT_HAND, desc="F-hand rises from palm"),
        _f(245, 260, _POINT, 175, 290, _FLAT_HAND, desc="Rise up"),
        _f(240, 240, _POINT, 180, 290, _FLAT_HAND, desc="Reach top — important"),
        _f(242, 245, _POINT, 178, 290, _FLAT_HAND, desc="Hold"),
    ],
    "CHANGE": [
        _f(250, 280, _S_FIST, 170, 280, _S_FIST, desc="Both fists together"),
        _f(255, 275, _S_FIST, 165, 275, _S_FIST, desc="Twist right"),
        _f(245, 280, _S_FIST, 175, 280, _S_FIST, desc="Twist left"),
        _f(250, 278, _S_FIST, 170, 278, _S_FIST, desc="Swap — change"),
    ],
    "MONEY": [
        _f(260, 280, _FLAT_HAND, 170, 290, _FLAT_HAND, desc="Flat hand taps palm"),
        _f(255, 275, _FLAT_HAND, 170, 290, _FLAT_HAND, desc="Tap"),
        _f(260, 280, _FLAT_HAND, 170, 290, _FLAT_HAND, desc="Tap again — money"),
        _f(258, 278, _FLAT_HAND, 170, 290, _FLAT_HAND, desc="Hold"),
    ],
    "QUESTION": [
        _f(270, 250, _POINT, desc="Index draws question mark"),
        _f(265, 235, _POINT, desc="Curve up"),
        _f(275, 245, _POINT, desc="Curve over"),
        _f(270, 260, _POINT, desc="Dot — question?"),
    ],
    "ANSWER": [
        _f(260, 195, _POINT, 180, 195, _POINT, desc="Both index at mouth"),
        _f(270, 215, _POINT, 170, 215, _POINT, desc="Drop down"),
        _f(280, 240, _POINT, 160, 240, _POINT, desc="Point outward — answer"),
        _f(278, 238, _POINT, 162, 238, _POINT, desc="Hold"),
    ],
}


class SignAnimator:
    """Generates ASL sign illustrations with a fixed character + AI hand configs."""

    def __init__(self, gemini_api_key: str = "", lava_token: str = "",
                 cache_path: str = _DEFAULT_CACHE):
        self._cache_path = cache_path
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._lava_token = lava_token
        self._gemini_client = None

        if gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=gemini_api_key)
                print("[anim] Gemini client initialized")
            except Exception as e:
                print(f"[anim] Gemini init failed: {e}")

        if lava_token:
            print("[anim] Lava/GPT-4o fallback available")

        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path) as f:
                    self._cache = json.load(f)
                print(f"[anim] Loaded {len(self._cache)} cached animations")
            except Exception:
                self._cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, separators=(",", ":"))
        except Exception as e:
            print(f"[anim] cache save error: {e}")

    def get_animation(self, sign: str) -> dict:
        key = sign.strip().upper()
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Check hardcoded signs first (multi-frame, accurate)
        if key in _HARDCODED_SIGNS:
            raw = _HARDCODED_SIGNS[key]
            # Single interpolation for signs with many keyframes, double for short ones
            frames = _interpolate_frames(raw)
            if len(raw) <= 6:
                frames = _interpolate_frames(frames)
            svgs = [_build_sign_svg(f, label=key) for f in frames]
            # Compute 3D hand landmarks for each keyframe (not interpolated — too many)
            hand_data = []
            for kf in raw:
                rh = kf.get("right_hand", {})
                lh = kf.get("left_hand", {})
                hand_data.append({
                    "right": hand_config_to_landmarks(rh),
                    "left": hand_config_to_landmarks(lh),
                })
            result = {"type": "svg_multi", "frames": svgs, "content": svgs[0],
                      "hand_data": hand_data}
        else:
            # AI fallback for unknown signs
            config = self._get_hand_config(key)
            svg = _build_sign_svg(config, label=key)
            result = {"type": "svg", "content": svg}

        with self._lock:
            self._cache[key] = result
            self._save_cache()

        return result

    def has_cached(self, sign: str) -> bool:
        return sign.strip().upper() in self._cache

    @property
    def idle_svg(self) -> str:
        return _IDLE_SVG

    @property
    def cached_signs(self) -> list[str]:
        return sorted(self._cache.keys())

    def _get_hand_config(self, sign: str) -> dict:
        """Ask AI for hand configuration JSON."""
        # Try Gemini
        if self._gemini_client:
            config = self._ask_gemini(sign)
            if config:
                return config

        # Fallback to Lava
        if self._lava_token:
            config = self._ask_lava(sign)
            if config:
                return config

        return _IDLE_CONFIG

    def _ask_gemini(self, sign: str) -> dict | None:
        print(f"[anim] Gemini config for {sign}...")
        try:
            resp = self._gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"ASL sign: {sign}",
                config={"system_instruction": _SYSTEM_PROMPT, "temperature": 0.3, "max_output_tokens": 1024},
            )
            return self._parse_json(resp.text.strip())
        except Exception as e:
            print(f"[anim] Gemini error: {e}")
        return None

    def _ask_lava(self, sign: str) -> dict | None:
        print(f"[anim] Lava config for {sign}...")
        try:
            r = requests.post(
                "https://api.lavapayments.com/v1/forward",
                params={"u": "https://api.openai.com/v1/chat/completions"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": f"ASL sign: {sign}"},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                },
                headers={
                    "Authorization": f"Bearer {self._lava_token}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            return self._parse_json(raw)
        except Exception as e:
            print(f"[anim] Lava error: {e}")
        return None

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            data = json.loads(raw)
            if "right_hand" in data or "left_hand" in data:
                return data
        except json.JSONDecodeError:
            # Try to find JSON in the response
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return None
