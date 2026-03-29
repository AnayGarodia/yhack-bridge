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

# ── Hardcoded ASL sign definitions (multi-frame for motion) ──────────────────
# Each sign is a list of frames. The client cycles through them.
# Format: [{"right_hand": {...}, "left_hand": {...}, "description": "..."}]
_HARDCODED_SIGNS = {
    "HELLO": [
        {"right_hand": {"x": 280, "y": 150, **_OPEN_HAND}, "left_hand": _REST_L,
         "description": "Open hand at forehead, palm out"},
        {"right_hand": {"x": 320, "y": 140, **_OPEN_HAND}, "left_hand": _REST_L,
         "description": "Wave hand outward from forehead"},
        {"right_hand": {"x": 340, "y": 155, **_OPEN_HAND}, "left_hand": _REST_L,
         "description": "Complete the wave to the side"},
    ],
    "THANK-YOU": [
        {"right_hand": {"x": 230, "y": 190, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Flat hand touches chin"},
        {"right_hand": {"x": 260, "y": 230, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Hand moves outward from chin"},
        {"right_hand": {"x": 300, "y": 280, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Hand extends forward — thank you"},
    ],
    "YES": [
        {"right_hand": {"x": 290, "y": 200, **_S_FIST}, "left_hand": _REST_L,
         "description": "Fist nods down (like a nodding head)"},
        {"right_hand": {"x": 290, "y": 240, **_S_FIST}, "left_hand": _REST_L,
         "description": "Fist moves down — nodding yes"},
        {"right_hand": {"x": 290, "y": 200, **_S_FIST}, "left_hand": _REST_L,
         "description": "Fist nods back up"},
    ],
    "NO": [
        {"right_hand": {"x": 290, "y": 200, "thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"},
         "left_hand": _REST_L, "description": "Index+middle extended, open"},
        {"right_hand": {"x": 290, "y": 200, "thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         "left_hand": _REST_L, "description": "Fingers snap shut to thumb — NO"},
    ],
    "PLEASE": [
        {"right_hand": {"x": 220, "y": 280, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Flat hand on chest"},
        {"right_hand": {"x": 240, "y": 300, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Hand circles on chest — please"},
        {"right_hand": {"x": 220, "y": 310, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Complete the circular motion"},
    ],
    "HELP": [
        {"right_hand": {"x": 200, "y": 310, **_S_FIST},
         "left_hand": {"x": 200, "y": 280, **_FLAT_HAND},
         "description": "Fist on open palm"},
        {"right_hand": {"x": 200, "y": 270, **_S_FIST},
         "left_hand": {"x": 200, "y": 260, **_FLAT_HAND},
         "description": "Both hands rise together — help"},
    ],
    "WATER": [
        {"right_hand": {"x": 230, "y": 185, "thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"},
         "left_hand": _REST_L, "description": "W-handshape taps chin"},
        {"right_hand": {"x": 230, "y": 175, "thumb": "extended", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "outward"},
         "left_hand": _REST_L, "description": "W taps chin twice — water"},
    ],
    "NAME": [
        {"right_hand": {"x": 260, "y": 230, "thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "sideways"},
         "left_hand": {"x": 180, "y": 250, "thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled", "palm_facing": "sideways"},
         "description": "H-fingers tap on H-fingers — name"},
    ],
    "MY": [
        {"right_hand": {"x": 210, "y": 280, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Flat hand on chest — my/mine"},
    ],
    "YOU": [
        {"right_hand": {"x": 300, "y": 240, **_POINT}, "left_hand": _REST_L,
         "description": "Point index finger forward — you"},
    ],
    "HAPPY": [
        {"right_hand": {"x": 230, "y": 260, **_FLAT_HAND},
         "left_hand": {"x": 170, "y": 260, **_FLAT_HAND},
         "description": "Both hands brush up on chest"},
        {"right_hand": {"x": 230, "y": 230, **_FLAT_HAND},
         "left_hand": {"x": 170, "y": 230, **_FLAT_HAND},
         "description": "Hands sweep upward — happy"},
    ],
    "SORRY": [
        {"right_hand": {"x": 220, "y": 270, **_S_FIST}, "left_hand": _REST_L,
         "description": "Fist circles on chest — sorry"},
        {"right_hand": {"x": 240, "y": 290, **_S_FIST}, "left_hand": _REST_L,
         "description": "Circular motion continues"},
    ],
    "GOOD": [
        {"right_hand": {"x": 230, "y": 190, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Flat hand at chin"},
        {"right_hand": {"x": 250, "y": 280, **_FLAT_HAND},
         "left_hand": {"x": 180, "y": 300, **_FLAT_HAND},
         "description": "Hand drops to open palm — good"},
    ],
    "BAD": [
        {"right_hand": {"x": 230, "y": 190, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Flat hand at chin, palm in"},
        {"right_hand": {"x": 250, "y": 290, **_FLAT_HAND}, "left_hand": _REST_L,
         "description": "Hand flips down — bad"},
    ],
    "I": [
        {"right_hand": {"x": 215, "y": 280, **_POINT}, "left_hand": _REST_L,
         "description": "Point to self — I/me"},
    ],
    "WANT": [
        {"right_hand": {"x": 270, "y": 260, **_OPEN_HAND},
         "left_hand": {"x": 150, "y": 260, **_OPEN_HAND},
         "description": "Both hands open, pull toward self"},
        {"right_hand": {"x": 250, "y": 280, "thumb": "bent", "index": "bent", "middle": "bent", "ring": "bent", "pinky": "bent", "palm_facing": "inward"},
         "left_hand": {"x": 160, "y": 280, "thumb": "bent", "index": "bent", "middle": "bent", "ring": "bent", "pinky": "bent", "palm_facing": "inward"},
         "description": "Hands curl inward — want"},
    ],
    "EAT": [
        {"right_hand": {"x": 240, "y": 190, **_S_FIST}, "left_hand": _REST_L,
         "description": "Bunched fingers tap mouth"},
        {"right_hand": {"x": 235, "y": 185, **_S_FIST}, "left_hand": _REST_L,
         "description": "Tap mouth repeatedly — eat"},
    ],
    "DRINK": [
        {"right_hand": {"x": 235, "y": 200, "thumb": "extended", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         "left_hand": _REST_L, "description": "C-shape at mouth, tip up"},
        {"right_hand": {"x": 235, "y": 185, "thumb": "extended", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled", "palm_facing": "inward"},
         "left_hand": _REST_L, "description": "Tip hand up to mouth — drink"},
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
            frames = _HARDCODED_SIGNS[key]
            svgs = [_build_sign_svg(f, label=key) for f in frames]
            result = {"type": "svg_multi", "frames": svgs, "content": svgs[0]}
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
