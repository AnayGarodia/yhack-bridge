"""
SVG Sign Generator — uses Gemini to create Bitmoji-style ASL sign SVGs.

Generates on first encounter, caches to disk for instant reuse.
"""

import json
import logging
import os
import re
import threading

import requests
from google import genai

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, "data", "sign_svgs.json")

_SYSTEM_PROMPT = """\
You are an SVG artist specializing in ASL (American Sign Language) illustrations.
Generate a single SVG of a friendly Bitmoji-style cartoon character performing an ASL sign.

Character (MUST be identical every time):
- Warm beige skin (#E1B48C), dark outlines (#2D2319, 2px)
- Round head, short dark brown hair (#352D28)
- Simple friendly eyes (white sclera, dark brown iris #3E2F1C, black pupil)
- Small upward-curved smile (#C47A6A)
- Teal crew-neck shirt (#3C78B4) with subtle collar shadow (#2D5F8E)
- Show from waist up. Arms and hands are skin-colored.

Hand requirements:
- Hands MUST clearly show the correct ASL hand shape for the given sign
- Arms positioned where the sign is performed (face level, chest level, etc.)
- Each finger must be individually visible when extended
- Curled fingers shown as small curved bumps on the fist

SVG requirements:
- viewBox="0 0 400 500"
- Dark background: <rect width="400" height="500" fill="#141414" rx="16"/>
- All elements use filled paths with 2px dark outlines
- No gradients, no filters, no animations — static pose only
- No text elements

Output ONLY the raw SVG code starting with <svg and ending with </svg>. No markdown, no explanation."""

_IDLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
  <rect width="400" height="500" fill="#141414" rx="16"/>
  <!-- Body -->
  <rect x="155" y="260" width="90" height="120" rx="10" fill="#3C78B4" stroke="#2D2319" stroke-width="2"/>
  <rect x="155" y="260" width="90" height="15" rx="5" fill="#2D5F8E"/>
  <!-- Neck -->
  <rect x="185" y="240" width="30" height="25" fill="#E1B48C"/>
  <!-- Head -->
  <ellipse cx="200" cy="200" rx="55" ry="62" fill="#E1B48C" stroke="#2D2319" stroke-width="2"/>
  <!-- Hair -->
  <ellipse cx="200" cy="175" rx="57" ry="40" fill="#352D28"/>
  <rect x="143" y="175" width="114" height="20" fill="#352D28"/>
  <!-- Eyes -->
  <ellipse cx="180" cy="200" rx="8" ry="9" fill="white" stroke="#2D2319" stroke-width="1"/>
  <circle cx="180" cy="200" r="4" fill="#3E2F1C"/>
  <circle cx="180" cy="200" r="2" fill="black"/>
  <circle cx="179" cy="198" r="1.5" fill="white"/>
  <ellipse cx="220" cy="200" rx="8" ry="9" fill="white" stroke="#2D2319" stroke-width="1"/>
  <circle cx="220" cy="200" r="4" fill="#3E2F1C"/>
  <circle cx="220" cy="200" r="2" fill="black"/>
  <circle cx="219" cy="198" r="1.5" fill="white"/>
  <!-- Eyebrows -->
  <path d="M170 186 Q180 182 190 186" stroke="#352D28" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="M210 186 Q220 182 230 186" stroke="#352D28" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- Smile -->
  <path d="M185 218 Q200 228 215 218" stroke="#C47A6A" stroke-width="2" fill="none" stroke-linecap="round"/>
  <!-- Left arm (hanging) -->
  <path d="M155 275 Q135 310 140 370" stroke="#2D2319" stroke-width="2" fill="none"/>
  <path d="M155 275 Q133 310 138 370" stroke="none" fill="#E1B48C"/>
  <rect x="128" y="270" width="30" height="100" rx="14" fill="#E1B48C" stroke="#2D2319" stroke-width="2" transform="rotate(-5 143 320)"/>
  <!-- Right arm (hanging) -->
  <rect x="242" y="270" width="30" height="100" rx="14" fill="#E1B48C" stroke="#2D2319" stroke-width="2" transform="rotate(5 257 320)"/>
  <!-- Shirt sleeves -->
  <rect x="133" y="265" width="28" height="20" rx="8" fill="#3C78B4" stroke="#2D2319" stroke-width="2" transform="rotate(-5 147 275)"/>
  <rect x="239" y="265" width="28" height="20" rx="8" fill="#3C78B4" stroke="#2D2319" stroke-width="2" transform="rotate(5 253 275)"/>
</svg>"""


class SVGSignGenerator:
    """Generates and caches Bitmoji-style ASL sign SVGs using Gemini."""

    def __init__(self, gemini_api_key: str = "", lava_token: str = "",
                 cache_path: str = _DEFAULT_CACHE):
        self._cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._client = None
        self._lava_token = lava_token

        if gemini_api_key:
            self._client = genai.Client(api_key=gemini_api_key)
            print("[svg] Gemini client initialized")
        elif lava_token:
            print("[svg] Using Lava gateway for SVG generation (GPT-4o)")
        else:
            print("[svg] No API key — SVG generation disabled")

        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path) as f:
                    self._cache = json.load(f)
                logger.info("[svg] Loaded %d cached sign SVGs", len(self._cache))
            except Exception as e:
                logger.warning("[svg] Failed to load cache: %s", e)
                self._cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning("[svg] Failed to save cache: %s", e)

    def get_svg(self, sign: str) -> str:
        """Get SVG for a sign (cached or freshly generated)."""
        key = sign.strip().upper()

        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Generate
        svg = self.generate(key)
        return svg

    def has_cached(self, sign: str) -> bool:
        return sign.strip().upper() in self._cache

    def generate(self, sign: str) -> str:
        """Generate SVG via Gemini (primary) or Lava/GPT-4o (fallback)."""
        key = sign.strip().upper()

        # Try Gemini first
        if self._client is not None:
            svg = self._generate_gemini(key)
            if svg:
                return svg

        # Fallback to Lava gateway
        if self._lava_token:
            svg = self._generate_lava(key)
            if svg:
                return svg

        print(f"[svg] no API available for {key}")
        return _IDLE_SVG

    def _generate_gemini(self, key: str) -> str | None:
        print(f"[svg] Gemini generating {key}...")
        try:
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"ASL sign: {key}",
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.4,
                    "max_output_tokens": 4096,
                },
            )
            raw = response.text.strip()
            svg = self._extract_svg(raw)
            if svg:
                self._cache_svg(key, svg)
                return svg
            print(f"[svg] Gemini: no valid SVG in response for {key}")
        except Exception as e:
            print(f"[svg] Gemini error for {key}: {e}")
        return None

    def _generate_lava(self, key: str) -> str | None:
        print(f"[svg] Lava/GPT-4o generating {key}...")
        try:
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"ASL sign: {key}"},
                ],
                "max_tokens": 4096,
                "temperature": 0.4,
            }
            r = requests.post(
                "https://api.lavapayments.com/v1/forward",
                params={"u": "https://api.openai.com/v1/chat/completions"},
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._lava_token}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            svg = self._extract_svg(raw)
            if svg:
                self._cache_svg(key, svg)
                return svg
            print(f"[svg] Lava: no valid SVG in response for {key}")
        except Exception as e:
            print(f"[svg] Lava error for {key}: {e}")
        return None

    def _cache_svg(self, key: str, svg: str):
        with self._lock:
            self._cache[key] = svg
            self._save_cache()
        print(f"[svg] cached {key} ({len(svg)} chars)")

    def _extract_svg(self, raw: str) -> str | None:
        """Extract <svg>...</svg> from Gemini response."""
        # Try to find SVG tags
        match = re.search(r"<svg[\s\S]*?</svg>", raw, re.IGNORECASE)
        if match:
            return match.group(0)

        # Maybe it's wrapped in markdown code blocks
        match = re.search(r"```(?:svg|xml)?\s*([\s\S]*?)```", raw)
        if match:
            inner = match.group(1).strip()
            svg_match = re.search(r"<svg[\s\S]*?</svg>", inner, re.IGNORECASE)
            if svg_match:
                return svg_match.group(0)

        return None

    @property
    def idle_svg(self) -> str:
        return _IDLE_SVG

    @property
    def cached_signs(self) -> list[str]:
        return sorted(self._cache.keys())
