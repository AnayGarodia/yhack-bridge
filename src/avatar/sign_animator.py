"""
SignAnimator — generates animated ASL sign SVGs with CSS @keyframes.

Uses Gemini (primary) or Lava/GPT-4o (fallback) to create SVGs where
the character's hands animate from neutral to the sign position.
Caches to disk for instant reuse.
"""

import json
import logging
import os
import re
import threading

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, "data", "sign_animations.json")

_SYSTEM_PROMPT = """\
You are an expert SVG animator specializing in ASL (American Sign Language).
Generate an ANIMATED SVG of a Bitmoji-style character performing an ASL sign.

CHARACTER (must be identical every time):
- Warm beige skin (#E1B48C), 2px dark outlines (#2D2319)
- Round head, short dark brown hair (#352D28)
- Friendly eyes: white sclera, brown iris (#3E2F1C), black pupil, white highlight dot
- Small curved smile (#C47A6A)
- Teal crew-neck shirt (#3C78B4)
- Upper body only (head, shoulders, arms, hands)
- Background: dark rounded rect (#141414, rx=16)

SVG SPECS:
- viewBox="0 0 400 500"
- xmlns="http://www.w3.org/2000/svg"

ANIMATION REQUIREMENTS (critical):
- Include a <style> block with CSS @keyframes animations
- The character's arms and hands MUST animate from a neutral resting position to the correct ASL sign position
- Animation: duration 0.8s, ease-in-out, forwards (hold final pose)
- Each moving part (upper arms, forearms, hands, individual fingers) should be a separate <g> or <path> with its own animation
- Use transform: translate() and rotate() for smooth motion
- Hands must clearly show the correct finger positions for the ASL sign
- Extended fingers should be individually visible as separate paths
- Curled fingers shown as small rounded shapes against the palm

STRUCTURE:
1. <rect> background
2. Static elements: torso/shirt, head, face features, hair
3. Animated elements: left arm group, right arm group, left hand group, right hand group
4. Each animated group has: animation-name, animation-duration: 0.8s, animation-timing-function: ease-in-out, animation-fill-mode: forwards

Output ONLY the SVG code. No markdown fences, no explanation. Start with <svg, end with </svg>."""

_IDLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
  <rect width="400" height="500" fill="#141414" rx="16"/>
  <rect x="155" y="260" width="90" height="120" rx="10" fill="#3C78B4" stroke="#2D2319" stroke-width="2"/>
  <rect x="155" y="260" width="90" height="15" rx="5" fill="#2D5F8E"/>
  <rect x="185" y="240" width="30" height="25" fill="#E1B48C"/>
  <ellipse cx="200" cy="200" rx="55" ry="62" fill="#E1B48C" stroke="#2D2319" stroke-width="2"/>
  <ellipse cx="200" cy="175" rx="57" ry="40" fill="#352D28"/>
  <rect x="143" y="175" width="114" height="20" fill="#352D28"/>
  <ellipse cx="180" cy="200" rx="8" ry="9" fill="white" stroke="#2D2319" stroke-width="1"/>
  <circle cx="180" cy="200" r="4" fill="#3E2F1C"/><circle cx="180" cy="200" r="2" fill="black"/>
  <circle cx="179" cy="198" r="1.5" fill="white"/>
  <ellipse cx="220" cy="200" rx="8" ry="9" fill="white" stroke="#2D2319" stroke-width="1"/>
  <circle cx="220" cy="200" r="4" fill="#3E2F1C"/><circle cx="220" cy="200" r="2" fill="black"/>
  <circle cx="219" cy="198" r="1.5" fill="white"/>
  <path d="M170 186 Q180 182 190 186" stroke="#352D28" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="M210 186 Q220 182 230 186" stroke="#352D28" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="M185 218 Q200 228 215 218" stroke="#C47A6A" stroke-width="2" fill="none" stroke-linecap="round"/>
  <rect x="128" y="270" width="30" height="100" rx="14" fill="#E1B48C" stroke="#2D2319" stroke-width="2" transform="rotate(-5 143 320)"/>
  <rect x="242" y="270" width="30" height="100" rx="14" fill="#E1B48C" stroke="#2D2319" stroke-width="2" transform="rotate(5 257 320)"/>
  <rect x="133" y="265" width="28" height="20" rx="8" fill="#3C78B4" stroke="#2D2319" stroke-width="2" transform="rotate(-5 147 275)"/>
  <rect x="239" y="265" width="28" height="20" rx="8" fill="#3C78B4" stroke="#2D2319" stroke-width="2" transform="rotate(5 253 275)"/>
</svg>"""


class SignAnimator:
    """Generates and caches animated ASL sign SVGs."""

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
        """Get animation for a sign. Returns {"type": "svg", "content": "<svg>..."}."""
        key = sign.strip().upper()

        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Generate animated SVG
        svg = self._generate(key)
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

    def _generate(self, key: str) -> str:
        """Try Gemini, then Lava/GPT-4o, then return idle."""
        # Try Gemini
        if self._gemini_client:
            svg = self._gen_gemini(key)
            if svg:
                return svg

        # Fallback to Lava
        if self._lava_token:
            svg = self._gen_lava(key)
            if svg:
                return svg

        print(f"[anim] no API for {key}, using idle")
        return _IDLE_SVG

    def _gen_gemini(self, key: str) -> str | None:
        print(f"[anim] Gemini generating {key}...")
        try:
            resp = self._gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"ASL sign: {key}",
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.5,
                    "max_output_tokens": 8192,
                },
            )
            svg = self._extract_svg(resp.text.strip())
            if svg:
                print(f"[anim] Gemini OK: {key} ({len(svg)} chars)")
                return svg
        except Exception as e:
            print(f"[anim] Gemini error: {e}")
        return None

    def _gen_lava(self, key: str) -> str | None:
        print(f"[anim] Lava/GPT-4o generating {key}...")
        try:
            r = requests.post(
                "https://api.lavapayments.com/v1/forward",
                params={"u": "https://api.openai.com/v1/chat/completions"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": f"ASL sign: {key}"},
                    ],
                    "max_tokens": 8192,
                    "temperature": 0.5,
                },
                headers={
                    "Authorization": f"Bearer {self._lava_token}",
                    "Content-Type": "application/json",
                },
                timeout=45,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            svg = self._extract_svg(raw)
            if svg:
                print(f"[anim] Lava OK: {key} ({len(svg)} chars)")
                return svg
        except Exception as e:
            print(f"[anim] Lava error: {e}")
        return None

    @staticmethod
    def _extract_svg(raw: str) -> str | None:
        match = re.search(r"<svg[\s\S]*?</svg>", raw, re.IGNORECASE)
        if match:
            return match.group(0)
        match = re.search(r"```(?:svg|xml|html)?\s*([\s\S]*?)```", raw)
        if match:
            inner = match.group(1).strip()
            svg_match = re.search(r"<svg[\s\S]*?</svg>", inner, re.IGNORECASE)
            if svg_match:
                return svg_match.group(0)
        return None
