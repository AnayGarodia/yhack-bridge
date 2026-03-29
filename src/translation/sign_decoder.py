"""
SignDecoder — decodes a sequence of ASL signs from a stream of model predictions.

Takes timestamped top-5 probability snapshots and asks gpt-4o-mini to find
the most likely sign sequence (handling noisy/overlapping predictions).

Uses frequency priors: common conversational signs (hello, please, name) are
boosted; rare/specific nouns (scissors, elephant, fireman) need much higher
confidence to survive.

Usage:
    decoder = SignDecoder(lava_token="...")
    signs = decoder.decode([
        {"t": 0.0,  "top5": [("hello", 0.82), ("dad", 0.05), ...]},
        {"t": 0.5,  "top5": [("hello", 0.45), ("dad", 0.35), ...]},
        {"t": 1.0,  "top5": [("dad", 0.72), ("hello", 0.12), ...]},
    ])
    # → ["hello", "dad"]
"""

import os
import re

import requests

LAVA_BASE = "https://api.lavapayments.com/v1/forward"
_DEFAULT_PROVIDER_URL = os.environ.get(
    "LAVA_PROVIDER_URL", "https://api.openai.com/v1/chat/completions"
)
_DEFAULT_MODEL = os.environ.get("E2S_MODEL", "gpt-4o-mini")

# ── Frequency tiers ─────────────────────────────────────────────────────────
# Weight multiplier applied to raw probabilities in the fallback voter.
# Higher = model needs LESS confidence to accept.  Lower = needs MORE.
#
# Tier 1 (1.5x): core conversational signs — greetings, pronouns, common verbs
# Tier 2 (1.0x): default — everything not listed
# Tier 3 (0.5x): rare / specific nouns unlikely in casual conversation

_HIGH_FREQ = {
    "hello", "bye", "yes", "no", "please", "thankyou", "sorry", "help",
    "want", "like", "have", "haveto", "need", "can",
    "name", "minemy", "you", "yourself", "hesheit", "weus",
    "mom", "dad", "brother", "grandma", "grandpa",
    "happy", "sad", "good", "bad", "fine",
    "food", "water", "eat", "drink", "hungry", "thirsty",
    "home", "school", "work", "friend", "family",
    "go", "come", "see", "look", "think", "know", "say", "talk",
    "what", "where", "who", "how", "why", "that", "there",
    "more", "finish", "not", "now", "will", "if", "because", "for",
    "love", "give", "make", "find", "hear", "listen", "understand",
    "morning", "night", "today", "tomorrow", "yesterday", "time",
    "hot", "sick", "clean", "open", "close",
    "up", "down", "on", "all", "every", "same", "another", "any",
    "first", "before", "after", "later", "better",
}

_LOW_FREQ = {
    "scissors", "elephant", "alligator", "helicopter", "fireman", "cowboy",
    "zebra", "giraffe", "donkey", "refrigerator", "toothbrush", "vacuum",
    "zipper", "pajamas", "underwear", "frenchfries", "icecream", "glasswindow",
    "callonphone", "dryer", "napkin", "mitten", "penny", "pencil", "pen",
    "puzzle", "pretend", "radio", "clown", "flag", "lamp", "drawer",
    "alligator", "goose", "hen", "frog", "bug", "bee", "owl", "wolf",
    "pig", "cow", "horse", "lion", "tiger", "bear", "duck", "kitty",
    "puppy", "mouse", "bird", "fish", "spider",
    "cereal", "carrot", "gum", "nuts", "chocolate", "pizza", "snack",
    "pool", "potty", "closet", "backyard", "stairs", "store", "farm",
    "balloon", "doll", "toy",
    "sticky", "noisy", "loud", "yucky", "empty", "wet", "dry", "dirty",
    "grass", "snow", "rain", "cloud", "sun", "moon",
}

def _freq_weight(sign_name: str) -> float:
    s = sign_name.lower()
    if s in _HIGH_FREQ:
        return 1.5
    if s in _LOW_FREQ:
        return 0.5
    return 1.0

# ── Known model confusions ─────────────────────────────────────────────────
# Pairs where the TFLite model frequently mixes up signs.
# The decoder prompt includes these so the LLM can pick the more likely one.
_KNOWN_CONFUSIONS = [
    ("scissors", "name"),
    ("scissors", "cut"),
    ("hat", "think"),
    ("hat", "know"),
    ("grass", "please"),
    ("empty", "finish"),
    ("taste", "food"),
    ("elephant", "that"),
    ("sun", "no"),
    ("fireman", "red"),
]

_SYSTEM_PROMPT = """\
You are an ASL sign sequence decoder. You receive a time-series of probability \
distributions from an ASL recognition model. Each line has a timestamp and the \
top predicted signs with their probabilities.

Your job: determine what sequence of ASL signs the person performed.

How to read the data:
- A sign typically dominates (high probability) for several consecutive snapshots
- When the signer transitions to a new sign, the old sign's probability drops \
and a new sign rises
- Brief spikes of a different sign amid a stable prediction are noise, not real signs
- A sign must appear as the top prediction for at least 2 snapshots to count
- If one sign dominates the ENTIRE stream, the person signed one word
- Consecutive duplicate predictions = same single sign, not repeated signing

FREQUENCY PRIORS — prefer common words over rare ones:
- HIGH frequency (prefer these): hello, bye, yes, no, please, thankyou, sorry, \
help, want, like, have, name, minemy, you, mom, dad, happy, sad, good, bad, \
food, water, eat, drink, home, go, see, think, know, what, where, who, how, \
more, finish, not, now, love, give, make, find, hear, understand
- LOW frequency (need strong sustained evidence): scissors, elephant, fireman, \
cowboy, zebra, giraffe, helicopter, vacuum, zipper, pajamas, refrigerator, \
clown, alligator, grass, empty, sun, moon, rain, snow, balloon, doll

When a HIGH-freq and LOW-freq sign compete at similar probabilities, prefer \
the HIGH-freq sign. A LOW-freq sign should only win if it dominates with >60% \
probability for multiple consecutive snapshots.

KNOWN MODEL CONFUSIONS — the recognition model often confuses these pairs:
- scissors <-> name, cut (if you see "scissors", consider whether "name" or "cut" fits better)
- hat <-> think, know (if you see "hat", consider whether "think" fits better)
- grass <-> please
- empty <-> finish
- taste <-> food
- elephant <-> that
- sun <-> no
- fireman <-> red

Rules:
- Output ONLY the sign sequence as lowercase space-separated words on ONE line
- Do not add signs that never appear in the top predictions
- Deduplicate consecutive repetitions
- Short streams (1-3 snapshots) are almost always a single sign
- If the stream is empty or unclear, output the single most confident sign
- When in doubt between a rare word and a common word, choose the common word\
"""


class SignDecoder:
    def __init__(self, lava_token: str, timeout: int = 3):
        self._lava_token = lava_token
        self._timeout = timeout
        self._provider_url = _DEFAULT_PROVIDER_URL
        self._model = _DEFAULT_MODEL

    def decode(self, prediction_stream: list[dict], context: list[str] | None = None) -> list[str]:
        """
        Decode a sign sequence from a prediction stream.

        Args:
            prediction_stream: list of {"t": float, "top5": [(name, prob), ...]}
            context: recent English sentences from the conversation (for LLM grounding)
        Returns:
            Ordered list of sign glosses, e.g. ["hello", "dad"]
        """
        if not prediction_stream:
            return []

        # Short stream — just return the top sign, no LLM needed
        if len(prediction_stream) <= 2:
            return self._fallback(prediction_stream)

        try:
            return self._llm_decode(prediction_stream, context=context)
        except Exception as e:
            print(f"[decoder] LLM failed ({e}), using fallback")
            return self._fallback(prediction_stream)

    def _format_stream(self, stream: list[dict]) -> str:
        """Format prediction stream as compact text for the LLM."""
        lines = []
        t0 = stream[0]["t"]
        for snap in stream:
            t = snap["t"] - t0
            parts = " ".join(f"{name}:{prob:.2f}" for name, prob in snap["top5"][:5])
            lines.append(f"{t:.1f}s | {parts}")
        return "\n".join(lines)

    def _llm_decode(self, stream: list[dict], context: list[str] | None = None) -> list[str]:
        user_msg = self._format_stream(stream)
        if context:
            history_str = "\n".join(f"- {s}" for s in context)
            user_msg = f"Prior conversation:\n{history_str}\n\nCurrent signs:\n{user_msg}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self._lava_token}",
            "Content-Type": "application/json",
        }
        r = requests.post(
            LAVA_BASE,
            params={"u": self._provider_url},
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        r.raise_for_status()

        raw = r.json()["choices"][0]["message"]["content"].strip()
        # Parse space-separated lowercase sign names
        signs = [re.sub(r"[^a-z]", "", w.lower()) for w in raw.split() if w.strip()]
        signs = [s for s in signs if s]

        # Validate: only accept signs that appeared in the stream
        seen = set()
        for snap in stream:
            for name, _ in snap["top5"]:
                seen.add(name.lower())
        signs = [s for s in signs if s in seen]

        return signs if signs else self._fallback(stream)

    def _fallback(self, stream: list[dict]) -> list[str]:
        """Frequency-weighted fallback: boost common signs, penalize rare ones."""
        votes = {}
        for snap in stream:
            if snap["top5"]:
                for name, prob in snap["top5"][:3]:
                    w = _freq_weight(name)
                    votes[name] = votes.get(name, 0.0) + prob * w
        if not votes:
            return []
        best = max(votes, key=votes.get)
        return [best]
