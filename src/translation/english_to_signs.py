"""
EnglishToSigns — converts English sentences to ordered ASL gloss lists.

Uses Nous Research Hermes 3 (via Lava API gateway) to handle:
  - ASL topic-comment reordering
  - Article/auxiliary dropping
  - Common English → ASL sign name mapping
  - Proper noun fingerspelling (John → J-O-H-N)

Falls back to a simple rule-based splitter if the API is unavailable.

Example:
    converter = EnglishToSigns(lava_token="aks_live_...")
    converter.convert("Hello, my name is John, nice to meet you")
    # → ["HELLO", "NAME", "MY", "J-O-H-N", "NICE", "MEET", "YOU"]

Install: no extra deps (uses requests, already installed)
"""

import os
import re
import sys

import requests

LAVA_BASE             = "https://api.lavapayments.com/v1/forward"
_DEFAULT_PROVIDER_URL = os.environ.get("LAVA_PROVIDER_URL", "https://api.openai.com/v1/chat/completions")
_DEFAULT_MODEL        = os.environ.get("E2S_MODEL", "gpt-4o-mini")

_SYSTEM_PROMPT = """\
You are an ASL (American Sign Language) gloss converter.
Convert English sentences into an ordered list of ASL sign glosses.

Rules:
- Drop articles: a, an, the
- Drop auxiliary verbs: is, are, am, was, were, be, been, being, do, does, did
- Drop filler words: just, very, really, quite, actually
- Use ASL topic-comment order when natural (topic first, then comment)
- Map contractions: don't → NOT, can't → CAN NOT, I'm → I, it's → IT
- Map common words to their standard ASL gloss names:
    hello/hi → HELLO, thank you/thanks → THANK-YOU, please → PLEASE,
    yes → YES, no → NO, help → HELP, sorry → SORRY, eat/eating → EAT,
    drink/drinking → DRINK, water → WATER, mom/mother → MOTHER,
    dad/father → FATHER, happy → HAPPY, sad → SAD, good → GOOD,
    bad → BAD, stop → STOP, go/going → GO, like/like → LIKE,
    want/want → WANT, more → MORE, finish/done/finished → FINISH,
    name → NAME, what → WHAT, how → HOW, nice → NICE, meet → MEET,
    understand → UNDERSTAND, repeat → REPEAT, slow → SLOW, fast → FAST,
    bathroom/restroom → BATHROOM, hungry → HUNGRY, tired → TIRED,
    sick → SICK, love → LOVE, family → FAMILY, friend → FRIEND,
    work → WORK, school → SCHOOL, home → HOME, today → TODAY,
    tomorrow → TOMORROW, yesterday → YESTERDAY
- Fingerspell proper nouns, names, and unknown words letter-by-letter with dashes:
    John → J-O-H-N, Sara → S-A-R-A, Bridge → B-R-I-D-G-E
- Output ONLY the gloss list as space-separated words on a single line.
  No explanation, no punctuation, no brackets.

Examples:
  Input:  Hello, my name is John.
  Output: HELLO NAME MY J-O-H-N

  Input:  Nice to meet you. Do you want water?
  Output: NICE MEET YOU WANT WATER YOU

  Input:  I am hungry and tired.
  Output: HUNGRY TIRED I

  Input:  Can you please help me?
  Output: HELP PLEASE YOU
"""

# ---------------------------------------------------------------------------
# Fallback: simple rule-based converter (no API)
# ---------------------------------------------------------------------------

_DROP_WORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
    "being", "do", "does", "did", "to", "of", "and", "or", "but",
    "just", "very", "really", "quite", "actually",
}

_WORD_MAP = {
    "hello": "HELLO", "hi": "HELLO", "hey": "HELLO",
    "thanks": "THANK-YOU", "thank": "THANK-YOU", "thankyou": "THANK-YOU",
    "much": "MUCH", "very": "VERY",
    "please": "PLEASE", "yes": "YES", "yeah": "YES",
    "no": "NO", "nope": "NO", "nah": "NO",
    "help": "HELP", "sorry": "SORRY",
    "eat": "EAT", "eating": "EAT", "food": "FOOD",
    "drink": "DRINK", "drinking": "DRINK", "water": "WATER",
    "mom": "MOTHER", "mother": "MOTHER", "dad": "FATHER", "father": "FATHER",
    "happy": "HAPPY", "sad": "SAD", "good": "GOOD", "bad": "BAD",
    "stop": "STOP", "go": "GO", "going": "GO",
    "like": "LIKE", "want": "WANT", "more": "MORE",
    "finish": "FINISH", "finished": "FINISH", "done": "FINISH",
    "name": "NAME", "what": "WHAT", "how": "HOW",
    "nice": "NICE", "meet": "MEET",
    "understand": "UNDERSTAND", "repeat": "REPEAT",
    "slow": "SLOW", "fast": "FAST",
    "bathroom": "BATHROOM", "restroom": "BATHROOM",
    "hungry": "HUNGRY", "tired": "TIRED", "sick": "SICK",
    "love": "LOVE", "family": "FAMILY", "friend": "FRIEND",
    "work": "WORK", "school": "SCHOOL", "home": "HOME",
    "today": "TODAY", "tomorrow": "TOMORROW", "yesterday": "YESTERDAY",
    "i": "I", "me": "ME", "my": "MY", "mine": "MY",
    "you": "YOU", "your": "YOUR",
    "he": "HE", "she": "SHE", "they": "THEY", "we": "WE",
    "not": "NOT", "dont": "NOT", "cant": "CAN NOT",
}


def _fingerspell(word: str) -> str:
    """Converts a word to fingerspelled gloss: John → J-O-H-N"""
    return "-".join(word.upper())


def _rule_based_convert(text: str) -> list[str]:
    """Simple fallback: drop articles, map known words, fingerspell unknowns."""
    # Normalize: lowercase, strip punctuation
    text = text.lower()
    text = re.sub(r"[''']", "", text)           # contractions: don't → dont
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = text.split()

    glosses = []
    for word in words:
        if word in _DROP_WORDS:
            continue
        if word in _WORD_MAP:
            glosses.append(_WORD_MAP[word])
        else:
            glosses.append(_fingerspell(word))
    return glosses


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EnglishToSigns:
    """
    Args:
        lava_token:  Lava Self Forward Token.
        timeout:     HTTP request timeout in seconds.
    """

    def __init__(self, lava_token: str, timeout: int = 10):
        self._lava_token    = lava_token
        self._timeout       = timeout
        self._provider_url  = _DEFAULT_PROVIDER_URL
        self._model         = _DEFAULT_MODEL

    def convert(self, english_text: str) -> list[str]:
        """
        Convert an English sentence to an ordered ASL gloss list.

        Args:
            english_text: e.g. "Hello, my name is John, nice to meet you"
        Returns:
            e.g. ["HELLO", "NAME", "MY", "J-O-H-N", "NICE", "MEET", "YOU"]
        """
        if not english_text.strip():
            return []

        try:
            return self._llm_convert(english_text)
        except Exception as e:
            print(f"[e2s] LLM failed ({e}), using rule-based fallback")
            return _rule_based_convert(english_text)

    def _llm_convert(self, text: str) -> list[str]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            "max_tokens": 128,
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self._lava_token}",
            "Content-Type":  "application/json",
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
        # Parse space-separated glosses, upper-case, strip stray punctuation
        glosses = [
            re.sub(r"[^A-Z0-9\-]", "", g.upper())
            for g in raw.split()
            if g.strip()
        ]
        return [g for g in glosses if g]


if __name__ == "__main__":
    token = os.environ.get("LAVA_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not token:
        # Load from .env if present
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("LAVA_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break

    if not token:
        print("Usage: LAVA_TOKEN=<key> python english_to_signs.py")
        sys.exit(1)

    converter = EnglishToSigns(lava_token=token)
    test_sentences = [
        "Hello, my name is John, nice to meet you.",
        "Can you please help me? I am hungry.",
        "Do you want water or food?",
        "I don't understand, can you repeat that slowly?",
        "My friend Sara works at the school.",
        "Thank you, I am happy to be here today.",
    ]

    print(f"Model: {converter._model} via Lava\n")
    for sentence in test_sentences:
        glosses = converter.convert(sentence)
        print(f"  IN:  {sentence}")
        print(f"  OUT: {glosses}")
        print()
