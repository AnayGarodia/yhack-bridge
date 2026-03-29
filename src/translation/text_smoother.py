"""
TextSmoother — converts raw ASL token sequences into fluid English.

Uses the Lava API gateway (OpenAI-compatible) to route to any LLM.
Lava endpoint: https://api.lavapayments.com/v1/forward?u=<provider_url>

Example:
    smoother = TextSmoother(
        lava_token="sk-...",
        provider_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-mini",
    )
    print(smoother.smooth(["HELLO", "NAME", "I", "JOHN"]))
    # → "Hello, my name is John."
"""

import os
import requests

LAVA_BASE = "https://api.lavapayments.com/v1/forward"

_SYSTEM_PROMPT = """\
You convert raw ASL sign recognition output into natural spoken English.

The input comes from an automated ASL recognizer (not a human typist), so expect:
- Lowercase sign labels like: hello, thankyou, please, dad, mom, hungry, water
- Compound words: "thankyou" = thank you, "haveto" = have to, "minemy" = mine/my, \
"hesheit" = he/she/it, "weus" = we/us, "callonphone" = call on phone, \
"frenchfries" = french fries, "icecream" = ice cream, "glasswindow" = glass window
- ASL topic-comment order: topic first, then comment (e.g. "food want" = "I want food")
- No articles, auxiliaries, or copulas — ASL drops: a, the, is, are, am, do, does
- Repeated signs may indicate emphasis or recognizer noise — deduplicate naturally
- Fingerspelled words appear as single letters: "J O H N" or "j o h n" = the name John
- The sign "fine" means "I'm fine / good", "owie" means "hurt/pain"
- Signs may arrive in imperfect order due to recognition errors

Rules:
- Output exactly ONE fluent English sentence — nothing else
- Be LITERAL — only use words that are directly implied by the signs given
- Do NOT invent context, motivation, or extra meaning (no "I want...", "I need...", \
"I'm sorry..." unless those signs actually appear)
- Do NOT apologize or say "no input" — if the signs are unclear, just translate \
the words you see as literally as possible
- Keep it short. 1-5 signs = very short sentence. Just add minimal grammar.
- If input is a single sign (hello, bye, thankyou, please, yes, no, etc.), \
just output that word naturally: "hello" → "Hello." not "Hello, how are you?"
- A few signs → short sentence. "dad happy" → "Dad is happy." NOT "My dad is feeling happy today."
- "minemy name" → "My name." / "minemy please scissors" → "My scissors, please."\
"""


class TextSmoother:
    """
    Args:
        lava_token:    Lava 'Self Forward Token' (from lava.so dashboard).
        provider_url:  Full chat-completions URL of the target LLM provider.
                       Default: OpenAI gpt-4o-mini.
        model:         Model name to pass in the request body.
        timeout:       HTTP request timeout in seconds.
    """

    def __init__(
        self,
        lava_token: str,
        provider_url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-4o-mini",
        timeout: int = 10,
    ):
        self._lava_token = lava_token
        self._provider_url = provider_url
        self._model = model
        self._timeout = timeout

    def smooth(self, tokens: list[str], context: list[str] | None = None) -> str:
        """
        Convert a list of ASL tokens into a fluid English sentence.

        Args:
            tokens:  e.g. ["HELLO", "NAME", "I", "JOHN"]
            context: recent English sentences from the conversation (for grounding)
        Returns:
            A single natural English sentence, or empty string on empty input.
        """
        raw = " ".join(t.upper() for t in tokens).strip()
        if not raw:
            return ""

        user_content = raw
        if context:
            history_str = "\n".join(f"- {s}" for s in context)
            user_content = f"Prior conversation:\n{history_str}\n\nCurrent signs: {raw}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 128,
            "temperature": 0.3,
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
        return r.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    import sys

    token = os.environ.get("LAVA_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not token:
        print("Usage: LAVA_TOKEN=<key> python text_smoother.py")
        print("   or: python text_smoother.py <lava_token>")
        sys.exit(1)

    smoother = TextSmoother(lava_token=token)

    test_cases = [
        ["HELLO", "HOW", "YOU"],
        ["NAME", "I", "JOHN"],
        ["FOOD", "WANT", "I"],
        ["THANK", "YOU", "HELP"],
        ["WHERE", "BATHROOM"],
        ["PLEASE", "SLOW", "DOWN"],
    ]
    for tokens in test_cases:
        result = smoother.smooth(tokens)
        print(f"  {tokens} → {result!r}")
