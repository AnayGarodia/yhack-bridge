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

_SYSTEM_PROMPT = (
    "You translate raw ASL sign sequences into natural spoken English. "
    "ASL is telegraphic: it omits articles, auxiliaries, and copulas, and uses "
    "topic-comment word order (e.g. 'FOOD WANT I' means 'I want food'). "
    "Output exactly one fluent English sentence — nothing else. "
    "Keep it concise. Do not add information that wasn't in the input."
)


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

    def smooth(self, tokens: list[str]) -> str:
        """
        Convert a list of ASL tokens into a fluid English sentence.

        Args:
            tokens: e.g. ["HELLO", "NAME", "I", "JOHN"]
        Returns:
            A single natural English sentence, or empty string on empty input.
        """
        raw = " ".join(t.upper() for t in tokens).strip()
        if not raw:
            return ""

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw},
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
