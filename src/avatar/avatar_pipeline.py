"""
Avatar pipeline — speech → ASL signs → animated display.

Receives individual words from STT, converts to ASL glosses,
and feeds them to the hand renderer one at a time.

Designed for real-time speed:
  - Max queue depth: 2 words. Drops oldest if more arrive.
  - 0.6 seconds per sign (18 frames at 30fps).
  - Consecutive duplicate words are collapsed.
  - Long fingerspelled words (>4 letters) are skipped during fast speech.
"""

import collections
import re
import threading
import time


class AvatarPipeline:
    """
    Connects STT word stream → ASL gloss conversion → hand renderer.

    Usage:
        pipeline = AvatarPipeline(english_to_signs)
        pipeline.on_word("hello")  # called by STT for each word
        sign = pipeline.next_sign()  # called by main loop to get next sign to render
    """

    MAX_QUEUE = 2

    def __init__(self, english_to_signs=None):
        self._e2s = english_to_signs
        self._queue = collections.deque(maxlen=self.MAX_QUEUE)
        self._lock = threading.Lock()
        self._last_sign = None
        self._word_count = 0
        self._word_times = collections.deque(maxlen=30)  # for WPM calculation

        # Simple word → ASL sign mapping (fast, no API call)
        # For single words, we use direct lookup instead of LLM
        self._quick_map = {
            "hello": "HELLO", "hi": "HELLO", "hey": "HELLO",
            "thanks": "THANK-YOU", "thank": "THANK-YOU",
            "please": "PLEASE", "yes": "YES", "yeah": "YES",
            "no": "NO", "nope": "NO", "help": "HELP",
            "sorry": "SORRY", "eat": "EAT", "eating": "EAT",
            "drink": "DRINK", "drinking": "DRINK", "water": "WATER",
            "mom": "MOM", "mother": "MOM", "dad": "DAD", "father": "DAD",
            "happy": "HAPPY", "sad": "SAD", "good": "GOOD", "bad": "BAD",
            "stop": "STOP", "go": "GO", "going": "GO",
            "like": "LIKE", "want": "WANT", "more": "MORE",
            "finish": "FINISH", "done": "FINISH", "name": "NAME",
            "what": "WHAT", "how": "HOW", "nice": "NICE",
            "meet": "MEET", "you": "YOU", "your": "YOUR",
            "i": "I", "me": "ME", "my": "MY",
            "he": "HE", "she": "SHE", "they": "THEY", "we": "WE",
            "love": "LOVE", "family": "FAMILY", "friend": "FRIEND",
            "work": "WORK", "school": "SCHOOL", "home": "HOME",
            "today": "TODAY", "tomorrow": "TOMORROW", "hungry": "HUNGRY",
            "tired": "TIRED", "sick": "SICK", "not": "NOT",
            "understand": "UNDERSTAND", "where": "WHERE", "who": "WHO",
            "why": "WHY", "when": "WHEN", "here": "HERE", "there": "THERE",
            "now": "NOW", "can": "CAN", "will": "WILL",
            "think": "THINK", "know": "KNOW", "see": "SEE",
            "look": "LOOK", "come": "COME", "open": "OPEN",
            "close": "CLOSE", "big": "BIG", "small": "SMALL",
            "hot": "HOT", "cold": "COLD", "new": "NEW", "old": "OLD",
        }

        # Words to skip entirely (articles, auxiliaries, fillers)
        self._skip_words = {
            "a", "an", "the", "is", "are", "am", "was", "were", "be",
            "been", "being", "do", "does", "did", "to", "of", "and",
            "or", "but", "just", "very", "really", "quite", "actually",
            "um", "uh", "like", "so", "well",
        }

    def on_word(self, word: str):
        """Called by STT for each transcribed word."""
        if not word or not word.strip():
            return

        clean = word.strip().lower()
        clean = re.sub(r"[^a-z']", "", clean)
        if not clean:
            return

        # Skip filler/article words
        if clean in self._skip_words:
            return

        # Convert to ASL sign
        sign = self._quick_map.get(clean)
        if sign is None:
            # Short unknown words: fingerspell
            if len(clean) <= 4:
                sign = "-".join(clean.upper())
            else:
                # Long unknown words during fast speech: skip
                # (fingerspelling "restaurant" takes too long)
                return

        with self._lock:
            # Dedup: don't queue same sign consecutively
            if self._queue and self._queue[-1] == sign:
                return
            if self._last_sign == sign:
                return

            # Drop oldest if queue is full (keep it real-time)
            if len(self._queue) >= self.MAX_QUEUE:
                self._queue.popleft()

            self._queue.append(sign)
            self._word_count += 1
            self._word_times.append(time.monotonic())

    def next_sign(self):
        """
        Pop the next sign to render. Returns None if queue is empty.
        Called by the main loop when the renderer finishes the current sign.
        """
        with self._lock:
            if not self._queue:
                return None
            sign = self._queue.popleft()
            self._last_sign = sign
            return sign

    def clear(self):
        """Clear pending signs (called when new words arrive and old ones are stale)."""
        with self._lock:
            self._queue.clear()

    @property
    def queue_depth(self):
        with self._lock:
            return len(self._queue)

    @property
    def words_per_minute(self):
        """Estimate current WPM from recent word emission times."""
        with self._lock:
            if len(self._word_times) < 2:
                return 0.0
            span = self._word_times[-1] - self._word_times[0]
            if span < 0.1:
                return 0.0
            return (len(self._word_times) - 1) / span * 60.0

    @property
    def total_words(self):
        return self._word_count
