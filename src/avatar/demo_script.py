"""
Demo Q&A Script -- hardcoded for pitch demo reliability.

These phrases are detected by STT and mapped directly to sign sequences,
bypassing the normal word lookup.  Guarantees the demo works even if
general recognition is imperfect.
"""

DEMO_QA = [
    # -- GREETINGS --
    {
        "triggers": ["hello", "hi", "hey", "good morning", "good afternoon"],
        "signs": ["HELLO", "NICE", "MEET", "YOU"],
        "display": "Hello! Nice to meet you.",
    },
    # -- INTRO / WHAT IS THIS --
    {
        "triggers": [
            "what is this", "what does this do", "explain this",
            "tell me about this", "what is bridge", "what are you building",
        ],
        "signs": ["THIS", "HELP", "DEAF", "PEOPLE", "COMMUNICATE"],
        "display": "This app helps deaf people communicate.",
    },
    # -- HOW DOES IT WORK --
    {
        "triggers": [
            "how does it work", "how does this work", "explain how",
            "how does the technology work", "what is the technology",
        ],
        "signs": ["CAMERA", "WATCH", "HANDS", "COMPUTER", "UNDERSTAND"],
        "display": "Camera watches hands. Computer understands words.",
    },
    # -- WHO IS IT FOR --
    {
        "triggers": [
            "who is this for", "who are your users", "target audience",
            "who uses this", "who are your customers",
        ],
        "signs": ["DEAF", "PEOPLE", "FAMILY"],
        "display": "Deaf people and families.",
    },
    # -- MARKET SIZE --
    {
        "triggers": [
            "market size", "how big is the market", "total addressable market",
            "tam", "how many people", "how large is this",
        ],
        "signs": ["MILLION", "PEOPLE", "DEAF", "WORLD"],
        "display": "70 million deaf people worldwide.",
    },
    # -- WHAT PROBLEM --
    {
        "triggers": [
            "what problem", "what pain point", "what are you solving",
            "what is the problem", "why does this matter",
        ],
        "signs": ["DEAF", "PEOPLE", "CANNOT", "COMMUNICATE"],
        "display": "Deaf people struggle to communicate easily.",
    },
    # -- FUNDING / INVESTMENT --
    {
        "triggers": [
            "funding", "investment", "how much", "raise", "money",
            "are you raising", "investment ask",
        ],
        "signs": ["WE", "NEED", "HELP", "GROW"],
        "display": "We need help to grow.",
    },
    # -- DEMO --
    {
        "triggers": [
            "show me", "demonstrate", "can you show", "give me a demo",
            "demo", "show how it works",
        ],
        "signs": ["WATCH", "THIS", "THANK-YOU"],
        "display": "Watch this! Thank you.",
    },
    # -- THANK YOU --
    {
        "triggers": [
            "thank you", "thanks", "thank", "appreciate",
            "great", "amazing", "impressive", "wonderful",
        ],
        "signs": ["THANK-YOU", "VERY", "MUCH"],
        "display": "Thank you very much!",
    },
    # -- GOODBYE --
    {
        "triggers": [
            "goodbye", "bye", "see you", "good luck", "good night",
            "farewell", "take care",
        ],
        "signs": ["GOODBYE", "NICE", "MEET", "YOU"],
        "display": "Goodbye! Nice to meet you.",
    },
    # -- YES --
    {
        "triggers": ["yes", "correct", "exactly", "right", "absolutely", "indeed"],
        "signs": ["YES"],
        "display": "Yes!",
    },
    # -- NO --
    {
        "triggers": ["no", "not", "incorrect", "wrong", "negative"],
        "signs": ["NO"],
        "display": "No.",
    },
    # -- HELP --
    {
        "triggers": ["help", "need help", "can you help", "assist"],
        "signs": ["HELP", "YOU", "NEED"],
        "display": "Help! What do you need?",
    },
    # -- WATER / FOOD --
    {
        "triggers": ["water", "drink", "thirsty"],
        "signs": ["WATER", "PLEASE"],
        "display": "Water please.",
    },
    {
        "triggers": ["eat", "food", "hungry", "lunch", "dinner"],
        "signs": ["EAT", "FOOD", "PLEASE"],
        "display": "Food please.",
    },
    # -- COMPETITION SPECIFIC --
    {
        "triggers": [
            "what stage are you", "stage", "how far along",
            "traction", "progress", "users",
        ],
        "signs": ["WE", "TEST", "HELP", "CHILDREN"],
        "display": "We tested at hospitals to help children.",
    },
    {
        "triggers": [
            "competition", "competitors", "who else", "similar",
            "what else is out there",
        ],
        "signs": ["WE", "DIFFERENT", "REAL", "TIME"],
        "display": "We're different -- real-time and smart.",
    },
]


def find_demo_response(text: str):
    """
    Check if input text matches any demo trigger phrase.
    Returns (signs_list, display_text) or None if no match.
    Case-insensitive. Longer triggers are checked first to avoid
    false positives (e.g., "hi" matching inside "this").
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return None

    # Build (trigger, entry) pairs sorted by trigger length descending
    # so "what is this" matches before "hi"
    candidates = []
    for entry in DEMO_QA:
        for trigger in entry["triggers"]:
            candidates.append((trigger.lower(), entry))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    text_words = set(text_lower.split())

    for trigger, entry in candidates:
        # Multi-word trigger: check if trigger appears as substring in text
        if " " in trigger:
            if trigger in text_lower:
                return entry["signs"], entry["display"]
        else:
            # Single-word trigger: must match a whole word in the text
            if trigger in text_words:
                return entry["signs"], entry["display"]
            # Also check if the entire text IS the trigger
            if text_lower == trigger:
                return entry["signs"], entry["display"]

    return None
