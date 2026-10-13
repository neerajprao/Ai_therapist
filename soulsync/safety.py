"""Crisis-language detection.

A deliberately simple, auditable keyword/regex layer. When it fires, the LLM is
skipped and a fixed message with helpline information is returned. It favours
false alarms over misses: "I would never hurt myself" still triggers.

It is not a clinical risk assessment and will miss indirect or coded language.
"""

from __future__ import annotations

import re
import unicodedata

# Helplines shown in the crisis reply (India). Keep these in one place.
HELPLINE_NAME = "Tele-MANAS"
HELPLINE_NUMBERS = "14416 or 1-800-891-4416"
EMERGENCY_NUMBER = "112"

CRISIS_MESSAGE = (
    "I'm really glad you told me, and I'm concerned about your safety. "
    "I'm an AI demo and can't give you the help you deserve right now, but people can. "
    f"In India you can call {HELPLINE_NAME} on {HELPLINE_NUMBERS}, free and open 24 hours a day. "
    f"If you are in immediate danger, please call {EMERGENCY_NUMBER} or go to the nearest emergency room. "
    "If you can, reach out to someone you trust and let them stay with you. You don't have to face this alone."
)

_CONTRACTIONS = [
    (r"\bi'?m\b", "i am"),
    (r"\bi'?ve\b", "i have"),
    (r"\bi'?d\b", "i would"),
    (r"\bi'?ll\b", "i will"),
    (r"\bcan'?t\b", "can not"),
    (r"\bcannot\b", "can not"),
    (r"\bwon'?t\b", "will not"),
    (r"\bdon'?t\b", "do not"),
    (r"\bdoesn'?t\b", "does not"),
    (r"\bdidn'?t\b", "did not"),
    (r"\bisn'?t\b", "is not"),
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
    (r"\bkms\b", "kill myself"),
]

_PATTERNS = [
    r"\bkill(ing)? myself\b",
    r"\bsuicid(e|al)\b",
    r"\bend(ing)? (my|this) (own )?life\b",
    r"\bend it all\b",
    r"\btak(e|ing) my (own )?life\b",
    r"\b(want|wish|going|plan|planning|ready|deserve) to die\b",
    r"\bwish i (was|were|am) dead\b",
    r"\bwant to be dead\b",
    r"\bbetter off (dead|without me)\b",
    r"\b(hurt|harm)(ing)? myself\b",
    r"\bcutting myself\b",
    r"\bcut myself\b(?! (while |when )?(shaving|cooking|chopping)| by accident| accidentally)",
    r"\bself harm(ing)?\b",
    r"\b(no|not any|nothing) (reason|point) (to|in) (live|living|go on|going on)\b",
    r"\bdo not want to (live|be alive|be here anymore|exist|wake up)\b",
    r"\b(life is )?not worth living\b",
    r"\bcan not go on (living|anymore|any longer|like this)\b",
    r"\bhang(ing)? myself\b",
    r"\b(shoot|drown|poison)(ing)? myself\b",
    r"\boverdos(e|ing)\b",
    r"\b(take|swallow) (all )?(my|the|these) pills\b",
    r"\bjump (off|from|in front of) (a|the|my) (bridge|building|roof|terrace|balcony|train)\b",
    r"\bnot be (here|around) (tomorrow|much longer|anymore)\b",
]
_CRISIS_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))


def normalize(text: str) -> str:
    """Lowercase, straighten quotes, expand contractions, drop punctuation."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    for pattern, repl in _CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_crisis(text: str) -> bool:
    return bool(_CRISIS_RE.search(normalize(text)))
