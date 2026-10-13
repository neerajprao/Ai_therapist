"""One conversational turn: safety check, memory recall, bounded history, LLM stream."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Iterator

from . import safety
from .llm import LLM, LLMError, Message
from .memory import MemoryStore, format_memories

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Samantha, a calm, warm AI companion for supportive conversation. "
    "You are not a therapist or a doctor: never claim to be one, never diagnose, "
    "and never give medical advice. "
    "User messages arrive as '[User Tone: <Emotion>] <Message>', where the tone was "
    "estimated from their voice and may be wrong. Let it shape your empathy, but never "
    "mention the tag. "
    "Keep replies to 2-3 short sentences. Speak only words, with no stage directions. "
    "Focus on: 1. reflecting what the person said, 2. naming the feeling behind it, "
    "3. gently asking what they need right now."
)

EMPTY_INPUT_REPLY = (
    "I didn't catch anything that time. Take your time, and tap the microphone "
    "again whenever you're ready."
)


class SessionStore:
    """Chat history for live sessions, kept in memory only and bounded in size.

    Keyed by (user_id, session_id) so one user can never read another's turns.
    """

    def __init__(self, max_sessions: int = 500, max_messages: int = 200):
        self._data: OrderedDict[tuple[str, str], list[Message]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.max_messages = max_messages

    def get(self, user_id: str, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._data.get((user_id, session_id), []))

    def append(self, user_id: str, session_id: str, user_content: str, reply: str) -> None:
        key = (user_id, session_id)
        with self._lock:
            history = self._data.setdefault(key, [])
            self._data.move_to_end(key)
            history += [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": reply},
            ]
            del history[: -self.max_messages]
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def forget_user(self, user_id: str) -> None:
        with self._lock:
            for key in [k for k in self._data if k[0] == user_id]:
                del self._data[key]


def tag_user_message(text: str, emotion: str) -> str:
    return f"[User Tone: {emotion}] {text}" if emotion and emotion != "Unavailable" else text


def build_messages(
    history: list[Message],
    user_content: str,
    memory_note: str | None,
    max_turns: int,
) -> list[Message]:
    """System prompt + last `max_turns` exchanges + (temporary memory note) + new message.

    The memory note is built fresh each turn and is never written to history.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-2 * max_turns :]
    if memory_note:
        messages.append({"role": "system", "content": memory_note})
    messages.append({"role": "user", "content": user_content})
    return messages


class ActionFilter:
    """Drops *stage directions* and [bracketed asides], even when split across chunks."""

    def __init__(self) -> None:
        self._in_star = False
        self._in_bracket = False

    def feed(self, piece: str) -> str:
        out = []
        for ch in piece:
            if self._in_bracket:
                if ch == "]":
                    self._in_bracket = False
            elif ch == "[":
                self._in_bracket = True
            elif ch == "*":
                self._in_star = not self._in_star
            elif not self._in_star:
                out.append(ch)
        return "".join(out)


class ChatEngine:
    def __init__(
        self,
        llm: LLM,
        memory: MemoryStore | None,
        sessions: SessionStore | None = None,
        max_turns: int = 8,
    ):
        self.llm = llm
        self.memory = memory
        self.sessions = sessions or SessionStore()
        self.max_turns = max_turns

    def recall(self, user_id: str, session_id: str, text: str) -> str | None:
        if self.memory is None:
            return None
        memories = self.memory.search(user_id, session_id, text)
        return format_memories(memories) if memories else None

    def respond(self, user_id: str, session_id: str, text: str, emotion: str) -> Iterator[str]:
        """Yield the reply in pieces. `emotion` is the display label, e.g. 'Sad'."""
        text = text.strip()
        if not text:
            yield EMPTY_INPUT_REPLY
            return

        user_content = tag_user_message(text, emotion)

        if safety.is_crisis(text):
            log.warning("Crisis language detected; LLM skipped for this turn.")
            yield safety.CRISIS_MESSAGE
            # Kept in the live session for context, but never written to long-term memory.
            self.sessions.append(user_id, session_id, user_content, safety.CRISIS_MESSAGE)
            return

        memory_note = self.recall(user_id, session_id, text)
        messages = build_messages(
            self.sessions.get(user_id, session_id), user_content, memory_note, self.max_turns
        )

        cleaner = ActionFilter()
        reply = ""
        try:
            for piece in self.llm.stream(messages):
                piece = cleaner.feed(piece)
                if piece:
                    reply += piece
                    yield piece
        except LLMError as exc:
            # Tell the user plainly; do not save a half reply as if it were real.
            yield ("" if not reply else " ... ") + exc.user_message
            return

        reply = reply.strip()
        if not reply:
            return
        self.sessions.append(user_id, session_id, user_content, reply)
        if self.memory is not None:
            self.memory.add(user_id, session_id, text, emotion=emotion.lower(), reply=reply)
