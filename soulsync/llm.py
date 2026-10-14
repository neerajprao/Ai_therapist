"""LLM backends with one streaming interface: Ollama (local) and Groq (cloud)."""

from __future__ import annotations

import logging
from typing import Iterator, Protocol

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """The LLM backend failed. `user_message` is safe to show to the user."""

    user_message = (
        "I'm sorry, I couldn't put a reply together just now. "
        "Please try again in a moment."
    )


class LLMRateLimited(LLMError):
    user_message = (
        "I'm getting more requests than I can handle right now. "
        "Please wait a minute and try again."
    )


class LLMUnavailable(LLMError):
    user_message = (
        "I can't reach the language model at the moment. "
        "If you're running locally, check that Ollama is running."
    )


class LLM(Protocol):
    name: str

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class OllamaLLM:
    def __init__(self, model: str, host: str, temperature: float = 0.7, max_tokens: int = 200):
        import ollama

        self._ollama = ollama
        self.client = ollama.Client(host=host)
        self.model = model
        self.name = f"ollama:{model}"
        self.options = {"temperature": temperature, "num_predict": max_tokens}

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            for chunk in self.client.chat(model=self.model, messages=messages, stream=True, options=self.options):
                piece = chunk["message"]["content"]
                if piece:
                    yield piece
        except self._ollama.ResponseError as exc:
            log.error("Ollama error (status %s): %s", exc.status_code, exc.error)
            raise LLMError(str(exc)) from exc
        except (ConnectionError, OSError) as exc:
            log.error("Ollama unreachable: %s", exc)
            raise LLMUnavailable(str(exc)) from exc

    def check(self) -> bool:
        """True if the server is up and the model tag is installed."""
        try:
            tags = {m["model"] for m in self.client.list()["models"]}
        except (ConnectionError, OSError, self._ollama.ResponseError) as exc:
            log.warning("Ollama check failed: %s", exc)
            return False
        return self.model in tags


class GroqLLM:
    def __init__(self, model: str, api_key: str, temperature: float = 0.7, max_tokens: int = 200):
        import groq

        self._groq = groq
        self.client = groq.Groq(api_key=api_key, max_retries=1, timeout=30.0)
        self.model = model
        self.name = f"groq:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(self, messages: list[Message]) -> Iterator[str]:
        groq = self._groq
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            for chunk in stream:
                piece = chunk.choices[0].delta.content if chunk.choices else None
                if piece:
                    yield piece
        except groq.RateLimitError as exc:
            log.warning("Groq rate limit: %s", exc)
            raise LLMRateLimited(str(exc)) from exc
        except groq.APIConnectionError as exc:
            log.error("Groq unreachable: %s", exc)
            raise LLMUnavailable(str(exc)) from exc
        except groq.APIError as exc:
            log.error("Groq API error: %s", exc)
            raise LLMError(str(exc)) from exc

    def check(self) -> bool:
        return True  # no free health probe; failures surface per request


def build_llm(settings) -> LLM:
    if settings.llm_backend == "groq":
        return GroqLLM(settings.llm_model, settings.groq_api_key)
    return OllamaLLM(settings.llm_model, settings.ollama_host)
