"""Speech-to-text backends: local Whisper or Groq's hosted Whisper."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class STTError(RuntimeError):
    user_message = "I couldn't transcribe that recording. Please try again."


class STTRateLimited(STTError):
    user_message = "Transcription is rate-limited right now. Please wait a minute and try again."


class STT(Protocol):
    name: str

    def transcribe(self, wave: np.ndarray, raw: bytes, filename: str) -> str:
        """`wave` is 16 kHz mono float32; `raw` is the original upload."""
        ...


class FasterWhisperSTT:
    def __init__(self, model: str = "base"):
        from faster_whisper import WhisperModel

        # CTranslate2 has no MPS backend; int8 on CPU is its fast path on Apple silicon.
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.name = f"faster-whisper:{model}"

    def transcribe(self, wave: np.ndarray, raw: bytes = b"", filename: str = "") -> str:
        segments, _ = self.model.transcribe(wave, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


class OpenAIWhisperSTT:
    def __init__(self, model: str = "base"):
        import whisper

        self.model = whisper.load_model(model)
        self.name = f"openai-whisper:{model}"

    def transcribe(self, wave: np.ndarray, raw: bytes = b"", filename: str = "") -> str:
        return self.model.transcribe(wave, language="en", fp16=False)["text"].strip()


class GroqSTT:
    def __init__(self, model: str, api_key: str):
        import groq

        self._groq = groq
        self.client = groq.Groq(api_key=api_key, max_retries=1, timeout=30.0)
        self.model = model
        self.name = f"groq:{model}"

    def transcribe(self, wave: np.ndarray, raw: bytes, filename: str = "audio.webm") -> str:
        groq = self._groq
        try:
            result = self.client.audio.transcriptions.create(
                file=(filename or "audio.webm", raw),
                model=self.model,
                language="en",
                response_format="text",
            )
        except groq.RateLimitError as exc:
            log.warning("Groq STT rate limit: %s", exc)
            raise STTRateLimited(str(exc)) from exc
        except groq.APIError as exc:
            log.error("Groq STT error: %s", exc)
            raise STTError(str(exc)) from exc
        return (result if isinstance(result, str) else result.text).strip()


def build_stt(settings) -> STT:
    if settings.stt_backend == "groq":
        return GroqSTT(settings.groq_stt_model, settings.groq_api_key)
    if settings.local_stt_engine == "openai-whisper":
        return OpenAIWhisperSTT(settings.whisper_model)
    return FasterWhisperSTT(settings.whisper_model)
