"""Shared fakes. Nothing here downloads a model or opens a network connection."""

from __future__ import annotations

import hashlib
import io
import math
import re
import struct
import wave

import numpy as np
import pytest
from chromadb.api.types import EmbeddingFunction

from soulsync.config import load_settings
from soulsync.emotion import LABELS, EmotionDetector
from soulsync.memory import MemoryStore


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding: shared words => small cosine distance."""

    DIM = 256

    def __init__(self) -> None:
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            v = np.zeros(self.DIM, dtype=np.float32)
            for word in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = np.linalg.norm(v)
            vectors.append(v / norm if norm else v)
        return vectors

    @staticmethod
    def name() -> str:
        return "test-hash"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


class FakeLLM:
    name = "fake:llm"

    def __init__(self, pieces=("I hear ", "you. ", "That sounds hard.")):
        self.pieces = list(pieces)
        self.calls: list[list[dict]] = []
        self.error: Exception | None = None

    def stream(self, messages):
        self.calls.append([dict(m) for m in messages])
        if self.error is not None:
            raise self.error
        yield from self.pieces


class FakeSTT:
    name = "fake:stt"

    def __init__(self, text="I had a rough day at work today"):
        self.text = text
        self.error: Exception | None = None

    def transcribe(self, wave, raw, filename=""):
        if self.error is not None:
            raise self.error
        return self.text


class FixedProbModel:
    def __init__(self, label="sad"):
        self.probs = np.full(len(LABELS), 0.05)
        self.probs[LABELS.index(label)] = 0.8

    def predict_proba(self, wave):
        return self.probs


def make_wav(seconds: float = 1.0, amplitude: float = 0.3, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = (
            struct.pack("<h", int(32767 * amplitude * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(int(seconds * rate))
        )
        w.writeframes(b"".join(frames))
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return load_settings({"MODE": "local", "DATA_DIR": str(tmp_path), "SECRET_KEY": "test-secret"})


@pytest.fixture
def memory():
    store = MemoryStore(None, top_k=3, max_distance=0.6, embedding_function=HashEmbedding())
    # EphemeralClient shares one in-process database, so start each test clean.
    ids = store.collection.get(include=[])["ids"]
    if ids:
        store.collection.delete(ids=ids)
    return store


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_stt():
    return FakeSTT()


@pytest.fixture
def detector():
    return EmotionDetector(FixedProbModel("sad"))


@pytest.fixture
def wav_bytes():
    return make_wav()
