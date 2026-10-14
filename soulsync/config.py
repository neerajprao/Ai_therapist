"""Environment-driven configuration.

Every setting has a default that depends on MODE, and any setting can be
overridden with an environment variable (or a line in `.env`).
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent

# Defaults that differ between the two modes. Explicit env vars always win.
_MODE_DEFAULTS = {
    "local": {
        "LLM_BACKEND": "ollama",
        "LLM_MODEL": "qwen2.5:7b-instruct",
        "STT_BACKEND": "local",
        "EMOTION_BACKEND": "torch",
        # Chosen on the validation split by eval/ensemble_eval.py.
        "ENABLE_EMOTION2VEC": "true",
        "ENSEMBLE_W2V_WEIGHT": "0.4",
    },
    "cloud": {
        "LLM_BACKEND": "groq",
        # Groq's only Qwen chat model (preview) when checked on 2026-10-13.
        # Production fallback: llama-3.1-8b-instant.
        "LLM_MODEL": "qwen/qwen3.8-27b",
        "STT_BACKEND": "groq",
        "EMOTION_BACKEND": "onnx",
        # Emotion2Vec needs PyTorch and 1.9 GB of weights: not in the cloud image.
        "ENABLE_EMOTION2VEC": "false",
        "ENSEMBLE_W2V_WEIGHT": "1.0",
    },
}

_CHOICES = {
    "MODE": {"local", "cloud"},
    "LLM_BACKEND": {"ollama", "groq"},
    "STT_BACKEND": {"local", "groq"},
    "EMOTION_BACKEND": {"torch", "onnx"},
    "TTS_BACKEND": {"browser", "say"},
    "LOCAL_STT_ENGINE": {"faster-whisper", "openai-whisper"},
}


class ConfigError(ValueError):
    """Raised when the environment describes an invalid configuration."""


def _bool(value: str, name: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class Settings:
    mode: str
    llm_backend: str
    llm_model: str
    stt_backend: str
    local_stt_engine: str
    whisper_model: str
    groq_stt_model: str
    emotion_backend: str
    emotion_model_dir: Path
    emotion_onnx_path: Path
    enable_emotion2vec: bool
    emotion2vec_dir: Path
    ensemble_w2v_weight: float
    silence_rms: float
    memory_enabled: bool
    memory_top_k: int
    memory_max_distance: float
    history_max_turns: int
    tts_backend: str
    data_dir: Path
    secret_key: str
    groq_api_key: str
    ollama_host: str
    cookie_secure: bool
    max_upload_mb: int
    log_level: str

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def data_flow(self) -> dict:
        """What leaves the device in this configuration. Shown in the UI."""
        external = []
        if self.stt_backend == "groq":
            external.append("your voice recording (to Groq, for transcription)")
        if self.llm_backend == "groq":
            external.append("your transcribed messages and recalled memories (to Groq, for the reply)")
        return {
            "mode": self.mode,
            "leaves_device": external,
            "stored_on_server": (
                ["text of your past messages, for memory (you can delete it)"]
                if self.memory_enabled
                else []
            ),
            "never_stored": ["voice recordings"],
        }


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (defaults to os.environ, after reading .env)."""
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except ImportError:  # python-dotenv is optional
            pass
        env = os.environ

    mode = env.get("MODE", "local").strip().lower()
    if mode not in _CHOICES["MODE"]:
        raise ConfigError(f"MODE must be one of {sorted(_CHOICES['MODE'])}, got {mode!r}")
    defaults = _MODE_DEFAULTS[mode]

    def get(name: str, default: str) -> str:
        value = env.get(name, "").strip()
        return value if value else defaults.get(name, default)

    def choice(name: str, default: str) -> str:
        value = get(name, default).lower()
        if value not in _CHOICES[name]:
            raise ConfigError(f"{name} must be one of {sorted(_CHOICES[name])}, got {value!r}")
        return value

    def path(name: str, default: str) -> Path:
        p = Path(get(name, default)).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    llm_backend = choice("LLM_BACKEND", "ollama")
    stt_backend = choice("STT_BACKEND", "local")
    llm_model = get("LLM_MODEL", "")
    groq_api_key = env.get("GROQ_API_KEY", "").strip()

    if not llm_model:
        raise ConfigError(f"LLM_MODEL is required when LLM_BACKEND={llm_backend}")
    if "groq" in (llm_backend, stt_backend) and not groq_api_key:
        raise ConfigError("GROQ_API_KEY is required when a Groq backend is selected")

    data_dir = path("DATA_DIR", "data")
    secret_key = env.get("SECRET_KEY", "").strip()
    if not secret_key:
        if mode == "cloud":
            raise ConfigError("SECRET_KEY is required in cloud mode")
        secret_key = _local_secret(data_dir)

    try:
        settings = Settings(
            mode=mode,
            llm_backend=llm_backend,
            llm_model=llm_model,
            stt_backend=stt_backend,
            local_stt_engine=choice("LOCAL_STT_ENGINE", "openai-whisper"),
            whisper_model=get("WHISPER_MODEL", "base"),
            groq_stt_model=get("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            emotion_backend=choice("EMOTION_BACKEND", "torch"),
            emotion_model_dir=path("EMOTION_MODEL_DIR", "models/emotion_w2v2"),
            emotion_onnx_path=path("EMOTION_ONNX_PATH", "models/emotion_w2v2_int8.onnx"),
            enable_emotion2vec=_bool(get("ENABLE_EMOTION2VEC", "false"), "ENABLE_EMOTION2VEC"),
            emotion2vec_dir=path("EMOTION2VEC_DIR", "models/emotion2vec"),
            ensemble_w2v_weight=float(get("ENSEMBLE_W2V_WEIGHT", "1.0")),
            silence_rms=float(get("SILENCE_RMS", "0.008")),
            memory_enabled=_bool(get("MEMORY_ENABLED", "true"), "MEMORY_ENABLED"),
            memory_top_k=int(get("MEMORY_TOP_K", "3")),
            memory_max_distance=float(get("MEMORY_MAX_DISTANCE", "0.7")),
            history_max_turns=int(get("HISTORY_MAX_TURNS", "8")),
            tts_backend=choice("TTS_BACKEND", "browser"),
            data_dir=data_dir,
            secret_key=secret_key,
            groq_api_key=groq_api_key,
            ollama_host=get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            cookie_secure=_bool(get("COOKIE_SECURE", "true" if mode == "cloud" else "false"), "COOKIE_SECURE"),
            max_upload_mb=int(get("MAX_UPLOAD_MB", "10")),
            log_level=get("LOG_LEVEL", "INFO").upper(),
        )
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"Invalid numeric setting: {exc}") from exc

    if not 0.0 <= settings.ensemble_w2v_weight <= 1.0:
        raise ConfigError("ENSEMBLE_W2V_WEIGHT must be between 0 and 1")
    if settings.memory_top_k < 1 or settings.history_max_turns < 1:
        raise ConfigError("MEMORY_TOP_K and HISTORY_MAX_TURNS must be at least 1")
    return settings


def _local_secret(data_dir: Path) -> str:
    """Create (once) and reuse a signing key so cookies survive restarts locally."""
    key_file = data_dir / "secret.key"
    if key_file.exists():
        return key_file.read_text().strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    key_file.write_text(key)
    key_file.chmod(0o600)
    return key


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Transcripts are sensitive: libraries that log request bodies stay quiet.
    for noisy in ("httpx", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
