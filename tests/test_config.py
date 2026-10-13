import pytest

from soulsync.config import ConfigError, load_settings


def base(tmp_path, **extra):
    return {"DATA_DIR": str(tmp_path), **extra}


def test_local_defaults(tmp_path):
    s = load_settings(base(tmp_path))
    assert (s.mode, s.llm_backend, s.stt_backend, s.emotion_backend) == ("local", "ollama", "local", "torch")
    assert s.llm_model == "qwen2.5:7b-instruct"
    assert s.tts_backend == "browser"
    assert s.enable_emotion2vec is False
    assert s.memory_enabled is True
    assert s.data_flow()["leaves_device"] == []


def test_cloud_mode_switches_backends(tmp_path):
    s = load_settings(
        base(tmp_path, MODE="cloud", GROQ_API_KEY="k", SECRET_KEY="s", LLM_MODEL="some-groq-model")
    )
    assert (s.llm_backend, s.stt_backend, s.emotion_backend) == ("groq", "groq", "onnx")
    assert s.cookie_secure is True
    assert len(s.data_flow()["leaves_device"]) == 2


def test_explicit_env_overrides_mode_defaults(tmp_path):
    s = load_settings(base(tmp_path, EMOTION_BACKEND="onnx", LLM_MODEL="qwen2.5:3b-instruct", MEMORY_ENABLED="false"))
    assert s.emotion_backend == "onnx"
    assert s.llm_model == "qwen2.5:3b-instruct"
    assert s.memory_enabled is False
    assert s.data_flow()["stored_on_server"] == []


def test_mixed_backends_report_what_leaves_the_device(tmp_path):
    s = load_settings(base(tmp_path, STT_BACKEND="groq", GROQ_API_KEY="k"))
    flow = s.data_flow()["leaves_device"]
    assert len(flow) == 1 and "voice recording" in flow[0]


@pytest.mark.parametrize(
    "env",
    [
        {"MODE": "edge"},
        {"LLM_BACKEND": "openai"},
        {"EMOTION_BACKEND": "tflite"},
        {"MEMORY_ENABLED": "maybe"},
        {"MEMORY_TOP_K": "three"},
        {"MEMORY_TOP_K": "0"},
        {"ENSEMBLE_W2V_WEIGHT": "1.5"},
        {"LLM_BACKEND": "groq", "LLM_MODEL": "m"},  # no GROQ_API_KEY
        {"MODE": "cloud", "GROQ_API_KEY": "k", "LLM_MODEL": "m"},  # no SECRET_KEY
        {"MODE": "cloud", "GROQ_API_KEY": "k", "SECRET_KEY": "s"},  # no LLM_MODEL
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, env):
    with pytest.raises(ConfigError):
        load_settings(base(tmp_path, **env))


def test_local_secret_is_created_once_and_reused(tmp_path):
    first = load_settings(base(tmp_path)).secret_key
    assert (tmp_path / "secret.key").exists()
    assert load_settings(base(tmp_path)).secret_key == first
