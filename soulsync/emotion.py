"""Speech emotion recognition.

One preprocessing path (`make_windows`) is shared by training, evaluation, the
PyTorch backend and the ONNX backend, so the numbers in eval/results describe
exactly what the app runs. The ONNX backend imports neither torch nor
transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from .audio import SAMPLE_RATE, rms

log = logging.getLogger(__name__)

LABELS: tuple[str, ...] = ("happy", "sad", "angry", "neutral", "anxious")
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(CLIP_SECONDS * SAMPLE_RATE)
MAX_WINDOWS = 8

# Emotion2Vec's nine output classes mapped onto our five. `None` means the
# class has no counterpart here and its probability mass is discarded.
E2V_TO_LABEL: dict[str, str | None] = {
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "fearful": "anxious",
    "disgusted": None,
    "surprised": None,
    "other": None,
    "<unk>": None,
}


class EmotionUnavailable(RuntimeError):
    """The emotion model could not be loaded."""


@dataclass
class EmotionResult:
    label: str  # one of LABELS, or "unavailable"
    source: str  # "model" | "silence" | "unavailable"
    probs: dict[str, float] = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.label.capitalize()


# ---------------------------------------------------------------- preprocessing


def normalize_clip(clip: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance over the real samples, then zero-pad to 3 s."""
    clip = clip[:CLIP_SAMPLES].astype(np.float32)
    if clip.size:
        clip = (clip - clip.mean()) / np.sqrt(clip.var() + 1e-7)
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    out[: clip.size] = clip
    return out


def make_windows(wave: np.ndarray, max_windows: int = MAX_WINDOWS) -> np.ndarray:
    """Split a recording into 3 s windows (50% overlap). Shape: [n, 48000].

    Clips of 3 s or less give one padded window. Longer clips give several
    windows whose probabilities are averaged by the caller.
    """
    if wave.size <= CLIP_SAMPLES:
        return normalize_clip(wave)[None, :]
    hop = CLIP_SAMPLES // 2
    starts = list(range(0, wave.size - CLIP_SAMPLES + 1, hop))
    if starts[-1] + CLIP_SAMPLES < wave.size:
        starts.append(wave.size - CLIP_SAMPLES)
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]
    return np.stack([normalize_clip(wave[s : s + CLIP_SAMPLES]) for s in starts])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ------------------------------------------------------------ Emotion2Vec labels


def normalize_e2v_label(raw: str) -> str | None:
    """'恐惧/fearful' -> 'anxious'; unmapped classes -> None.

    Raises KeyError for a label this code has never seen, so a changed
    upstream label set cannot be silently scored as zero again.
    """
    english = raw.split("/")[-1].strip().lower()
    return E2V_TO_LABEL[english]


def map_e2v_scores(labels: Sequence[str], scores: Sequence[float]) -> np.ndarray:
    """Project Emotion2Vec's scores onto LABELS and renormalise to sum to 1."""
    probs = np.zeros(len(LABELS), dtype=np.float64)
    for raw, score in zip(labels, scores):
        mapped = normalize_e2v_label(raw)
        if mapped is not None:
            probs[LABELS.index(mapped)] += float(score)
    total = probs.sum()
    if total <= 0:
        return np.full(len(LABELS), 1.0 / len(LABELS))
    return probs / total


def combine(w2v_probs: np.ndarray, e2v_probs: np.ndarray, w2v_weight: float) -> np.ndarray:
    return w2v_weight * w2v_probs + (1.0 - w2v_weight) * e2v_probs


# --------------------------------------------------------------------- backends


class ProbModel(Protocol):
    def predict_proba(self, wave: np.ndarray) -> np.ndarray: ...


class TorchWav2Vec2:
    """Fine-tuned Wav2Vec2ForSequenceClassification (PyTorch, MPS if available)."""

    def __init__(self, model_dir: Path, device: str | None = None):
        import torch
        from transformers import Wav2Vec2ForSequenceClassification

        if not (model_dir / "config.json").exists():
            raise EmotionUnavailable(f"no model at {model_dir}")
        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        id2label = self.model.config.id2label
        got = tuple(id2label[i].lower() for i in range(len(id2label)))
        if got != LABELS:
            raise EmotionUnavailable(f"model labels {got} do not match {LABELS}")

    def predict_logits(self, windows: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(windows).to(self.device)
            return self.model(input_values=x).logits.float().cpu().numpy()

    def predict_proba(self, wave: np.ndarray) -> np.ndarray:
        return softmax(self.predict_logits(make_windows(wave))).mean(axis=0)


class OnnxWav2Vec2:
    """The same classifier exported to ONNX. Needs only onnxruntime and numpy."""

    def __init__(self, onnx_path: Path, threads: int | None = None):
        import onnxruntime as ort

        if not onnx_path.exists():
            raise EmotionUnavailable(f"no ONNX model at {onnx_path}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_logits(self, windows: np.ndarray) -> np.ndarray:
        # Exported with a dynamic batch axis, but run one window at a time to
        # keep peak memory flat on small hosts.
        return np.concatenate(
            [self.session.run(None, {self.input_name: w[None, :]})[0] for w in windows]
        )

    def predict_proba(self, wave: np.ndarray) -> np.ndarray:
        return softmax(self.predict_logits(make_windows(wave))).mean(axis=0)


class Emotion2Vec:
    """FunASR Emotion2Vec, with its labels mapped onto LABELS."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        from funasr import AutoModel

        if not (model_dir / "model.pt").exists() or (model_dir / "model.pt").stat().st_size < 1_000_000:
            raise EmotionUnavailable(f"Emotion2Vec weights missing at {model_dir} (Git LFS pointer only?)")
        self.model = AutoModel(model=str(model_dir), device=device, disable_update=True, disable_pbar=True)

    def predict_proba(self, wave: np.ndarray) -> np.ndarray:
        res = self.model.generate(input=wave.astype(np.float32), fs=SAMPLE_RATE, granularity="utterance")
        return map_e2v_scores(res[0]["labels"], res[0]["scores"])


# ------------------------------------------------------------------- detector


class EmotionDetector:
    """Silence gate, then the acoustic model (optionally ensembled)."""

    def __init__(
        self,
        primary: ProbModel | None,
        secondary: ProbModel | None = None,
        w2v_weight: float = 1.0,
        silence_rms: float = 0.008,
    ):
        self.primary = primary
        self.secondary = secondary
        self.w2v_weight = w2v_weight
        self.silence_rms = silence_rms

    @property
    def available(self) -> bool:
        return self.primary is not None

    def detect(self, wave: np.ndarray) -> EmotionResult:
        if self.primary is None:
            return EmotionResult("unavailable", "unavailable")
        if rms(wave) < self.silence_rms:
            return EmotionResult("neutral", "silence")
        probs = self.primary.predict_proba(wave)
        if self.secondary is not None and self.w2v_weight < 1.0:
            probs = combine(probs, self.secondary.predict_proba(wave), self.w2v_weight)
        label = LABELS[int(np.argmax(probs))]
        return EmotionResult(label, "model", {l: round(float(p), 4) for l, p in zip(LABELS, probs)})


def build_detector(settings) -> EmotionDetector:
    """Load the configured backend. A missing model is reported, not hidden:
    the detector then returns label "unavailable" and /health says so."""
    primary: ProbModel | None = None
    try:
        if settings.emotion_backend == "onnx":
            primary = OnnxWav2Vec2(settings.emotion_onnx_path)
        else:
            primary = TorchWav2Vec2(settings.emotion_model_dir)
    except (EmotionUnavailable, ImportError, OSError) as exc:
        log.error("Emotion model not loaded (%s). Emotion will show as 'Unavailable'.", exc)

    secondary: ProbModel | None = None
    if primary is not None and settings.enable_emotion2vec:
        if settings.emotion_backend == "onnx":
            log.warning("ENABLE_EMOTION2VEC is ignored with EMOTION_BACKEND=onnx (it needs PyTorch).")
        else:
            try:
                secondary = Emotion2Vec(settings.emotion2vec_dir)
            except (EmotionUnavailable, ImportError, OSError) as exc:
                log.error("Emotion2Vec not loaded (%s). Using Wav2Vec2 alone.", exc)

    return EmotionDetector(primary, secondary, settings.ensemble_w2v_weight, settings.silence_rms)
