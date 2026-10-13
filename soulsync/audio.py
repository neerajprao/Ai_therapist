"""Decode browser recordings (WebM/Opus, MP4, WAV...) to 16 kHz mono float32.

Uses PyAV, which bundles FFmpeg, so neither a system ffmpeg nor PyTorch is
needed. Recordings are decoded in memory and never written to disk.
"""

from __future__ import annotations

import io

import numpy as np

SAMPLE_RATE = 16_000


class AudioDecodeError(ValueError):
    """The uploaded bytes could not be decoded as audio."""


def decode_audio(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import av

    if not data:
        raise AudioDecodeError("empty upload")
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                raise AudioDecodeError("no audio stream in upload")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            chunks = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except (av.error.FFmpegError, ValueError, IndexError) as exc:
        raise AudioDecodeError(f"could not decode audio: {exc}") from exc

    if not chunks:
        raise AudioDecodeError("audio stream contained no samples")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_audio(f.read(), sample_rate)


def rms(wave: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wave)))) if wave.size else 0.0
