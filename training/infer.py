"""Batched inference that mirrors the app: 3 s windows, probabilities averaged per clip."""

from __future__ import annotations

import numpy as np

from soulsync.emotion import LABELS, make_windows, softmax

from .data import WaveCache


def clip_probs(logits_fn, waves: list[np.ndarray], batch: int = 32) -> np.ndarray:
    """`logits_fn` maps float32 [n, 48000] windows to [n, 5] logits."""
    windows, owner = [], []
    for i, wave in enumerate(waves):
        w = make_windows(wave)
        windows.append(w)
        owner += [i] * len(w)
    windows = np.concatenate(windows)
    owner = np.array(owner)
    logits = np.concatenate([logits_fn(windows[s : s + batch]) for s in range(0, len(windows), batch)])
    probs = softmax(logits)
    return np.stack([probs[owner == i].mean(axis=0) for i in range(len(waves))])


def load_split_waves(rows: list[dict], cache: WaveCache) -> tuple[list[np.ndarray], np.ndarray]:
    waves = [cache.get(r["path"]) for r in rows]
    labels = np.array([LABELS.index(r["label"]) for r in rows])
    return waves, labels


def torch_logits_fn(model, device: str):
    import torch

    def fn(windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
            return model(input_values=x).logits.float().cpu().numpy()

    return fn
