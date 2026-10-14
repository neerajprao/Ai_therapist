"""Waveform cache and split loading shared by training and evaluation.

Every five-class clip is decoded once to 16 kHz mono int16 (with the same
decoder the app uses) and stored in one array, so epochs do no audio decoding.

Build the cache from the repo root:  python -m training.data
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from soulsync.audio import load_audio_file

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "custom_model" / "Emotions"
SPLITS = ROOT / "training" / "splits"
CACHE = ROOT / "training" / "cache"


def read_split(name: str) -> list[dict]:
    with (SPLITS / name).open() as f:
        return list(csv.DictReader(f))


def build_cache() -> None:
    with (SPLITS / "dataset_index.csv").open() as f:
        paths = sorted(r["path"] for r in csv.DictReader(f) if r["label"])
    CACHE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    chunks, index, offset = [], {}, 0
    for i, rel in enumerate(paths):
        wave = load_audio_file(str(DATA / rel))
        pcm = np.clip(np.round(wave * 32767.0), -32768, 32767).astype(np.int16)
        chunks.append(pcm)
        index[rel] = [offset, int(pcm.size)]
        offset += pcm.size
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(paths)} clips decoded")
    np.save(CACHE / "waves_int16.npy", np.concatenate(chunks))
    (CACHE / "index.json").write_text(json.dumps(index))
    print(f"cached {len(paths)} clips, {offset / 16000 / 3600:.2f} h of audio, in {time.time() - start:.0f} s")


class WaveCache:
    def __init__(self) -> None:
        if not (CACHE / "index.json").exists():
            raise FileNotFoundError("No cache. Run: python -m training.data")
        self.index = json.loads((CACHE / "index.json").read_text())
        self.waves = np.load(CACHE / "waves_int16.npy", mmap_mode="r")

    def get(self, rel_path: str) -> np.ndarray:
        offset, length = self.index[rel_path]
        return np.asarray(self.waves[offset : offset + length], dtype=np.float32) / 32767.0


if __name__ == "__main__":
    build_cache()
