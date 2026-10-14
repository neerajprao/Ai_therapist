"""End-to-end latency by stage, over real speech clips.

  python -m eval.latency_eval                 # local mode from .env / defaults
  MODE=cloud python -m eval.latency_eval      # needs GROQ_API_KEY

Uses 24 clips drawn at random (seed 0) from the test split (acted speech).
Stages: decode, speech-to-text, emotion, memory retrieval, time to
first LLM token, full LLM reply, total. Also compares the two local Whisper
engines when --compare-stt is given. Writes eval/results/latency_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from soulsync.audio import decode_audio
from soulsync.chat import build_messages, tag_user_message
from soulsync.config import load_settings
from soulsync.emotion import build_detector
from soulsync.llm import build_llm
from soulsync.memory import MemoryStore, format_memories
from soulsync.stt import FasterWhisperSTT, OpenAIWhisperSTT, build_stt
from training.data import DATA, ROOT, read_split

RESULTS = ROOT / "eval" / "results"
N_CLIPS = 24


def pct(values: list[float]) -> dict:
    return {"p50_ms": round(float(np.percentile(values, 50)), 1), "p95_ms": round(float(np.percentile(values, 95)), 1),
            "mean_ms": round(float(np.mean(values)), 1), "n": len(values)}


def pick_clips() -> list[Path]:
    rows = [r for r in read_split("split_v2.csv") if r["split"] == "test" and float(r["duration"]) >= 2.0]
    random.Random(0).shuffle(rows)
    return [DATA / r["path"] for r in rows[:N_CLIPS]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-stt", action="store_true", help="also time faster-whisper vs openai-whisper")
    p.add_argument("--tag", default="", help="suffix for the output file, e.g. 'w2v_only'")
    p.add_argument("--pace", type=float, default=0.0, help="seconds to wait between clips (API rate limits)")
    args = p.parse_args()

    settings = load_settings()
    clips = [(path, path.read_bytes()) for path in pick_clips()]
    stt, llm, detector = build_stt(settings), build_llm(settings), build_detector(settings)

    # A small pre-filled memory so the retrieval stage does real work.
    memory = MemoryStore(None, settings.memory_top_k, settings.memory_max_distance)
    stale = memory.collection.get(include=[])["ids"]
    if stale:
        memory.collection.delete(ids=stale)
    cases = json.loads((ROOT / "eval" / "data" / "memory_cases.json").read_text())
    for m in cases["users"]["meera"]:
        memory.add("bench", m["session"], m["text"])
    knowledge = None
    if settings.knowledge_enabled:
        from soulsync.knowledge import KnowledgeBase, format_notes

        knowledge = KnowledgeBase(None, settings.knowledge_max_distance)

    def run(raw: bytes, name: str) -> dict:
        t = {}
        t0 = time.perf_counter()
        wave = decode_audio(raw)
        t["decode"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        emotion = detector.detect(wave)
        t["emotion"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        text = stt.transcribe(wave, raw, name)
        t["stt"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        found = memory.search("bench", "now", text) if text.strip() else []
        notes = knowledge.search(text) if knowledge is not None and text.strip() else []
        t["retrieval"] = time.perf_counter() - t1
        messages = build_messages([], tag_user_message(text or "(silence)", emotion.display),
                                  format_memories(found) if found else None, settings.history_max_turns,
                                  format_notes(notes) if notes else None)
        t1 = time.perf_counter()
        first = None
        chars = 0
        for piece in llm.stream(messages):
            if first is None:
                first = time.perf_counter() - t1
            chars += len(piece)
        t["llm_first_token"] = first if first is not None else time.perf_counter() - t1
        t["llm_total"] = time.perf_counter() - t1
        t["time_to_first_token_end_to_end"] = t["decode"] + t["emotion"] + t["stt"] + t["retrieval"] + t["llm_first_token"]
        t["total"] = time.perf_counter() - t0
        t["reply_chars"] = chars
        return t

    run(clips[0][1], clips[0][0].name)  # warm-up, not recorded
    records = []
    for path, raw in clips:
        time.sleep(args.pace)  # not timed; keeps cloud APIs under their per-minute limits
        records.append(run(raw, path.name))

    stages = ["decode", "emotion", "stt", "retrieval", "llm_first_token", "llm_total",
              "time_to_first_token_end_to_end", "total"]
    result = {
        "mode": settings.mode,
        "llm": llm.name,
        "stt": stt.name,
        "emotion_backend": (
            "unavailable" if not detector.available
            else settings.emotion_backend + (f" + emotion2vec (w2v weight {detector.w2v_weight})" if detector.secondary is not None else "")
        ),
        "clips": len(records),
        "clip_source": "test split, acted speech, 2 s or longer",
        "machine": f"{platform.machine()} {platform.platform()}",
        "stages": {s: pct([r[s] * 1000 for r in records]) for s in stages},
        "mean_reply_chars": round(float(np.mean([r["reply_chars"] for r in records])), 1),
        "pace_seconds_between_clips": args.pace,
        "knowledge_base_enabled": knowledge is not None,
        "notes": "One warm-up request excluded. Sequential requests, single user. "
        "No text-to-speech stage: the browser speaks the reply.",
    }

    if args.compare_stt and settings.stt_backend == "local":
        compare = {}
        for label, cls in (("faster-whisper", FasterWhisperSTT), ("openai-whisper", OpenAIWhisperSTT)):
            engine = cls(settings.whisper_model)
            waves = [decode_audio(raw) for _, raw in clips]
            engine.transcribe(waves[0])
            times, texts = [], []
            for wave in waves:
                t0 = time.perf_counter()
                texts.append(engine.transcribe(wave))
                times.append((time.perf_counter() - t0) * 1000)
            compare[label] = {**pct(times), "model": settings.whisper_model, "example": texts[0]}
        result["local_stt_comparison"] = compare

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"latency_{settings.mode}{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
