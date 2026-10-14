"""Score an emotion model on one split and save JSON, a log and a confusion matrix.

  python -m eval.emotion_eval --model-dir training/runs/main/best --name test
  python -m eval.emotion_eval --model-dir training/runs/v1_repro/best \
      --split-csv split_v1_repro.csv --name baseline_v1_repro

Outputs eval/results/emotion_<name>.json, .log and .png.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from soulsync.emotion import LABELS
from training.data import ROOT, WaveCache, read_split
from training.infer import clip_probs, load_split_waves, torch_logits_fn
from training.metrics import compute_metrics, save_confusion_png

RESULTS = ROOT / "eval" / "results"


def breakdown(rows: list[dict], labels: np.ndarray, probs: np.ndarray, key: str) -> dict:
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[key]].append(i)
    pred = probs.argmax(axis=1)
    return {g: {"n": len(idx), "accuracy": round(float((pred[idx] == labels[idx]).mean()), 4)} for g, idx in sorted(groups.items())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--split-csv", default="split_v2.csv")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--only-corpus", default="", help="score only this corpus, from every split (cross-corpus test)")
    p.add_argument("--name", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import torch
    from transformers import Wav2Vec2ForSequenceClassification

    rows = read_split(args.split_csv)
    if args.only_corpus:
        rows = [r for r in rows if r["corpus"] == args.only_corpus]
        split_desc = f"all {args.only_corpus} clips"
    else:
        rows = [r for r in rows if r["split"] == args.split]
        split_desc = f"{args.split} split of {args.split_csv}"
    waves, labels = load_split_waves(rows, WaveCache())

    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_dir).to(args.device).eval()
    start = time.time()
    probs = clip_probs(torch_logits_fn(model, args.device), waves)
    seconds = time.time() - start

    metrics = compute_metrics(labels, probs)
    per_speaker = breakdown(rows, labels, probs, "speaker")
    speaker_accs = [v["accuracy"] for v in per_speaker.values()]
    result = {
        "name": args.name,
        "model_dir": args.model_dir,
        "evaluated_on": split_desc,
        "speakers": len(per_speaker),
        **metrics,
        "by_corpus": breakdown(rows, labels, probs, "corpus"),
        "per_speaker_accuracy": {"min": min(speaker_accs), "median": float(np.median(speaker_accs)), "max": max(speaker_accs)},
        "ci_note": "95% CIs are a clip-level bootstrap (2,000 resamples). They reflect test-set size only, "
        "not seed variance or the small number of speakers.",
        "inference_seconds": round(seconds, 1),
        "device": args.device,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"emotion_{args.name}.json").write_text(json.dumps(result, indent=2))
    save_confusion_png(metrics, RESULTS / f"emotion_{args.name}.png", args.title or f"{args.name} ({split_desc})")
    log = (
        f"{args.name}: {split_desc}, n={metrics['n']}, speakers={len(per_speaker)}\n"
        f"accuracy {metrics['accuracy']} CI95 {metrics['accuracy_ci95']}\n"
        f"macro-F1 {metrics['macro_f1']} CI95 {metrics['macro_f1_ci95']}\n"
        + "\n".join(f"  {l:8s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} n={v['support']}" for l, v in metrics["per_class"].items())
        + f"\nby corpus: {result['by_corpus']}\n"
    )
    (RESULTS / f"emotion_{args.name}.log").write_text(log)
    print(log)


if __name__ == "__main__":
    main()
