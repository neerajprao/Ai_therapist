"""Compare PyTorch, ONNX FP32 and ONNX INT8: file size, CPU latency, accuracy.

  python -m eval.onnx_benchmark

Latency is for one 3-second window on CPU (batch 1), after warm-up, over the
first 200 test clips. Accuracy and macro-F1 use the full test split with the
same windowing as the app. Writes eval/results/onnx_benchmark.json and .log.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from soulsync.emotion import OnnxWav2Vec2, make_windows
from training.data import ROOT, WaveCache, read_split
from training.infer import clip_probs, load_split_waves
from training.metrics import quick_scores

RESULTS = ROOT / "eval" / "results"


def latency(logits_fn, clips: list[np.ndarray], warmup: int = 10) -> dict:
    windows = [make_windows(w)[:1] for w in clips]
    for w in windows[:warmup]:
        logits_fn(w)
    times = []
    for w in windows:
        t0 = time.perf_counter()
        logits_fn(w)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(times, 50)), 1), "p95_ms": round(float(np.percentile(times, 95)), 1),
            "mean_ms": round(float(np.mean(times)), 1), "n": len(times)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/emotion_w2v2")
    p.add_argument("--onnx-prefix", default="models/emotion_w2v2")
    args = p.parse_args()

    import torch
    from transformers import Wav2Vec2ForSequenceClassification

    from training.infer import torch_logits_fn

    rows = [r for r in read_split("split_v2.csv") if r["split"] == "test"]
    waves, labels = load_split_waves(rows, WaveCache())
    model_dir = ROOT / args.model_dir
    variants = {}

    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).eval()
    torch_fn = torch_logits_fn(model, "cpu")
    paths = {
        "pytorch_fp32": model_dir / "model.safetensors",
        "onnx_fp32": ROOT / f"{args.onnx_prefix}_fp32.onnx",
        "onnx_int8": ROOT / f"{args.onnx_prefix}_int8.onnx",
    }
    fns = {
        "pytorch_fp32": torch_fn,
        "onnx_fp32": OnnxWav2Vec2(paths["onnx_fp32"]).predict_logits,
        "onnx_int8": OnnxWav2Vec2(paths["onnx_int8"]).predict_logits,
    }
    for name, fn in fns.items():
        acc, f1 = quick_scores(labels, clip_probs(fn, waves, batch=8))
        variants[name] = {
            "file": str(paths[name].relative_to(ROOT)),
            "size_mb": round(paths[name].stat().st_size / 1e6, 1),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "latency_per_3s_clip": latency(fn, waves[:200]),
        }
        print(name, variants[name], flush=True)

    base, q = variants["pytorch_fp32"], variants["onnx_int8"]
    result = {
        "test_clips": len(rows),
        "machine": f"{platform.machine()} {platform.platform()}",
        "cpu_threads": torch.get_num_threads(),
        "variants": variants,
        "int8_vs_pytorch": {
            "size_reduction_pct": round(100 * (1 - q["size_mb"] / base["size_mb"]), 1),
            "size_ratio": round(base["size_mb"] / q["size_mb"], 2),
            "p50_speedup": round(base["latency_per_3s_clip"]["p50_ms"] / q["latency_per_3s_clip"]["p50_ms"], 2),
            "accuracy_change": round(q["accuracy"] - base["accuracy"], 4),
            "macro_f1_change": round(q["macro_f1"] - base["macro_f1"], 4),
        },
        "notes": "CPU only (PyTorch on CPU, onnxruntime CPUExecutionProvider), batch 1. "
        "INT8 = dynamic quantisation of MatMul/Gemm; conv feature encoder left in FP32.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "onnx_benchmark.json").write_text(json.dumps(result, indent=2))
    log = "\n".join(
        f"{n:13s} {v['size_mb']:7.1f} MB  acc {v['accuracy']:.4f}  f1 {v['macro_f1']:.4f}  "
        f"p50 {v['latency_per_3s_clip']['p50_ms']} ms  p95 {v['latency_per_3s_clip']['p95_ms']} ms"
        for n, v in variants.items()
    ) + f"\nINT8 vs PyTorch: {result['int8_vs_pytorch']}\n"
    (RESULTS / "onnx_benchmark.log").write_text(log)
    print(log)


if __name__ == "__main__":
    main()
