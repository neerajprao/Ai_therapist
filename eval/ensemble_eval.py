"""Decide with data whether Emotion2Vec should be ensembled with Wav2Vec2.

  python -m eval.ensemble_eval --model-dir training/runs/main/best

Steps:
  1. Wav2Vec2 and (correctly label-mapped) Emotion2Vec probabilities for val and test.
  2. On VAL only: sweep the Wav2Vec2 weight w from 0.0 to 1.0 in steps of 0.1,
     p = w * p_w2v + (1 - w) * p_e2v.
  3. Decision rule, fixed before looking at test: enable the ensemble only if
     the best w beats Wav2Vec2 alone on val macro-F1 by at least 0.01 AND the
     paired-bootstrap 95% CI of that gain excludes zero.
  4. Report Wav2Vec2 alone, Emotion2Vec alone and the chosen configuration on TEST.

Writes eval/results/emotion_ensemble.json and .log. Emotion2Vec probabilities
are cached in training/cache/e2v_probs.npz (1.9 GB model, slow on CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from sklearn.metrics import f1_score

from soulsync.emotion import LABELS, Emotion2Vec, combine
from training.data import CACHE, ROOT, WaveCache, read_split
from training.infer import clip_probs, load_split_waves, torch_logits_fn
from training.metrics import compute_metrics, quick_scores

RESULTS = ROOT / "eval" / "results"
MIN_GAIN = 0.01


def macro_f1(y, probs) -> float:
    return float(f1_score(y, probs.argmax(axis=1), labels=list(range(len(LABELS))), average="macro", zero_division=0))


def paired_bootstrap_gain(y, probs_a, probs_b, n=2000, seed=0) -> list[float]:
    """95% CI of macro-F1(b) - macro-F1(a), resampling clips."""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        gains.append(macro_f1(y[idx], probs_b[idx]) - macro_f1(y[idx], probs_a[idx]))
    return [round(float(np.percentile(gains, q)), 4) for q in (2.5, 97.5)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="training/runs/main/best")
    p.add_argument("--device", default="cpu")
    p.add_argument("--e2v-only", action="store_true", help="only compute and cache Emotion2Vec probabilities")
    args = p.parse_args()

    from transformers import Wav2Vec2ForSequenceClassification

    cache = WaveCache()
    rows = read_split("split_v2.csv")
    data = {s: load_split_waves([r for r in rows if r["split"] == s], cache) for s in ("val", "test")}

    e2v_file = CACHE / "e2v_probs.npz"
    if e2v_file.exists():
        saved = np.load(e2v_file)
        e2v = {s: saved[s] for s in data}
        e2v_seconds = float(saved["seconds"])
    else:
        e2v_model = Emotion2Vec(ROOT / "models" / "emotion2vec", device="cpu")
        t0 = time.time()
        e2v = {s: np.stack([e2v_model.predict_proba(w) for w in data[s][0]]) for s in data}
        e2v_seconds = round(time.time() - t0, 1)
        np.savez(e2v_file, seconds=e2v_seconds, **e2v)
    if args.e2v_only:
        print(f"Emotion2Vec probabilities cached ({e2v_seconds} s on CPU)")
        return

    model = Wav2Vec2ForSequenceClassification.from_pretrained(ROOT / args.model_dir).to(args.device).eval()
    w2v = {s: clip_probs(torch_logits_fn(model, args.device), data[s][0]) for s in data}

    y_val, y_test = data["val"][1], data["test"][1]
    sweep = []
    for i in range(11):
        w = round(0.1 * i, 1)
        acc, f1 = quick_scores(y_val, combine(w2v["val"], e2v["val"], w))
        sweep.append({"w2v_weight": w, "val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4)})
    alone = next(r for r in sweep if r["w2v_weight"] == 1.0)
    best = max(sweep, key=lambda r: (r["val_macro_f1"], r["w2v_weight"]))
    gain = round(best["val_macro_f1"] - alone["val_macro_f1"], 4)
    gain_ci = paired_bootstrap_gain(y_val, w2v["val"], combine(w2v["val"], e2v["val"], best["w2v_weight"]))
    enable = best["w2v_weight"] < 1.0 and gain >= MIN_GAIN and gain_ci[0] > 0
    chosen_w = best["w2v_weight"] if enable else 1.0

    def test_metrics(probs):
        m = compute_metrics(y_test, probs)
        return {k: m[k] for k in ("n", "accuracy", "macro_f1", "accuracy_ci95", "macro_f1_ci95", "per_class")}

    result = {
        "model_dir": args.model_dir,
        "val_clips": int(len(y_val)),
        "test_clips": int(len(y_test)),
        "e2v_label_mapping": "angry, happy, neutral, sad kept; fearful -> anxious; disgusted, surprised, other, <unk> dropped; renormalised",
        "val_sweep": sweep,
        "decision_rule": f"enable only if val macro-F1 gain >= {MIN_GAIN} and paired-bootstrap 95% CI of the gain > 0",
        "best_on_val": best,
        "val_gain_over_w2v_alone": gain,
        "val_gain_ci95": gain_ci,
        "enable_emotion2vec": bool(enable),
        "chosen_w2v_weight": chosen_w,
        "test": {
            "wav2vec2_alone": test_metrics(w2v["test"]),
            "emotion2vec_alone": test_metrics(e2v["test"]),
            "chosen": test_metrics(combine(w2v["test"], e2v["test"], chosen_w)),
            "best_val_weight_for_reference": test_metrics(combine(w2v["test"], e2v["test"], best["w2v_weight"])),
        },
        "e2v_inference_seconds_cpu": e2v_seconds,
        "caveat": "Emotion2Vec is a public pretrained model whose training data is not controlled here. If its "
        "training set included RAVDESS, CREMA-D or SAVEE, its scores on this test set are optimistic, because our "
        "test speakers may have been seen by it. Its numbers are therefore not a clean speaker-independent result.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "emotion_ensemble.json").write_text(json.dumps(result, indent=2))
    t = result["test"]
    log = "\n".join(
        [f"val w={r['w2v_weight']:.1f} acc={r['val_accuracy']:.4f} f1={r['val_macro_f1']:.4f}" for r in sweep]
        + [
            f"best on val: w={best['w2v_weight']} gain over w2v alone {gain} CI95 {gain_ci}",
            f"decision: ENABLE_EMOTION2VEC={'true' if enable else 'false'}, w2v weight {chosen_w}",
            f"test w2v alone   acc {t['wav2vec2_alone']['accuracy']} f1 {t['wav2vec2_alone']['macro_f1']}",
            f"test e2v alone   acc {t['emotion2vec_alone']['accuracy']} f1 {t['emotion2vec_alone']['macro_f1']}",
            f"test chosen      acc {t['chosen']['accuracy']} f1 {t['chosen']['macro_f1']}",
        ]
    )
    (RESULTS / "emotion_ensemble.log").write_text(log + "\n")
    print(log)


if __name__ == "__main__":
    main()
