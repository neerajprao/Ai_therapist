"""Fine-tune Wav2Vec2 for five-class speech emotion recognition.

Examples (from the repo root):
  python -m training.train --run-name main --bench-steps 60         # timing only
  python -m training.train --run-name main --epochs 12              # train
  python -m training.train --run-name main --epochs 12 --resume     # continue

Checkpoints: training/runs/<run>/last (every epoch) and .../best.
Log:         eval/results/training_<run>.json (per-epoch metrics, wall-clock).
Only the train and val splits are touched here; the test set is scored once by
eval/emotion_eval.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification

from soulsync.emotion import CLIP_SAMPLES, LABELS, normalize_clip

from .data import ROOT, WaveCache, read_split
from .infer import clip_probs, load_split_waves, torch_logits_fn
from .metrics import quick_scores

RUNS = ROOT / "training" / "runs"
RESULTS = ROOT / "eval" / "results"


class ClipDataset(Dataset):
    def __init__(self, rows: list[dict], cache: WaveCache, random_crop: bool, seed: int):
        self.rows, self.cache, self.random_crop = rows, cache, random_crop
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        wave = self.cache.get(row["path"])
        if wave.size > CLIP_SAMPLES:
            start = int(self.rng.integers(0, wave.size - CLIP_SAMPLES + 1)) if self.random_crop else 0
            wave = wave[start : start + CLIP_SAMPLES]
        return torch.from_numpy(normalize_clip(wave)), LABELS.index(row["label"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--split", default="split_v2.csv")
    p.add_argument("--base-model", default="facebook/wav2vec2-base")
    p.add_argument("--exclude-corpus", default="", help="drop this corpus from train and val (cross-corpus test)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--select", choices=["macro_f1", "accuracy"], default="macro_f1")
    p.add_argument("--no-random-crop", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-minutes", type=float, default=180.0, help="stop cleanly after this much training time")
    p.add_argument("--bench-steps", type=int, default=0, help="time N steps and exit without saving")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = read_split(args.split)
    if args.exclude_corpus:
        rows = [r for r in rows if r["corpus"] != args.exclude_corpus or r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    cache = WaveCache()
    val_waves, val_labels = load_split_waves(val_rows, cache)

    run_dir = RUNS / args.run_name
    last_dir, best_dir = run_dir / "last", run_dir / "best"
    state_file = last_dir / "state.pt"
    resuming = args.resume and state_file.exists()

    label2id = {l: i for i, l in enumerate(LABELS)}
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        last_dir if resuming else args.base_model,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label={i: l for l, i in label2id.items()},
    )
    model.freeze_feature_encoder()
    model.to(device)  # fp32: half precision is unreliable on MPS

    counts = np.bincount([LABELS.index(r["label"]) for r in train_rows], minlength=len(LABELS))
    weights = None if args.no_class_weights else torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(
        ClipDataset(train_rows, cache, not args.no_random_crop, args.seed),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    steps_per_epoch = len(loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    warmup = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )

    state = {"epoch": 0, "best": -1.0, "best_epoch": 0, "bad_epochs": 0, "train_seconds": 0.0, "history": []}
    if resuming:
        saved = torch.load(state_file, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        print(f"resumed after epoch {state['epoch']} ({state['train_seconds'] / 60:.1f} min used)")

    config = {**vars(args), "device": device, "train_clips": len(train_rows), "val_clips": len(val_rows),
              "class_counts": dict(zip(LABELS, counts.tolist())),
              "class_weights": None if weights is None else [round(float(w), 4) for w in weights.cpu()],
              "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "steps_per_epoch": steps_per_epoch, "torch": torch.__version__}
    print(json.dumps(config, indent=2))

    def validate() -> tuple[float, float]:
        model.eval()
        probs = clip_probs(torch_logits_fn(model, device), val_waves)
        model.train()
        return quick_scores(val_labels, probs)

    def write_log(status: str) -> None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"training_{args.run_name}.json").write_text(json.dumps({
            "run": args.run_name, "status": status, "config": config,
            "epochs_completed": state["epoch"], "best_epoch": state["best_epoch"],
            f"best_val_{args.select}": round(state["best"], 4),
            "train_seconds": round(state["train_seconds"], 1),
            "train_minutes": round(state["train_seconds"] / 60, 2),
            "history": state["history"],
        }, indent=2))

    # ---------------------------------------------------------------- benchmark
    if args.bench_steps:
        model.train()
        it = iter(loader)
        times = []
        for step in range(args.bench_steps):
            x, y = next(it)
            t0 = time.time()
            loss = loss_fn(model(input_values=x.to(device)).logits, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.item()  # forces MPS to finish the step
            times.append(time.time() - t0)
        steady = float(np.mean(times[5:]))
        t0 = time.time()
        model.eval()
        clip_probs(torch_logits_fn(model, device), val_waves[:200])
        val_seconds = (time.time() - t0) * len(val_waves) / 200
        epoch_minutes = (steady * len(loader) + val_seconds) / 60
        bench = {
            "run": args.run_name, "device": device, "batch_size": args.batch_size, "steps_timed": args.bench_steps - 5,
            "seconds_per_step": round(steady, 4), "clips_per_second": round(args.batch_size / steady, 1),
            "train_clips": len(train_rows), "batches_per_epoch": len(loader),
            "projected_val_seconds_per_epoch": round(val_seconds, 1),
            "projected_minutes_per_epoch": round(epoch_minutes, 2),
            "projected_minutes_for_epochs": {str(e): round(e * epoch_minutes, 1) for e in (8, 10, 12, 15)},
            "benchmark_wall_seconds": round(sum(times) + val_seconds * 200 / len(val_waves), 1),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"training_benchmark_{args.run_name}.json").write_text(json.dumps(bench, indent=2))
        print(json.dumps(bench, indent=2))
        return

    # ------------------------------------------------------------------- train
    model.train()
    status = "completed"
    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        t0 = time.time()
        running, seen = 0.0, 0
        optimizer.zero_grad()
        for step, (x, y) in enumerate(loader, 1):
            loss = loss_fn(model(input_values=x.to(device)).logits, y.to(device))
            (loss / args.accum).backward()
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.item() * len(y)
            seen += len(y)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / seen:.4f}", flush=True)
        acc, f1 = validate()
        seconds = time.time() - t0
        state["epoch"] = epoch
        state["train_seconds"] += seconds
        score = f1 if args.select == "macro_f1" else acc
        improved = score > state["best"]
        state["history"].append({"epoch": epoch, "train_loss": round(running / seen, 4), "val_accuracy": round(acc, 4),
                                 "val_macro_f1": round(f1, 4), "seconds": round(seconds, 1), "improved": improved})
        print(f"== epoch {epoch}: loss {running / seen:.4f} val_acc {acc:.4f} val_f1 {f1:.4f} "
              f"({seconds / 60:.1f} min, total {state['train_seconds'] / 60:.1f} min){' *' if improved else ''}", flush=True)

        model.save_pretrained(last_dir)
        if improved:
            state.update(best=score, best_epoch=epoch, bad_epochs=0)
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(last_dir, best_dir, ignore=shutil.ignore_patterns("state.pt"))
        else:
            state["bad_epochs"] += 1
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state}, state_file)

        if state["bad_epochs"] >= args.patience:
            status = f"early_stopped_after_epoch_{epoch}"
        elif state["train_seconds"] / 60 >= args.max_minutes:
            status = f"time_limit_after_epoch_{epoch}"
        else:
            status = "completed" if epoch == args.epochs else "running"
        write_log(status)
        if status != "running" and status != "completed":
            break
    print(f"done: {status}; best val {args.select} {state['best']:.4f} at epoch {state['best_epoch']}; "
          f"{state['train_seconds'] / 60:.1f} min of training")


if __name__ == "__main__":
    main()
