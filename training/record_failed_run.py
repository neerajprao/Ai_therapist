"""Record a training run that was stopped before finishing an epoch.

  python -m training.record_failed_run --run-name base960h --reason "..."

train.py only writes eval/results/training_<run>.json after an epoch, so a run
killed earlier would vanish from the budget. This reads the run's log file and
counts its wall-clock time from creation to last write.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--extra-minutes", type=float, default=0.0, help="diagnostic time spent on this run")
    args = p.parse_args()

    log = ROOT / "training" / "logs" / f"{args.run_name}.log"
    stat = os.stat(log)
    minutes = (stat.st_mtime - stat.st_birthtime) / 60
    text = log.read_text(errors="replace")
    steps = re.findall(r"epoch (\d+) step (\d+)/(\d+) loss (\S+)", text)
    config = json.loads(text[text.index("{"): text.index("}\n") + 1]) if "{" in text else {}
    record = {
        "run": args.run_name,
        "status": "failed",
        "reason": args.reason,
        "config": config,
        "epochs_completed": 0,
        "best_epoch": 0,
        "logged_steps": [{"epoch": int(e), "step": int(s), "of": int(n), "loss": l} for e, s, n, l in steps],
        "train_seconds": round(minutes * 60 + args.extra_minutes * 60, 1),
        "train_minutes": round(minutes + args.extra_minutes, 2),
        "history": [],
        "minutes_from_log_file_times": round(minutes, 2),
        "extra_diagnostic_minutes": args.extra_minutes,
    }
    out = ROOT / "eval" / "results" / f"training_{args.run_name}.json"
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
