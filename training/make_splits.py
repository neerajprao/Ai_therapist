"""Speaker-independent train/val/test split, plus the v1-reproduction split.

Reads  training/splits/dataset_index.csv  (from training.inventory)
Writes training/splits/split_v2.csv        main split, speech only
       training/splits/split_v1_repro.csv  old notebook's data recipe, new test set
       eval/results/splits.json            sizes and leakage checks

Rules for the main split:
  * five target labels only; RAVDESS song is excluded;
  * identical audio with conflicting labels is dropped, exact duplicates deduplicated;
  * no speaker appears in more than one split;
  * TESS (2 speakers, 1,000 clips each) is train-only, because one TESS speaker
    would otherwise be about half of a held-out set;
  * val and test each get 10 CREMA-D, 3 RAVDESS and 1 SAVEE speaker.

Run from the repo root:  python -m training.make_splits
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "training" / "splits"
SEED = 42
HELD_OUT = {"CREMA-D": 10, "RAVDESS": 3, "SAVEE": 1}  # speakers per held-out split
FIELDS = ["path", "corpus", "speaker", "channel", "label", "duration", "split"]


def load_index() -> list[dict]:
    with (SPLITS / "dataset_index.csv").open() as f:
        return list(csv.DictReader(f))


def clean(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep five-class clips; resolve duplicate audio."""
    rows = [r for r in rows if r["label"]]
    groups = defaultdict(list)
    for r in rows:
        groups[r["md5"]].append(r)
    keep, conflicts, exact = [], 0, 0
    for group in groups.values():
        if len({r["label"] for r in group}) > 1:
            conflicts += len(group)  # same audio, different labels: unusable
            continue
        exact += len(group) - 1
        keep.append(sorted(group, key=lambda r: r["path"])[0])
    keep.sort(key=lambda r: r["path"])
    return keep, {"dropped_conflicting_label_duplicates": conflicts, "dropped_exact_duplicates": exact}


def assign_speakers(rows: list[dict]) -> dict[str, str]:
    rng = random.Random(SEED)
    by_corpus = defaultdict(set)
    for r in rows:
        by_corpus[r["corpus"]].add(r["speaker"])
    assignment = {}
    for corpus in sorted(by_corpus):
        speakers = sorted(by_corpus[corpus])
        n = HELD_OUT.get(corpus, 0)
        if corpus == "RAVDESS":
            # Odd actor numbers are male, even are female: alternate so each
            # held-out split has both.
            male = [s for s in speakers if int(s.split("_")[1]) % 2 == 1]
            female = [s for s in speakers if int(s.split("_")[1]) % 2 == 0]
            rng.shuffle(male)
            rng.shuffle(female)
            test, val = [male[0], female[0], male[1]], [female[1], male[2], female[2]]
        else:
            rng.shuffle(speakers)
            test, val = speakers[:n], speakers[n : 2 * n]
        for s in speakers:
            assignment[s] = "test" if s in test else "val" if s in val else "train"
    return assignment


def write(name: str, rows: list[dict]) -> None:
    with (SPLITS / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def describe(rows: list[dict]) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        out[split] = {
            "clips": len(part),
            "speakers": len({r["speaker"] for r in part}),
            "by_label": dict(sorted(Counter(r["label"] for r in part).items())),
            "by_corpus": dict(sorted(Counter(r["corpus"] for r in part).items())),
        }
    return out


def speaker_overlap(rows: list[dict]) -> dict:
    sets = {s: {r["speaker"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def main() -> None:
    rows, dup_stats = clean(load_index())
    assignment = assign_speakers(rows)

    # ---- main split: speech only
    main_rows = [dict(r, split=assignment[r["speaker"]]) for r in rows if r["channel"] == "speech"]
    write("split_v2.csv", main_rows)

    # ---- v1 reproduction: the old notebook used RAVDESS only (speech + song)
    # and split by file. Reproduce that on the speakers that are in our train
    # split, and keep the new speaker-independent test set for scoring.
    rng = random.Random(SEED)
    pool = [r for r in rows if r["corpus"] == "RAVDESS" and assignment[r["speaker"]] == "train"]
    by_label = defaultdict(list)
    for r in pool:
        by_label[r["label"]].append(r)
    v1_rows = []
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        cut = round(0.2 * len(items))
        v1_rows += [dict(r, split="val") for r in items[:cut]]
        v1_rows += [dict(r, split="train") for r in items[cut:]]
    v1_rows += [r for r in main_rows if r["split"] == "test"]
    v1_rows.sort(key=lambda r: r["path"])
    write("split_v1_repro.csv", v1_rows)

    main_test = {r["path"] for r in main_rows if r["split"] == "test"}
    summary = {
        "seed": SEED,
        "duplicates": dup_stats,
        "excluded_ravdess_song_clips": sum(1 for r in rows if r["channel"] == "song"),
        "split_v2": {
            **describe(main_rows),
            "speaker_overlap": speaker_overlap(main_rows),
            "held_out_speakers": {
                s: sorted(k for k, v in assignment.items() if v == s) for s in ("val", "test")
            },
        },
        "split_v1_repro": {
            **describe(v1_rows),
            "note": "train/val are split by FILE over the same RAVDESS speakers (as in the v1 notebook), "
            "so its val score is leaky by design; test is the v2 speaker-independent test set.",
            "test_identical_to_v2": {r["path"] for r in v1_rows if r["split"] == "test"} == main_test,
            "train_speakers_in_v2_test": len(
                {r["speaker"] for r in v1_rows if r["split"] != "test"}
                & {r["speaker"] for r in main_rows if r["split"] == "test"}
            ),
        },
    }
    out = ROOT / "eval" / "results" / "splits.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
