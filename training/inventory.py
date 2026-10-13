"""Inventory custom_model/Emotions: corpus, speaker and true emotion for every clip.

Writes:
  training/splits/dataset_index.csv   one row per clip (no audio)
  eval/results/dataset_inventory.json counts used by docs/DATASET.md
  eval/results/dataset_inventory.log  short human-readable log
  docs/DATASET.md                     tables generated from the JSON

Run from the repo root:  python -m training.inventory
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "custom_model" / "Emotions"
INDEX = ROOT / "training" / "splits" / "dataset_index.csv"
RESULTS = ROOT / "eval" / "results"

# Source emotion -> our five labels. Anything else is dropped from training.
TO_LABEL5 = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "fearful": "anxious",
    "disgusted": None,
    "surprised": None,
    "calm": None,
}
FOLDER_EMOTION = {
    "Angry": "angry", "Disgusted": "disgusted", "Fearful": "fearful", "Happy": "happy",
    "Neutral": "neutral", "Sad": "sad", "Suprised": "surprised",
}

RAVDESS = re.compile(r"^1?(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$")
RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry",
               "06": "fearful", "07": "disgusted", "08": "surprised"}
CREMA = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{1,2})\.wav$")
CREMA_EMO = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
TESS = re.compile(r"^(OAF?|YAF)_([a-z]+)_([a-z]+)\.wav$")
TESS_EMO = {"angry": "angry", "disgust": "disgusted", "fear": "fearful", "happy": "happy",
            "neutral": "neutral", "sad": "sad", "ps": "surprised"}
SAVEE = re.compile(r"^(a|d|f|h|n|sa|su)(\d{2})\.wav$")
SAVEE_EMO = {"a": "angry", "d": "disgusted", "f": "fearful", "h": "happy", "n": "neutral",
             "sa": "sad", "su": "surprised"}


def parse(folder: str, name: str) -> dict | None:
    """Corpus, speaker, emotion (from the filename) and channel for one file."""
    if m := RAVDESS.match(name):
        return {
            "corpus": "RAVDESS",
            "speaker": f"ravdess_{m.group(7)}",
            "emotion": RAVDESS_EMO[m.group(3)],
            "channel": {"01": "speech", "02": "song"}[m.group(2)],
        }
    if m := CREMA.match(name):
        return {"corpus": "CREMA-D", "speaker": f"crema_{m.group(1)}", "emotion": CREMA_EMO[m.group(3)], "channel": "speech"}
    if m := TESS.match(name):
        speaker = "OAF" if m.group(1).startswith("OA") else "YAF"
        return {"corpus": "TESS", "speaker": f"tess_{speaker}", "emotion": TESS_EMO[m.group(3)], "channel": "speech"}
    if m := SAVEE.match(name):
        # The speaker prefix (DC/JE/JK/KL) was stripped upstream. Each class
        # folder holds one complete 120-clip speaker set, so the folder is the
        # only remaining speaker key. The emotion comes from the filename.
        return {"corpus": "SAVEE", "speaker": f"savee_in_{folder}", "emotion": SAVEE_EMO[m.group(1)], "channel": "speech"}
    return None


def main() -> None:
    rows, unparsed = [], []
    for path in sorted(DATA.rglob("*.wav")):
        folder = path.parent.name
        info = parse(folder, path.name)
        if info is None:
            unparsed.append(str(path.relative_to(DATA)))
            continue
        audio = sf.info(str(path))
        rows.append(
            {
                "path": str(path.relative_to(DATA)),
                "folder": folder,
                **info,
                "folder_emotion": FOLDER_EMOTION[folder],
                "label": TO_LABEL5[info["emotion"]] or "",
                "duration": round(audio.frames / audio.samplerate, 3),
                "samplerate": audio.samplerate,
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            }
        )

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_md5 = defaultdict(list)
    for r in rows:
        by_md5[r["md5"]].append(r)
    dup_groups = [g for g in by_md5.values() if len(g) > 1]

    def table(key_a: str, key_b: str, subset=None) -> dict:
        out: dict = defaultdict(Counter)
        for r in subset or rows:
            out[r[key_a]][r[key_b] or "(dropped)"] += 1
        return {a: dict(sorted(c.items())) for a, c in sorted(out.items())}

    speakers = defaultdict(set)
    for r in rows:
        speakers[r["corpus"]].add(r["speaker"])
    per_speaker = Counter(r["speaker"] for r in rows if r["label"])
    mismatched = [r for r in rows if r["emotion"] != r["folder_emotion"]]
    five = [r for r in rows if r["label"]]
    durations = sorted(r["duration"] for r in five)

    summary = {
        "total_wav_files": len(rows) + len(unparsed),
        "unparsed_files": unparsed,
        "files_per_folder": dict(Counter(r["folder"] for r in rows)),
        "files_per_corpus": dict(Counter(r["corpus"] for r in rows)),
        "corpus_by_source_emotion": table("corpus", "emotion"),
        "corpus_by_label5": table("corpus", "label"),
        "corpus_by_channel": table("corpus", "channel"),
        "label5_totals": dict(Counter(r["label"] for r in five)),
        "label5_totals_speech_only": dict(Counter(r["label"] for r in five if r["channel"] == "speech")),
        "dropped_by_source_emotion": dict(Counter(r["emotion"] for r in rows if not r["label"])),
        "speakers_per_corpus": {c: len(s) for c, s in sorted(speakers.items())},
        "total_speakers": sum(len(s) for s in speakers.values()),
        "clips_per_speaker_label5": {
            "min": min(per_speaker.values()), "max": max(per_speaker.values()),
            "largest": dict(per_speaker.most_common(6)),
        },
        "folder_label_mismatches": {
            "count": len(mismatched),
            "by_corpus": dict(Counter(r["corpus"] for r in mismatched)),
        },
        "exact_duplicate_audio": {
            "groups": len(dup_groups),
            "files_involved": sum(len(g) for g in dup_groups),
            "cross_speaker_groups": sum(1 for g in dup_groups if len({r["speaker"] for r in g}) > 1),
            "conflicting_label_groups": sum(1 for g in dup_groups if len({r["emotion"] for r in g}) > 1),
            "examples": [[r["path"] for r in g] for g in dup_groups[:5]],
        },
        "duration_seconds_label5": {
            "min": durations[0], "median": durations[len(durations) // 2],
            "p95": durations[int(len(durations) * 0.95)], "max": durations[-1],
            "share_longer_than_3s": round(sum(d > 3.0 for d in durations) / len(durations), 4),
        },
        "samplerates": dict(Counter(str(r["samplerate"]) for r in rows)),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset_inventory.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log = [
        f"files: {summary['total_wav_files']} (unparsed: {len(unparsed)})",
        f"per corpus: {summary['files_per_corpus']}",
        f"speakers: {summary['speakers_per_corpus']} total {summary['total_speakers']}",
        f"5-class totals: {summary['label5_totals']}",
        f"5-class speech only: {summary['label5_totals_speech_only']}",
        f"dropped: {summary['dropped_by_source_emotion']}",
        f"folder/filename label mismatches: {summary['folder_label_mismatches']}",
        f"duplicate audio groups: {summary['exact_duplicate_audio']['groups']}",
        f"index written to {INDEX.relative_to(ROOT)}",
    ]
    (RESULTS / "dataset_inventory.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    write_doc(summary)


LICENCES = [
    ("RAVDESS", "CC BY-NC-SA 4.0", "Livingstone & Russo, 2018. Non-commercial, share-alike. https://zenodo.org/record/1188976"),
    ("CREMA-D", "Open Database License (ODbL) 1.0", "Cao et al., 2014. https://github.com/CheyneyComputerScience/CREMA-D"),
    ("TESS", "CC BY-NC-ND 4.0", "Pichora-Fuller & Dupuis, 2020. Non-commercial, no derivatives. https://doi.org/10.5683/SP2/E8H2MF"),
    ("SAVEE", "Research use only, by registration", "Haq & Jackson, 2009. http://kahlan.eps.surrey.ac.uk/savee/"),
]


def md_table(header: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def write_doc(s: dict) -> None:
    emotions = ["angry", "happy", "sad", "neutral", "fearful", "disgusted", "surprised"]
    labels = ["happy", "sad", "angry", "neutral", "anxious"]
    corpora = list(s["corpus_by_source_emotion"])
    dup = s["exact_duplicate_audio"]
    dur = s["duration_seconds_label5"]
    doc = f"""# Dataset

Generated by `python -m training.inventory` from `eval/results/dataset_inventory.json`. Do not edit the numbers by hand.

Source: Kaggle dataset `uldisvalainis/audio-emotions`, a repackaging of four acted-speech corpora into seven class folders. The audio is not committed to this repository.

## Files per corpus and source emotion

Emotion is read from each filename, not from the folder (see "Label problems").

{md_table(["Corpus", "Speakers", *emotions, "Total"], [[c, s["speakers_per_corpus"][c], *[s["corpus_by_source_emotion"][c].get(e, 0) for e in emotions], s["files_per_corpus"][c]] for c in corpora])}

Total: {s["total_wav_files"]} files, {s["total_speakers"]} speakers. Unparsed files: {len(s["unparsed_files"])}.

RAVDESS contains both speech and song: {s["corpus_by_channel"]["RAVDESS"]}.

## Mapping to the five SoulSync labels

`fearful` becomes `anxious`. `disgusted` and `surprised` are dropped: {s["dropped_by_source_emotion"]}.

{md_table(["Corpus", *labels, "Dropped"], [[c, *[s["corpus_by_label5"][c].get(l, 0) for l in labels], s["corpus_by_label5"][c].get("(dropped)", 0)] for c in corpora])}

All five-class clips: {s["label5_totals"]}

Speech only (RAVDESS song removed): {s["label5_totals_speech_only"]}

## Speakers

Five-class clips per speaker range from {s["clips_per_speaker_label5"]["min"]} to {s["clips_per_speaker_label5"]["max"]}. The largest: {s["clips_per_speaker_label5"]["largest"]}.

Speaker IDs come from filenames: the actor number for RAVDESS and CREMA-D, and the `OAF`/`YAF` prefix for TESS. SAVEE's speaker prefix was removed upstream; each of four class folders holds one complete 120-clip set of distinct audio, so the folder name is used as the speaker key. That is an inference, not ground truth.

## Label problems found

- **SAVEE is filed under the wrong folders.** {s["folder_label_mismatches"]["count"]} of 480 SAVEE clips sit in a folder whose emotion differs from the emotion code in the filename. The filename is used.
- **Exact duplicate audio:** {dup["groups"]} groups ({dup["files_involved"]} files), {dup["conflicting_label_groups"]} of them with two different labels for identical audio. Examples: {dup["examples"]}.
- Two RAVDESS filename variants (`103-...`) and one TESS typo (`OA_bite_neutral.wav`) are parsed as their intended forms.

## Clip length and sample rate

Five-class clips: min {dur["min"]} s, median {dur["median"]} s, 95th percentile {dur["p95"]} s, max {dur["max"]} s. {dur["share_longer_than_3s"]:.1%} are longer than the 3 s training window.

Sample rates: {s["samplerates"]}. Everything is resampled to 16 kHz.

## Licences

{md_table(["Corpus", "Licence", "Reference"], [list(row) for row in LICENCES])}

These are the terms published by each corpus's authors as I understand them; confirm against the source pages before publishing model weights. Three of the four restrict commercial use, so a model trained on this data should be treated as non-commercial research output.

All four corpora are acted, English-language recordings. None contains spontaneous or clinical speech.
"""
    out = ROOT / "docs" / "DATASET.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(doc)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
