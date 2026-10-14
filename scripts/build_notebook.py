"""Build and execute training/SoulSync_training_and_eval.ipynb.

  python scripts/build_notebook.py

The notebook is the single place to look up every number: dataset counts, the
split, every epoch of every training run, test metrics, confusion matrices,
the ensemble sweep, ONNX, memory and latency results. It reads the same JSON
files that the scripts write to eval/results/, so it cannot disagree with them.
Training cells only retrain when RETRAIN is set to True inside the notebook.
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "SoulSync_training_and_eval.ipynb"

cells = [
    md("""# SoulSync v2: training and evaluation record

Everything about how the emotion model was trained and how the whole system was measured, in one place.

**How to read this notebook**

- Every number shown here is loaded from a JSON file in `eval/results/`, written by a script in this repository. The script that produced it is named above each section.
- The saved outputs are from the actual runs. Re-running the notebook redisplays those results; it does **not** retrain unless you set `RETRAIN = True` in the first code cell (about 2.5 hours on an M3 Pro).
- The test split was scored once, after all choices were made on the validation split."""),
    code("""import json, os, platform, subprocess, sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd() if (Path.cwd() / "soulsync").exists() else Path.cwd().parent
os.chdir(ROOT)
RESULTS = ROOT / "eval" / "results"
RETRAIN = False          # True = rerun training from scratch (hours)
RERUN_EVAL = False       # True = rerun the evaluation scripts (needs dataset, models, Ollama)
pd.set_option("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 120)

def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f"NOT RUN: eval/results/{name} does not exist")
        return None
    return json.loads(path.read_text())

def run(module_args, enabled):
    cmd = f"{sys.executable} -m {module_args}"
    if not enabled:
        print(f"(not re-run) command that produced the results below:\\n  python -m {module_args}")
        return
    print("$", cmd)
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}).stdout[-6000:])

def raw(obj, title="Raw JSON"):
    display(Markdown(f"<details><summary>{title}</summary>\\n\\n```json\\n{json.dumps(obj, indent=2, ensure_ascii=False)}\\n```\\n</details>"))

def pct(x): return f"{100*x:.2f}%"
print("repo:", ROOT)"""),
    md("## 0. Environment"),
    code("""import importlib.metadata as im
print("python", sys.version.split()[0], "|", platform.platform(), "|", platform.machine())
for pkg in ["torch", "transformers", "numpy", "scikit-learn", "onnx", "onnxruntime", "chromadb", "faster-whisper", "funasr", "fastapi", "ollama"]:
    try: print(f"{pkg:16s} {im.version(pkg)}")
    except im.PackageNotFoundError: print(f"{pkg:16s} not installed")
try:
    import torch; print("MPS available:", torch.backends.mps.is_available())
except ImportError: pass"""),
    md("""## 1. Dataset inventory

Script: `python -m training.inventory`. Source: Kaggle `uldisvalainis/audio-emotions` (RAVDESS, CREMA-D, TESS, SAVEE). Emotion labels are read from filenames, not folders."""),
    code("""run("training.inventory", RERUN_EVAL)
inv = load("dataset_inventory.json")
if inv:
    print("total files:", inv["total_wav_files"], "| speakers:", inv["total_speakers"], "| unparsed:", len(inv["unparsed_files"]))
    display(Markdown("**Files per corpus and source emotion**"))
    df = pd.DataFrame(inv["corpus_by_source_emotion"]).T.fillna(0).astype(int)
    df["speakers"] = pd.Series(inv["speakers_per_corpus"]); df["total"] = pd.Series(inv["files_per_corpus"]); display(df)
    display(Markdown("**Mapped to the five SoulSync labels** (fearful -> anxious; disgusted and surprised dropped)"))
    display(pd.DataFrame(inv["corpus_by_label5"]).T.fillna(0).astype(int))
    print("five-class totals, all clips:    ", inv["label5_totals"])
    print("five-class totals, speech only:  ", inv["label5_totals_speech_only"])
    print("dropped by source emotion:       ", inv["dropped_by_source_emotion"])
    print("RAVDESS speech vs song:          ", inv["corpus_by_channel"]["RAVDESS"])
    print("folder label != filename label:  ", inv["folder_label_mismatches"])
    print("exact duplicate audio:           ", {k: v for k, v in inv["exact_duplicate_audio"].items() if k != "examples"})
    print("clip duration (s), five-class:   ", inv["duration_seconds_label5"])
    print("sample rates:                    ", inv["samplerates"])
    raw(inv)"""),
    md("""## 2. Speaker-independent split

Script: `python -m training.make_splits` (seed 42). No speaker appears in more than one split. TESS (2 speakers, 1,000 clips each) is train-only. RAVDESS song is excluded."""),
    code("""run("training.make_splits", RERUN_EVAL)
sp = load("splits.json")
if sp:
    for name in ("split_v2", "split_v1_repro"):
        display(Markdown(f"**{name}**"))
        display(pd.DataFrame({k: {"clips": v["clips"], "speakers": v["speakers"], **v["by_label"], **{f"[{c}]": n for c, n in v["by_corpus"].items()}}
                              for k, v in sp[name].items() if k in ("train", "val", "test")}).T.fillna(0).astype(int))
    print("speaker overlap between splits (must all be 0):", sp["split_v2"]["speaker_overlap"])
    print("held-out speakers:", sp["split_v2"]["held_out_speakers"])
    print("duplicates removed:", sp["duplicates"], "| RAVDESS song clips excluded:", sp["excluded_ravdess_song_clips"])
    print("v1 reproduction:", sp["split_v1_repro"]["note"])
    raw(sp)"""),
    md("""## 3. Training

Script: `python -m training.train`. Wav2Vec2 with the CNN feature encoder frozen, 3-second clips at 16 kHz, fp32 on Apple MPS, class-weighted cross-entropy, early stopping on validation macro-F1, checkpoint every epoch.

### 3.1 Timing benchmark (run before the long job)"""),
    code("""run("training.train --run-name main --bench-steps 60", RETRAIN)
bench = load("training_benchmark_main.json")
if bench: display(pd.Series(bench, name="benchmark").to_frame())"""),
    md("### 3.2 Every training run, epoch by epoch"),
    code("""import matplotlib.pyplot as plt

COMMANDS = {
    "v1_repro": "training.train --run-name v1_repro --split split_v1_repro.csv --base-model facebook/wav2vec2-base-960h --epochs 10 --batch-size 4 --accum 4 --lr 3e-5 --warmup-frac 0 --patience 10 --select accuracy --no-random-crop --no-class-weights",
    "main": "training.train --run-name main --epochs 15 --patience 3 --max-minutes 170",
    "no_tess": "training.train --run-name no_tess --exclude-corpus TESS --epochs 8 --patience 3 --max-minutes 60",
    "base960h": "training.train --run-name base960h --base-model facebook/wav2vec2-base-960h --epochs 15 --patience 3 --max-minutes 100",
}
DESCRIPTION = {
    "v1_repro": "Baseline: the old notebook's recipe reproduced (RAVDESS speech+song only, split by FILE, wav2vec2-base-960h). Its validation score is leaky by design.",
    "main": "The v2 model: four corpora, speaker-independent split, wav2vec2-base.",
    "no_tess": "Cross-corpus check: same recipe trained without TESS, later tested on TESS.",
    "base960h": "Comparison: the main recipe on wav2vec2-base-960h instead of wav2vec2-base. FAILED: loss became NaN in the first epoch; stopped by hand.",
}
total_minutes = 0.0
for name, cmd in COMMANDS.items():
    display(Markdown(f"#### Run `{name}`\\n{DESCRIPTION[name]}"))
    run(cmd, RETRAIN)
    log = load(f"training_{name}.json")
    if not log: continue
    total_minutes += log["train_minutes"]
    cfg = log["config"]
    print(f"status: {log['status']} | epochs completed: {log['epochs_completed']} | best epoch: {log['best_epoch']} | "
          f"training time: {log['train_minutes']} min")
    print({k: cfg[k] for k in ["base_model", "split", "exclude_corpus", "epochs", "batch_size", "accum", "lr", "weight_decay", "warmup_frac",
                               "patience", "select", "no_random_crop", "no_class_weights", "seed", "device", "train_clips", "val_clips",
                               "trainable_params", "steps_per_epoch"]})
    print("class counts:", cfg.get("class_counts"), "| class weights:", cfg.get("class_weights"))
    if not log["history"]:
        print("NO COMPLETED EPOCHS. reason:", log.get("reason", ""))
        print("logged steps:", log.get("logged_steps"))
        raw(log, f"Raw record for {name}")
        continue
    hist = pd.DataFrame(log["history"]).set_index("epoch")
    display(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    hist["train_loss"].plot(ax=ax[0], marker="o", title=f"{name}: training loss"); ax[0].set_xlabel("epoch")
    hist[["val_accuracy", "val_macro_f1"]].plot(ax=ax[1], marker="o", title=f"{name}: validation"); ax[1].set_xlabel("epoch")
    ax[1].axvline(log["best_epoch"], color="grey", ls="--", lw=1)
    plt.tight_layout(); plt.show()
    raw(log, f"Raw training log for {name}")"""),
    md("### 3.3 Training time against the 4.5 hour budget"),
    code("""rows = []
for path in sorted(RESULTS.glob("training_*.json")):
    d = json.loads(path.read_text())
    if "train_minutes" in d: rows.append({"run": d["run"], "minutes": d["train_minutes"], "epochs": d["epochs_completed"], "status": d["status"], "file": path.name})
    elif "benchmark_wall_seconds" in d: rows.append({"run": d["run"] + " (benchmark)", "minutes": round(d["benchmark_wall_seconds"] / 60, 2), "epochs": "-", "status": "benchmark", "file": path.name})
budget = pd.DataFrame(rows); display(budget)
if len(budget): print(f"TOTAL training wall-clock: {budget['minutes'].sum():.1f} min = {budget['minutes'].sum()/60:.2f} h of 4.5 h (includes per-epoch validation)")"""),
    md("""## 4. Test-set results

Script: `python -m eval.emotion_eval`. Same windowing as the app. 95% CIs are a clip-level bootstrap (test-set size only; not seed variance, not speaker count)."""),
    code("""def show_eval(name, heading):
    display(Markdown(f"### {heading}"))
    r = load(f"emotion_{name}.json")
    if not r: return None
    print(f"evaluated on: {r['evaluated_on']} | clips: {r['n']} | speakers: {r['speakers']} | model: {r['model_dir']}")
    print(f"ACCURACY  {pct(r['accuracy'])}   95% CI {pct(r['accuracy_ci95'][0])} to {pct(r['accuracy_ci95'][1])}")
    print(f"MACRO-F1  {pct(r['macro_f1'])}   95% CI {pct(r['macro_f1_ci95'][0])} to {pct(r['macro_f1_ci95'][1])}")
    display(pd.DataFrame(r["per_class"]).T)
    cm = r["confusion_matrix"]
    display(Markdown("Confusion matrix (rows = true, columns = predicted)"))
    display(pd.DataFrame(cm["rows_true_cols_pred"], index=cm["labels"], columns=cm["labels"]))
    print("by corpus:", r["by_corpus"]); print("per-speaker accuracy:", r["per_speaker_accuracy"])
    png = RESULTS / f"emotion_{name}.png"
    if png.exists(): display(Image(filename=str(png), width=520))
    raw(r)
    return r

run("eval.emotion_eval --model-dir training/runs/v1_repro/best --split-csv split_v1_repro.csv --name baseline_v1_repro", RERUN_EVAL)
base = show_eval("baseline_v1_repro", "4.1 Baseline: v1 recipe reproduced, on the speaker-independent test set")"""),
    code("""run("eval.emotion_eval --model-dir training/runs/main/best --name test", RERUN_EVAL)
test = show_eval("test", "4.2 SoulSync v2 model, on the speaker-independent test set")"""),
    code("""display(Markdown("### 4.3 Before and after"))
if base and test:
    display(pd.DataFrame({"v1 recipe (reproduced)": {"accuracy": pct(base["accuracy"]), "macro_f1": pct(base["macro_f1"]), "clips": base["n"]},
                          "v2 model": {"accuracy": pct(test["accuracy"]), "macro_f1": pct(test["macro_f1"]), "clips": test["n"]}}).T)
    print(f"accuracy change: {100*(test['accuracy']-base['accuracy']):+.2f} points | macro-F1 change: {100*(test['macro_f1']-base['macro_f1']):+.2f} points")
    print("Note: the baseline is a reproduction of the v1 recipe, not the original v1 weights (not available).")"""),
    md("""### 4.4 Cross-corpus generalisation

The `no_tess` model never saw TESS. It is scored on all five-class TESS clips: two unseen speakers from an unseen corpus."""),
    code("""run("eval.emotion_eval --model-dir training/runs/no_tess/best --only-corpus TESS --name cross_corpus_tess", RERUN_EVAL)
cross = show_eval("cross_corpus_tess", "Leave-TESS-out model, tested on TESS")"""),
    md("""## 5. Emotion2Vec: fixed, then decided with data

Script: `python -m eval.ensemble_eval`. v1 dropped every Emotion2Vec score because of a label mismatch. Here the labels are mapped correctly, the ensemble weight is swept on validation, and the decision rule is fixed before test is looked at."""),
    code("""run("eval.ensemble_eval --model-dir training/runs/main/best", RERUN_EVAL)
ens = load("emotion_ensemble.json")
if ens:
    print("label mapping:", ens["e2v_label_mapping"])
    display(Markdown("**Validation sweep** (weight 1.0 = Wav2Vec2 alone, 0.0 = Emotion2Vec alone)"))
    display(pd.DataFrame(ens["val_sweep"]).set_index("w2v_weight"))
    print("best on validation:", ens["best_on_val"])
    print("gain over Wav2Vec2 alone (val macro-F1):", ens["val_gain_over_w2v_alone"], "| 95% CI:", ens["val_gain_ci95"])
    print("rule:", ens["decision_rule"])
    print("DECISION: ENABLE_EMOTION2VEC =", ens["enable_emotion2vec"], "| chosen Wav2Vec2 weight:", ens["chosen_w2v_weight"])
    display(Markdown("**Test set**"))
    display(pd.DataFrame({k: {m: v[m] for m in ("n", "accuracy", "macro_f1", "accuracy_ci95", "macro_f1_ci95")} for k, v in ens["test"].items()}).T)
    print("CAVEAT:", ens["caveat"])
    raw(ens)"""),
    md("""## 6. ONNX export and INT8 quantisation

Scripts: `python -m training.export_onnx`, then `python -m eval.onnx_benchmark`. CPU only, batch 1, one 3-second clip."""),
    code("""run("training.export_onnx", RERUN_EVAL)
run("eval.onnx_benchmark", RERUN_EVAL)
ob = load("onnx_benchmark.json")
if ob:
    display(pd.DataFrame({k: {"file": v["file"], "size_mb": v["size_mb"], "accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                              **{f"latency_{a}": b for a, b in v["latency_per_3s_clip"].items()}} for k, v in ob["variants"].items()}).T)
    print("INT8 vs PyTorch:", ob["int8_vs_pytorch"]); print("machine:", ob["machine"], "| threads:", ob["cpu_threads"]); print(ob["notes"])
    raw(ob)"""),
    md("""## 7. Memory retrieval (RAG)

Script: `python -m eval.memory_eval`. Synthetic cases in `eval/data/memory_cases.json`. Threshold chosen on one half, reported on the other."""),
    code("""run("eval.memory_eval", RERUN_EVAL)
me = load("memory_eval.json")
if me:
    print(f"{me['users']} users | {me['memories']} memories | {me['followup_cases']} follow-ups | {me['unrelated_cases']} unrelated | {me['embedding_model']} | {me['distance']} | top-{me['top_k']}")
    display(Markdown("**Threshold sweep on the development half**")); display(pd.DataFrame(me["dev_sweep"]).set_index("threshold"))
    print("chosen MEMORY_MAX_DISTANCE:", me["chosen_max_distance"], "|", me["selection_rule"])
    display(Markdown("**Held-out half**")); display(pd.Series(me["test"], name="test").to_frame())
    print("cross-user leaks:", me["cross_user_leaks"]); print("current-session exclusion:", me["current_session_exclusion"])
    print("latency:", me["latency_ms"]); print("CAVEAT:", me["caveat"])
    raw(me)"""),
    md("""## 8. End-to-end latency

Script: `python -m eval.latency_eval --compare-stt`. Real speech clips from the test split, timed stage by stage."""),
    code("""run("eval.latency_eval --compare-stt", RERUN_EVAL)
for mode in ("local", "local_w2v_only", "cloud"):
    display(Markdown(f"### {mode}"))
    lat = load(f"latency_{mode}.json")
    if not lat: continue
    print(f"LLM {lat['llm']} | STT {lat['stt']} | emotion {lat['emotion_backend']} | {lat['clips']} clips | {lat['machine']}")
    display(pd.DataFrame(lat["stages"]).T)
    if "local_stt_comparison" in lat:
        display(Markdown("**Local Whisper engines compared**")); display(pd.DataFrame(lat["local_stt_comparison"]).T)
    print(lat["notes"]); raw(lat)"""),
    md("""## 9. Coping-skills knowledge base (off by default; notes pending review)

Script: `python -m eval.knowledge_eval`. Cases in `eval/data/knowledge_cases.json`."""),
    code("""run("eval.knowledge_eval", RERUN_EVAL)
ke = load("knowledge_eval.json")
if ke:
    display(pd.DataFrame(ke["dev_sweep"]).set_index("threshold"))
    print("chosen KNOWLEDGE_MAX_DISTANCE:", ke["chosen_max_distance"]); print("held-out:", ke["test"]); print("CAVEAT:", ke["caveat"]); raw(ke)"""),
    md("""## 10. Reply quality: automated LLM-as-judge (NOT human evaluation)

Script: `python -m eval.judge_eval`. 20 scripted conversations in `eval/data/judge_conversations.json`. The judge is the same Qwen 2.5 7B model that writes the replies, so its scores are biased upwards; the keyword rule checks are independent of it. The first run is kept as `judge_eval_prompt_v1.json`; fixes were made in response to it."""),
    code("""run("eval.judge_eval", RERUN_EVAL)
for name in ("judge_eval_prompt_v1.json", "judge_eval.json"):
    j = load(name)
    if not j: continue
    display(Markdown(f"**{name}**"))
    print("mean scores:", j["mean_scores"]); print("share 4-5:", j["share_scoring_4_or_5"])
    if "rule_checks" in j: print("rule checks [passed, applicable]:", j["rule_checks"])
    print("crisis layer fired on:", j["crisis_layer_fired"]); print(j.get("history_note", ""))
    display(pd.DataFrame([{"id": r["id"], "category": r["category"], **r["scores"], **r.get("rule_checks", {}), "reply": r["final_reply"]} for r in j["records"]]))"""),
    md("## 11. Every result file, for reference"),
    code("""for path in sorted(RESULTS.iterdir()):
    print(f"{path.stat().st_size:>9,d} bytes  {path.name}")"""),
]

nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
ExecutePreprocessor(timeout=1800, kernel_name="soulsync-venv").preprocess(nb, {"metadata": {"path": str(ROOT)}})
OUT.write_text(nbformat.writes(nb))
print(f"wrote and executed {OUT.relative_to(ROOT)} ({len(cells)} cells)")
