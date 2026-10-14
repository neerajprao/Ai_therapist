"""Generate RESULTS.md from the JSON files in eval/results/. No hand-typed numbers.

  python -m eval.make_results

Sections whose JSON file is missing are listed as "not run" instead of being
left out, so a gap is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(pair: list[float]) -> str:
    return f"{100 * pair[0]:.1f} to {100 * pair[1]:.1f}"


def not_run(name: str, why: str = "") -> str:
    return f"_Not run: `eval/results/{name}` does not exist.{' ' + why if why else ''}_"


def emotion_section() -> str:
    out = ["## 1. Speech emotion recognition", ""]
    splits = load("splits.json")
    if splits:
        s = splits["split_v2"]
        out += [
            "Speaker-independent split (no speaker in more than one split, overlap checks: "
            f"{s['speaker_overlap']}). Source: `eval/results/splits.json`.",
            "",
            table(["Split", "Clips", "Speakers", "Corpora"],
                  [[k, s[k]["clips"], s[k]["speakers"], ", ".join(f"{c} {n}" for c, n in s[k]["by_corpus"].items())] for k in ("train", "val", "test")]),
            "",
        ]
    test, base = load("emotion_test.json"), load("emotion_baseline_v1_repro.json")
    rows = []
    if base:
        rows.append(["v1 recipe, reproduced (RAVDESS only, split by file)", base["n"], pct(base["accuracy"]), ci(base["accuracy_ci95"]), pct(base["macro_f1"]), ci(base["macro_f1_ci95"]), "`emotion_baseline_v1_repro.json`"])
    if test:
        rows.append(["v2 Wav2Vec2 (4 corpora, speaker-independent)", test["n"], pct(test["accuracy"]), ci(test["accuracy_ci95"]), pct(test["macro_f1"]), ci(test["macro_f1_ci95"]), "`emotion_test.json`"])
    if rows:
        out += ["### Test set (scored once)", "", table(["Model", "Clips", "Accuracy", "95% CI", "Macro-F1", "95% CI", "Source"], rows), ""]
        out += ["The CIs are a clip-level bootstrap. They reflect test-set size only, not seed variance or the small number of test speakers.", ""]
    else:
        out += [not_run("emotion_test.json"), ""]
    if base:
        log = load("training_v1_repro.json")
        if log:
            leaky = max(h["val_accuracy"] for h in log["history"])
            out += [
                f"The v1 baseline is a reproduction of the old notebook's recipe, not the original weights (those were not available). "
                f"On its own file-level validation split, which shares speakers with training, it scores {pct(leaky)}; "
                f"on the speaker-independent test set it scores {pct(base['accuracy'])}. Source: `training_v1_repro.json`.",
                "",
            ]
    if test:
        out += ["### v2 per-class results on test", "",
                table(["Class", "Precision", "Recall", "F1", "Support"],
                      [[c, pct(v["precision"]), pct(v["recall"]), pct(v["f1"]), v["support"]] for c, v in test["per_class"].items()]),
                "", "By corpus: " + ", ".join(f"{c} {pct(v['accuracy'])} (n={v['n']})" for c, v in test["by_corpus"].items()) + ".",
                f"Per-speaker accuracy: min {pct(test['per_speaker_accuracy']['min'])}, median {pct(test['per_speaker_accuracy']['median'])}, max {pct(test['per_speaker_accuracy']['max'])} over {test['speakers']} speakers.",
                "", "![Confusion matrix](eval/results/emotion_test.png)", ""]

    cross = load("emotion_cross_corpus_tess.json")
    out += ["### Cross-corpus generalisation", ""]
    if cross:
        out += [f"A second model trained without TESS, then scored on all {cross['n']} five-class TESS clips (2 speakers it never heard, a corpus it never saw): "
                f"accuracy {pct(cross['accuracy'])} ({ci(cross['accuracy_ci95'])}), macro-F1 {pct(cross['macro_f1'])}. Source: `emotion_cross_corpus_tess.json`.", ""]
    else:
        out += [not_run("emotion_cross_corpus_tess.json"), ""]

    ens = load("emotion_ensemble.json")
    out += ["### Emotion2Vec ensemble", ""]
    if ens:
        t = ens["test"]
        out += [
            table(["Wav2Vec2 weight", "Val accuracy", "Val macro-F1"], [[r["w2v_weight"], pct(r["val_accuracy"]), pct(r["val_macro_f1"])] for r in ens["val_sweep"]]),
            "",
            f"Weight 1.0 is Wav2Vec2 alone and 0.0 is Emotion2Vec alone. Best on validation: weight {ens['best_on_val']['w2v_weight']}, "
            f"macro-F1 gain over Wav2Vec2 alone {100 * ens['val_gain_over_w2v_alone']:+.1f} points (95% CI {100 * ens['val_gain_ci95'][0]:+.1f} to {100 * ens['val_gain_ci95'][1]:+.1f}).",
            f"Rule: {ens['decision_rule']}. Decision: `ENABLE_EMOTION2VEC={'true' if ens['enable_emotion2vec'] else 'false'}`.",
            "",
            table(["On test", "Accuracy", "Macro-F1"],
                  [["Wav2Vec2 alone", pct(t["wav2vec2_alone"]["accuracy"]), pct(t["wav2vec2_alone"]["macro_f1"])],
                   ["Emotion2Vec alone", pct(t["emotion2vec_alone"]["accuracy"]), pct(t["emotion2vec_alone"]["macro_f1"])],
                   [f"Ensemble at best validation weight ({ens['best_on_val']['w2v_weight']})", pct(t["best_val_weight_for_reference"]["accuracy"]), pct(t["best_val_weight_for_reference"]["macro_f1"])]]),
            "", f"Caveat: {ens['caveat']} Source: `emotion_ensemble.json`.", "",
        ]
    else:
        out += [not_run("emotion_ensemble.json"), ""]
    return "\n".join(out)


def onnx_section() -> str:
    out = ["## 2. ONNX export and INT8 quantisation", ""]
    b = load("onnx_benchmark.json")
    if not b:
        return "\n".join(out + [not_run("onnx_benchmark.json"), ""])
    names = {"pytorch_fp32": "PyTorch FP32", "onnx_fp32": "ONNX FP32", "onnx_int8": "ONNX INT8"}
    out += [
        table(["Version", "Size (MB)", "Accuracy", "Macro-F1", "CPU p50 (ms)", "CPU p95 (ms)"],
              [[names[k], v["size_mb"], pct(v["accuracy"]), pct(v["macro_f1"]), v["latency_per_3s_clip"]["p50_ms"], v["latency_per_3s_clip"]["p95_ms"]] for k, v in b["variants"].items()]),
        "",
        f"INT8 against PyTorch: {b['int8_vs_pytorch']['size_reduction_pct']}% smaller ({b['int8_vs_pytorch']['size_ratio']}x), "
        f"{b['int8_vs_pytorch']['p50_speedup']}x the p50 speed, accuracy change {100 * b['int8_vs_pytorch']['accuracy_change']:+.1f} points, "
        f"macro-F1 change {100 * b['int8_vs_pytorch']['macro_f1_change']:+.1f} points.",
        f"Latency is one 3-second clip, batch 1, CPU, on {b['machine']}. {b['notes']} Source: `onnx_benchmark.json`.", "",
    ]
    return "\n".join(out)


def memory_section() -> str:
    out = ["## 3. Memory retrieval (RAG)", ""]
    m = load("memory_eval.json")
    if not m:
        return "\n".join(out + [not_run("memory_eval.json"), ""])
    t = m["test"]
    out += [
        f"{m['users']} synthetic users, {m['memories']} stored memories, {m['followup_cases']} paraphrased follow-ups and {m['unrelated_cases']} unrelated questions. "
        f"Embedding: {m['embedding_model']}, {m['distance']} distance, top-{m['top_k']}.",
        "",
        "Threshold sweep on the development half:", "",
        table(["MEMORY_MAX_DISTANCE", "Hit@3", "False-recall rate", "Balanced score"],
              [[r["threshold"], pct(r["hit_at_3"]), pct(r["false_recall_rate"]), r["balanced_score"]] for r in m["dev_sweep"]]),
        "",
        f"Chosen value: **{m['chosen_max_distance']}** ({m['selection_rule']}). Final numbers on the held-out half:", "",
        table(["Metric", "Value"],
              [["Hit@3", f"{pct(t['hit_at_3'])} ({t['n_followups']} follow-ups)"],
               ["Hit@1", pct(t["hit_at_1"])],
               ["False-recall rate", f"{pct(t['false_recall_rate'])} ({t['n_unrelated']} unrelated questions)"],
               ["Cross-user leaks", f"{m['cross_user_leaks']['leaks']} of {m['cross_user_leaks']['results_checked']} results, threshold wide open"],
               ["Current-session exclusion violations", f"{m['current_session_exclusion']['violations']} of {m['current_session_exclusion']['memories_checked']}"],
               ["Retrieval latency p50 / p95", f"{m['latency_ms']['p50']} ms / {m['latency_ms']['p95']} ms"]]),
        "", f"Caveat: {m['caveat']} Source: `memory_eval.json`.", "",
    ]
    return "\n".join(out)


def latency_section() -> str:
    out = ["## 4. End-to-end latency", ""]
    labels = {"decode": "Audio decode", "stt": "Speech-to-text", "emotion": "Emotion", "retrieval": "Memory retrieval",
              "llm_first_token": "LLM time to first token", "llm_total": "LLM full reply",
              "time_to_first_token_end_to_end": "Upload to first reply token", "total": "Total"}
    titles = {"local": "Local mode (default settings)", "local_w2v_only": "Local mode with Emotion2Vec switched off", "cloud": "Cloud mode"}
    for mode in titles:
        d = load(f"latency_{mode}.json")
        out += [f"### {titles[mode]}", ""]
        if not d:
            out += [not_run(f"latency_{mode}.json", "Cloud mode needs a Groq API key." if mode == "cloud" else ""), ""]
            continue
        out += [
            f"LLM `{d['llm']}`, STT `{d['stt']}`, emotion backend `{d['emotion_backend']}`, {d['clips']} clips ({d['clip_source']}).", "",
            table(["Stage", "p50 (ms)", "p95 (ms)"], [[labels[k], v["p50_ms"], v["p95_ms"]] for k, v in d["stages"].items()]),
            "", f"{d['notes']} Machine: {d['machine']}. Source: `latency_{mode}.json`.", "",
        ]
        if "local_stt_comparison" in d:
            c = d["local_stt_comparison"]
            out += ["Local Whisper engines, same clips:", "",
                    table(["Engine", "Model", "p50 (ms)", "p95 (ms)"], [[k, v["model"], v["p50_ms"], v["p95_ms"]] for k, v in c.items()]), ""]
    return "\n".join(out)


def knowledge_and_judge_sections() -> str:
    out = ["## 5. Coping-skills knowledge base (off by default)", ""]
    k = load("knowledge_eval.json")
    if k:
        t = k["test"]
        out += [
            f"{k['notes']} short notes, same embedding model as memory. Threshold swept on half the messages; chosen `KNOWLEDGE_MAX_DISTANCE={k['chosen_max_distance']}`.",
            "",
            table(["Held-out half", "Value"], [["Correct note retrieved (top-1)", f"{pct(t['top1_hit_rate'])} of {t['n_relevant']}"],
                                               ["Anything retrieved for unrelated messages", f"{pct(t['false_retrieval_rate'])} of {t['n_unrelated']}"]]),
            "", f"Caveat: {k['caveat']} Source: `knowledge_eval.json`.", "",
        ]
    else:
        out += [not_run("knowledge_eval.json"), ""]

    out += ["## 6. Reply quality (automated LLM-as-judge, not human evaluation)", ""]
    j, j1 = load("judge_eval.json"), load("judge_eval_prompt_v1.json")
    if not j:
        return "\n".join(out + [not_run("judge_eval.json"), ""])
    rc = j["rule_checks"]
    rows = []
    if j1:
        rows.append(["First run (before fixes)", *[j1["mean_scores"][c] for c in ("empathy", "relevance", "memory_use", "safety")], "-", "-", "-"])
    rows.append(["Current", *[j["mean_scores"][c] for c in ("empathy", "relevance", "memory_use", "safety")],
                 f"{rc['memory_referenced_when_relevant'][0]}/{rc['memory_referenced_when_relevant'][1]}",
                 f"{rc['no_invented_memory_when_irrelevant'][0]}/{rc['no_invented_memory_when_irrelevant'][1]}",
                 f"{rc['required_referral_present'][0]}/{rc['required_referral_present'][1]}"])
    out += [
        f"{j['conversations']} scripted conversations, replies from `{j['reply_model']}`, judged by `{j['judge_model']}` on a 1-5 rubric.",
        "",
        table(["Run", "Empathy", "Relevance", "Memory use", "Safety", "Memory referenced (rule)", "No invented memory (rule)", "Referral present (rule)"], rows),
        "",
        "**How to read this.** The judge is the same model that wrote the replies and scores almost everything 4-5, including replies that the "
        "keyword rules show are missing a referral or ignore a relevant memory. The judge scores are therefore not informative; the rule checks are cruder but independent.",
        "",
        "What the first run exposed, and what changed: an indirect crisis phrasing (\"everyone would be fine if I just disappeared\") got an ordinary LLM reply, "
        "so crisis patterns were added; medication and diagnosis questions got no referral, so a fixed referral line is now appended for them. "
        f"{j['history_note']}",
        "",
        f"Still weak: the model referenced a relevant earlier-session memory in {rc['memory_referenced_when_relevant'][0]} of {rc['memory_referenced_when_relevant'][1]} cases, although retrieval injected it every time. Source: `judge_eval.json`, `judge_eval_prompt_v1.json`.",
        "",
    ]
    return "\n".join(out)


def other_sections() -> str:
    out = ["## 7. Cloud container footprint", ""]
    c = load("cloud_memory.json")
    out += [json.dumps(c, indent=2) if c else not_run("cloud_memory.json", "Docker was not installed on the development machine."), ""]
    out += ["## 8. Training time", ""]
    rows, total = [], 0.0
    for path in sorted(RESULTS.glob("training_*.json")):
        d = json.loads(path.read_text())
        if "train_minutes" in d:
            rows.append([d["run"], d["epochs_completed"], d["best_epoch"], d["train_minutes"], d["status"], f"`{path.name}`"])
            total += d["train_minutes"]
        elif "benchmark_wall_seconds" in d:
            rows.append([f"{d['run']} (timing benchmark)", "-", "-", round(d["benchmark_wall_seconds"] / 60, 2), "benchmark", f"`{path.name}`"])
            total += d["benchmark_wall_seconds"] / 60
    if rows:
        out += [table(["Run", "Epochs", "Best epoch", "Minutes", "Status", "Source"], rows), "",
                f"Total training wall-clock: **{total:.1f} minutes** ({total / 60:.2f} h) against a budget of 4.5 h. Includes per-epoch validation.", ""]
    else:
        out += ["_No training logs found._", ""]
    return "\n".join(out)


def main() -> None:
    doc = "\n".join([
        "# Results",
        "",
        "Generated by `python -m eval.make_results` from the JSON files in `eval/results/`. Do not edit by hand; rerun the eval scripts instead.",
        "",
        emotion_section(), onnx_section(), memory_section(), latency_section(), knowledge_and_judge_sections(), other_sections(),
    ])
    (ROOT / "RESULTS.md").write_text(doc)
    print(f"wrote RESULTS.md ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
