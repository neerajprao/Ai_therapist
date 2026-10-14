"""Evaluate memory retrieval with the real embedding model (MiniLM-L6-v2 ONNX).

  python -m eval.memory_eval

Protocol:
  * all memories for all users go into one collection, as in production;
  * cases are split in half (even index = dev, odd index = test);
  * MEMORY_MAX_DISTANCE is swept from 0.30 to 0.80 on dev only and the value
    with the best balanced score, (Hit@3 + (1 - false-recall)) / 2, is chosen;
  * the chosen value is reported once on test.

Writes eval/results/memory_eval.json and .log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from soulsync.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "eval" / "data" / "memory_cases.json"
RESULTS = ROOT / "eval" / "results"
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]
TOP_K = 3


def score(store: MemoryStore, text_to_id: dict, followups: list, unrelated: list, threshold: float) -> dict:
    hit3 = hit1 = 0
    for case in followups:
        got = [text_to_id[m.text] for m in store.search(case["user"], "now", case["query"], TOP_K, threshold)]
        hit3 += case["target"] in got
        hit1 += bool(got) and got[0] == case["target"]
    false = sum(bool(store.search(c["user"], "now", c["query"], TOP_K, threshold)) for c in unrelated)
    h3, fr = hit3 / len(followups), false / len(unrelated)
    return {
        "threshold": threshold,
        "hit_at_3": round(h3, 4),
        "hit_at_1": round(hit1 / len(followups), 4),
        "false_recall_rate": round(fr, 4),
        "balanced_score": round((h3 + 1 - fr) / 2, 4),
        "n_followups": len(followups),
        "n_unrelated": len(unrelated),
    }


def main() -> None:
    data = json.loads(CASES.read_text())
    store = MemoryStore(None, top_k=TOP_K)
    stale = store.collection.get(include=[])["ids"]
    if stale:
        store.collection.delete(ids=stale)

    text_to_id, owner = {}, {}
    for user, memories in data["users"].items():
        for m in memories:
            assert store.add(user, m["session"], m["text"]), m
            text_to_id[m["text"]] = m["id"]
            owner[m["text"]] = user

    followups, unrelated = data["followups"], data["unrelated"]
    dev_f, test_f = followups[0::2], followups[1::2]
    dev_u, test_u = unrelated[0::2], unrelated[1::2]

    sweep = [score(store, text_to_id, dev_f, dev_u, t) for t in THRESHOLDS]
    best = max(sweep, key=lambda r: (r["balanced_score"], -r["threshold"]))
    chosen = best["threshold"]
    final = score(store, text_to_id, test_f, test_u, chosen)

    # Cross-user leaks: every query, with the threshold wide open (worst case).
    leaks = checked = 0
    for case in followups + unrelated:
        for m in store.search(case["user"], "now", case["query"], top_k=10, max_distance=2.0):
            checked += 1
            leaks += owner[m.text] != case["user"]

    # Current-session exclusion: ask for a memory verbatim from its own session.
    violations = recalled_elsewhere = total = 0
    for user, memories in data["users"].items():
        for m in memories:
            total += 1
            same = store.search(user, m["session"], m["text"], top_k=10, max_distance=2.0)
            violations += any(x.session_id == m["session"] for x in same)
            other = store.search(user, "now", m["text"], top_k=1, max_distance=chosen)
            recalled_elsewhere += bool(other) and other[0].text == m["text"]

    # Latency of one retrieval (embed the query + filtered vector search).
    queries = [(c["user"], c["query"]) for c in followups + unrelated]
    for user, q in queries[:5]:
        store.search(user, "now", q)  # warm-up
    times = []
    for _ in range(3):
        for user, q in queries:
            t0 = time.perf_counter()
            store.search(user, "now", q, TOP_K, chosen)
            times.append((time.perf_counter() - t0) * 1000)

    result = {
        "embedding_model": "all-MiniLM-L6-v2 (ONNX, ChromaDB default)",
        "distance": "cosine",
        "top_k": TOP_K,
        "users": len(data["users"]),
        "memories": len(text_to_id),
        "followup_cases": len(followups),
        "unrelated_cases": len(unrelated),
        "dev_sweep": sweep,
        "chosen_max_distance": chosen,
        "selection_rule": "max balanced_score on dev; ties go to the smaller threshold",
        "test": final,
        "cross_user_leaks": {"leaks": leaks, "results_checked": checked, "threshold": 2.0, "top_k": 10},
        "current_session_exclusion": {
            "violations": violations,
            "memories_checked": total,
            "same_text_recalled_from_another_session": recalled_elsewhere,
        },
        "latency_ms": {
            "p50": round(float(np.percentile(times, 50)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
            "n": len(times),
            "note": "single retrieval on CPU, in-memory collection of 30 memories",
        },
        "caveat": "Small synthetic set written by the project author; half of 42 follow-ups and 16 unrelated "
        "questions are used for the final numbers, so each test case moves Hit@3 by about 5 points.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "memory_eval.json").write_text(json.dumps(result, indent=2))
    log = "\n".join(
        [f"dev  t={r['threshold']:.2f} hit@3={r['hit_at_3']:.3f} false_recall={r['false_recall_rate']:.3f} balanced={r['balanced_score']:.3f}" for r in sweep]
        + [
            f"chosen MEMORY_MAX_DISTANCE = {chosen}",
            f"test hit@3={final['hit_at_3']} hit@1={final['hit_at_1']} false_recall={final['false_recall_rate']} "
            f"(n={final['n_followups']} follow-ups, {final['n_unrelated']} unrelated)",
            f"cross-user leaks: {leaks} of {checked} results",
            f"current-session violations: {violations} of {total}",
            f"latency ms p50={result['latency_ms']['p50']} p95={result['latency_ms']['p95']}",
        ]
    )
    (RESULTS / "memory_eval.log").write_text(log + "\n")
    print(log)


if __name__ == "__main__":
    main()
