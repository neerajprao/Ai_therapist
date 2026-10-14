"""Tune and evaluate the coping-skills knowledge base threshold.

  python -m eval.knowledge_eval

Same protocol as eval/memory_eval.py: sweep KNOWLEDGE_MAX_DISTANCE from 0.30
to 0.80 on the even-indexed cases, pick the best balanced score
((top-1 hit rate + (1 - false-retrieval rate)) / 2, ties to the smaller
threshold), and report on the odd-indexed cases.
Writes eval/results/knowledge_eval.json and .log.
"""

from __future__ import annotations

import json
from pathlib import Path

from soulsync.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]


def score(kb: KnowledgeBase, relevant: list, unrelated: list, t: float) -> dict:
    hits = sum(bool(r := kb.search(c["query"], 1, t)) and r[0].id == c["target"] for c in relevant)
    false = sum(bool(kb.search(c["query"], 1, t)) for c in unrelated)
    h, f = hits / len(relevant), false / len(unrelated)
    return {"threshold": t, "top1_hit_rate": round(h, 4), "false_retrieval_rate": round(f, 4),
            "balanced_score": round((h + 1 - f) / 2, 4), "n_relevant": len(relevant), "n_unrelated": len(unrelated)}


def main() -> None:
    cases = json.loads((ROOT / "eval" / "data" / "knowledge_cases.json").read_text())
    kb = KnowledgeBase(None)
    rel, unr = cases["relevant"], cases["unrelated"]
    sweep = [score(kb, rel[0::2], unr[0::2], t) for t in THRESHOLDS]
    best = max(sweep, key=lambda r: (r["balanced_score"], -r["threshold"]))
    final = score(kb, rel[1::2], unr[1::2], best["threshold"])
    result = {
        "notes": kb.collection.count(),
        "embedding_model": "all-MiniLM-L6-v2 (ONNX, ChromaDB default)",
        "dev_sweep": sweep,
        "chosen_max_distance": best["threshold"],
        "test": final,
        "caveat": "Very small synthetic set (9 relevant and 5 unrelated messages in the held-out half), written by the "
        "project author. Treat it as a sanity check of the threshold, not a measurement worth quoting.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "knowledge_eval.json").write_text(json.dumps(result, indent=2))
    log = "\n".join(
        [f"dev t={r['threshold']:.2f} hit={r['top1_hit_rate']:.3f} false={r['false_retrieval_rate']:.3f} bal={r['balanced_score']:.3f}" for r in sweep]
        + [f"chosen KNOWLEDGE_MAX_DISTANCE = {best['threshold']}", f"test: {final}"]
    )
    (RESULTS / "knowledge_eval.log").write_text(log + "\n")
    print(log)


if __name__ == "__main__":
    main()
