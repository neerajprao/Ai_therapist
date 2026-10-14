"""Classification metrics and the confusion-matrix figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from soulsync.emotion import LABELS


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1, confusion matrix and bootstrap 95% CIs.

    The bootstrap resamples clips, so it reflects test-set size only. It does
    not capture run-to-run (seed) variance or the small number of speakers.
    """
    y_pred = probs.argmax(axis=1)
    ids = list(range(len(LABELS)))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=ids, average="macro", zero_division=0))
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)), 4),
        "accuracy_ci95": [round(float(np.percentile(accs, q)), 4) for q in (2.5, 97.5)],
        "macro_f1_ci95": [round(float(np.percentile(f1s, q)), 4) for q in (2.5, 97.5)],
        "per_class": {
            LABELS[i]: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(support[i]),
            }
            for i in ids
        },
        "confusion_matrix": {"labels": list(LABELS), "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=ids).tolist()},
    }


def quick_scores(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    y_pred = probs.argmax(axis=1)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
    )


def save_confusion_png(metrics: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"]["rows_true_cols_pred"])
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\naccuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}, n={metrics['n']}", fontsize=10)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "#1e293b", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
