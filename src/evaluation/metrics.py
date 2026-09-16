"""
Shared metric computation so every classifier is scored identically.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 1000, seed: int = 42, ci: float = 0.95) -> dict:
    """
    Non-parametric bootstrap confidence interval for any metric_fn(y_true, y_pred) -> float.
    Resamples (with replacement) pairs of (true, pred) n_boot times.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    lower = float(np.percentile(stats, (1 - ci) / 2 * 100))
    upper = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return {"point_estimate": float(metric_fn(y_true, y_pred)), "ci_lower": lower, "ci_upper": upper,
            "ci_level": ci, "n_boot": n_boot, "n_examples": n}


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels) if labels is not None else None,
        "per_class_report": report,
    }
