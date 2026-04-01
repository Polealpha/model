from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


LABEL_ORDER = ["immediate", "delay", "none"]


def compute_timing_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_score: np.ndarray | None = None,
) -> Dict[str, float]:
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_score is not None and len(set(y_true)) > 1:
        y_true_ovr = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=float)
        for row_idx, label in enumerate(y_true):
            y_true_ovr[row_idx, LABEL_ORDER.index(label)] = 1.0
        try:
            metrics["macro_auroc"] = float(
                roc_auc_score(y_true_ovr, y_score, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["macro_auroc"] = float("nan")
    return metrics


def false_interrupt_rate(
    decisions: Sequence[str],
    busy_flags: Sequence[bool],
    privacy_flags: Sequence[bool],
) -> float:
    total = 0
    violations = 0
    for decision, busy, privacy in zip(decisions, busy_flags, privacy_flags):
        if busy or privacy:
            total += 1
            if decision == "immediate":
                violations += 1
    return 0.0 if total == 0 else float(violations / total)


def summarize_prediction_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(rows)
    y_true = [str(row["timing_label"]) for row in rows]
    y_pred = [str(row["prediction"]) for row in rows]
    busy = [bool(row.get("busy_speaking", False)) for row in rows]
    privacy = [bool(row.get("privacy_on", False)) for row in rows]
    metrics = compute_timing_metrics(y_true, y_pred)
    metrics["false_interrupt_rate"] = false_interrupt_rate(y_pred, busy, privacy)
    return metrics

