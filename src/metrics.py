from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else math.nan


def binary_metrics(labels: Iterable[float], probabilities: Iterable[float]) -> dict[str, float | int]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    probabilities_array = np.asarray(list(probabilities), dtype=np.float64)
    predictions = (probabilities_array >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1])
    has_both_classes = len(np.unique(labels_array)) == 2
    return {
        "count": int(len(labels_array)),
        "roc_auc": _safe_auc(labels_array, probabilities_array),
        "average_precision": (
            float(average_precision_score(labels_array, probabilities_array)) if has_both_classes else math.nan
        ),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def grouped_metrics(
    labels: list[float], probabilities: list[float], groups: list[str]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    groups_array = np.asarray(groups)
    for group in sorted(set(groups)):
        mask = groups_array == group
        result[group] = binary_metrics(labels_array[mask], probabilities_array[mask])
    return result


def source_contrast_metrics(
    labels: list[float], probabilities: list[float], source_classes: list[str]
) -> dict[str, dict[str, float | int]]:
    """Compare each positive source class against all real examples.

    AUC within a pure source class is undefined because that group has only one
    label. These contrasts provide the useful full-synthetic-vs-real and
    tampered-vs-real measurements used for error analysis.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities)
    sources_array = np.asarray(source_classes)
    result: dict[str, dict[str, float | int]] = {}
    positive_sources = sorted(set(sources_array[labels_array == 1].tolist()))
    for source in positive_sources:
        mask = (labels_array == 0) | (sources_array == source)
        result[f"real_vs_{source}"] = binary_metrics(labels_array[mask], probabilities_array[mask])
    return result


def robustness_summary(condition_metrics: dict[str, dict]) -> dict[str, float]:
    degraded = [
        float(metrics["overall"]["roc_auc"])
        for name, metrics in condition_metrics.items()
        if name != "clean" and math.isfinite(float(metrics["overall"]["roc_auc"]))
    ]
    if not degraded:
        clean = float(condition_metrics.get("clean", {}).get("overall", {}).get("roc_auc", math.nan))
        degraded = [clean] if math.isfinite(clean) else []
    if not degraded:
        return {"mean_degraded_auc": math.nan, "worst_degraded_auc": math.nan, "robust_score": math.nan}
    mean_auc = float(np.mean(degraded))
    worst_auc = float(np.min(degraded))
    return {
        "mean_degraded_auc": mean_auc,
        "worst_degraded_auc": worst_auc,
        "robust_score": 0.8 * mean_auc + 0.2 * worst_auc,
    }
