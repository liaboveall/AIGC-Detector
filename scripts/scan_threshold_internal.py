"""Threshold rescan on internal selection predictions only (no confirmation, no WildFake).

Objective mirrors scripts/analyze_wildfake.py: mean condition-balanced accuracy,
ties broken by worst condition then proximity to 0.5.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_THRESHOLD = 0.209


def condition_balanced_accuracy(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    pos = labels == 1
    neg = labels == 0
    tpr = float(np.sum(pos & pred) / max(int(pos.sum()), 1))
    tnr = float(np.sum(neg & ~pred) / max(int(neg.sum()), 1))
    return (tpr + tnr) / 2


def scan(frame: pd.DataFrame) -> dict:
    grouped = {
        name: (g["label"].to_numpy(np.int64), g["probability"].to_numpy(np.float64))
        for name, g in frame.groupby("condition", sort=True)
    }
    thresholds = np.linspace(0.01, 0.99, 981)
    best_key = None
    best_threshold = 0.5
    best_scores: dict[str, float] = {}
    history: list[tuple[float, float]] = []
    for threshold in thresholds:
        scores = {
            name: condition_balanced_accuracy(labels, probs, float(threshold))
            for name, (labels, probs) in grouped.items()
        }
        mean_score = float(np.mean(list(scores.values())))
        worst_score = float(np.min(list(scores.values())))
        history.append((float(threshold), mean_score))
        key = (mean_score, worst_score, -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_scores = scores
    current_scores = {
        name: condition_balanced_accuracy(labels, probs, CURRENT_THRESHOLD)
        for name, (labels, probs) in grouped.items()
    }
    return {
        "best_threshold": best_threshold,
        "best_mean_balanced_accuracy": float(np.mean(list(best_scores.values()))),
        "best_worst_balanced_accuracy": float(np.min(list(best_scores.values()))),
        "current_threshold": CURRENT_THRESHOLD,
        "current_mean_balanced_accuracy": float(np.mean(list(current_scores.values()))),
        "current_worst_balanced_accuracy": float(np.min(list(current_scores.values()))),
        "best_by_condition": best_scores,
        "current_by_condition": current_scores,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/model_soup_v1/predictions_beta_0p50.csv")
    parser.add_argument("--output", default="outputs/acceptance/threshold_scan_selection.json")
    args = parser.parse_args()
    frame = pd.read_csv(Path(args.predictions))
    result = scan(frame)
    history = result.pop("history")
    pd.DataFrame(history, columns=["threshold", "mean_balanced_accuracy"]).to_csv(
        Path(args.output).with_suffix(".csv"), index=False
    )
    import json

    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"best_threshold={result['best_threshold']:.3f} "
        f"mean_bacc={result['best_mean_balanced_accuracy']:.6f} | "
        f"current 0.209 mean_bacc={result['current_mean_balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
