"""Reconstruct standard condition metrics from an evaluation prediction CSV."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import binary_metrics, grouped_metrics, robustness_summary, source_contrast_metrics


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = resolve(args.predictions)
    frame = pd.read_csv(prediction_path)
    required = {"path", "dataset", "source_class", "label", "condition", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
    if frame.duplicated(["path", "condition"]).any():
        raise ValueError("prediction file contains duplicate path/condition keys")
    if not frame["probability"].map(math.isfinite).all():
        raise ValueError("prediction file contains non-finite probabilities")
    conditions = {}
    for condition, group in frame.groupby("condition", sort=False):
        labels = group["label"].astype(int).tolist()
        probabilities = group["probability"].astype(float).tolist()
        sources = group["source_class"].astype(str).tolist()
        datasets = group["dataset"].astype(str).tolist()
        conditions[str(condition)] = {
            "loss": math.nan,
            "overall": binary_metrics(labels, probabilities),
            "by_source_class": grouped_metrics(labels, probabilities, sources),
            "source_contrasts": source_contrast_metrics(labels, probabilities, sources),
            "by_dataset": grouped_metrics(labels, probabilities, datasets),
        }
    payload = {
        "predictions": str(prediction_path),
        "rows": len(frame),
        "conditions": conditions,
        "robustness": robustness_summary(conditions),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output} robust={payload['robustness']['robust_score']:.6f}")


if __name__ == "__main__":
    main()
