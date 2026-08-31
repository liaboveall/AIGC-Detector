"""Verify that packaged-checkpoint predictions match a direct ensemble sweep."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


KEY_COLUMNS = ("path", "condition")
STATIC_COLUMNS = ("dataset", "source_class", "label")
EXPECTED_COLUMNS = ["path", "dataset", "source_class", "label", "condition", "probability"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", required=True, help="Predictions from the live two-model sweep")
    parser.add_argument("--packaged", required=True, help="Predictions from the packaged checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct-metrics", help="Optional direct-sweep evaluation JSON")
    parser.add_argument("--packaged-metrics", help="Optional packaged-checkpoint evaluation JSON")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute probability tolerance for CUDA execution-path rounding (default: 1e-3)",
    )
    parser.add_argument("--metric-atol", type=float, default=1e-5)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise RuntimeError(f"Unexpected prediction schema in {path}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            key = tuple(row[column] for column in KEY_COLUMNS)
            if key in rows:
                raise RuntimeError(f"Duplicate path/condition key in {path} at line {line_number}: {key!r}")
            rows[key] = row
    return rows


def main() -> None:
    args = parse_args()
    if args.atol < 0 or args.metric_atol < 0:
        raise ValueError("tolerances must be non-negative")
    if bool(args.direct_metrics) != bool(args.packaged_metrics):
        raise ValueError("--direct-metrics and --packaged-metrics must be supplied together")

    direct_path = Path(args.direct).resolve()
    packaged_path = Path(args.packaged).resolve()
    output_path = Path(args.output).resolve()
    for path in (direct_path, packaged_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    direct_rows = load_predictions(direct_path)
    packaged_rows = load_predictions(packaged_path)
    direct_keys = set(direct_rows)
    packaged_keys = set(packaged_rows)
    if direct_keys != packaged_keys:
        raise RuntimeError(
            "Prediction key sets differ: "
            f"direct_only={len(direct_keys - packaged_keys)}, "
            f"packaged_only={len(packaged_keys - direct_keys)}"
        )

    count = len(direct_rows)
    max_absolute_difference = 0.0
    sum_absolute_difference = 0.0
    mismatches = 0
    for key in sorted(direct_keys):
        direct_row = direct_rows[key]
        packaged_row = packaged_rows[key]
        direct_static = tuple(direct_row[column] for column in STATIC_COLUMNS)
        packaged_static = tuple(packaged_row[column] for column in STATIC_COLUMNS)
        if direct_static != packaged_static:
            raise RuntimeError(
                f"Static prediction fields differ for {key!r}: "
                f"direct={direct_static!r}, packaged={packaged_static!r}"
            )
        direct_probability = float(direct_row["probability"])
        packaged_probability = float(packaged_row["probability"])
        if not math.isfinite(direct_probability) or not math.isfinite(packaged_probability):
            raise RuntimeError(f"Non-finite probability for {key!r}")
        difference = abs(direct_probability - packaged_probability)
        max_absolute_difference = max(max_absolute_difference, difference)
        sum_absolute_difference += difference
        mismatches += int(difference > args.atol)

    metric_equivalence = None
    metric_mismatches = 0
    if args.direct_metrics:
        direct_metrics_path = Path(args.direct_metrics).resolve()
        packaged_metrics_path = Path(args.packaged_metrics).resolve()
        direct_metrics = json.loads(direct_metrics_path.read_text(encoding="utf-8"))
        packaged_metrics = json.loads(packaged_metrics_path.read_text(encoding="utf-8"))
        direct_conditions = direct_metrics["conditions"]
        packaged_conditions = packaged_metrics["conditions"]
        if set(direct_conditions) != set(packaged_conditions):
            raise RuntimeError("Evaluation JSON condition sets differ")
        condition_auc_differences = {
            condition: abs(
                float(direct_conditions[condition]["overall"]["roc_auc"])
                - float(packaged_conditions[condition]["overall"]["roc_auc"])
            )
            for condition in direct_conditions
        }
        robust_difference = abs(
            float(direct_metrics["robustness"]["robust_score"])
            - float(packaged_metrics["robustness"]["robust_score"])
        )
        maximum_auc_difference = max(condition_auc_differences.values(), default=0.0)
        metric_mismatches = int(robust_difference > args.metric_atol) + int(
            maximum_auc_difference > args.metric_atol
        )
        metric_equivalence = {
            "direct": str(direct_metrics_path),
            "packaged": str(packaged_metrics_path),
            "absolute_tolerance": args.metric_atol,
            "robust_score_difference": robust_difference,
            "max_condition_auc_difference": maximum_auc_difference,
            "conditions_over_tolerance": sum(
                difference > args.metric_atol for difference in condition_auc_differences.values()
            ),
        }

    passed = mismatches == 0 and metric_mismatches == 0
    result = {
        "status": "PASS" if passed else "FAIL",
        "direct": str(direct_path),
        "packaged": str(packaged_path),
        "rows": count,
        "key_alignment_exact": True,
        "static_fields_exact": True,
        "absolute_tolerance": args.atol,
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": sum_absolute_difference / count if count else 0.0,
        "rows_over_tolerance": mismatches,
        "metric_equivalence": metric_equivalence,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
