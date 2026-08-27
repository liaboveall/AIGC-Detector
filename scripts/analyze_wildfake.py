from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PREDICTION_COLUMNS = {
    "path",
    "dataset",
    "source_class",
    "label",
    "condition",
    "probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a threshold without WildFake leakage and build the final error report"
    )
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--full-evaluation", required=True)
    parser.add_argument("--output-dir", default="reports/wildfake_analysis")
    parser.add_argument("--error-cases-per-type", type=int, default=12)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["label"] = frame["label"].astype(np.int64)
    frame["probability"] = frame["probability"].astype(np.float64)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{path} contains labels outside {{0, 1}}")
    if not frame["probability"].between(0.0, 1.0).all():
        raise ValueError(f"{path} contains probabilities outside [0, 1]")
    duplicate_keys = frame.duplicated(["path", "condition"])
    if duplicate_keys.any():
        raise ValueError(f"{path} contains duplicate path/condition rows")
    return frame


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(labels), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    has_both_classes = len(np.unique(labels)) == 2
    return {
        "count": int(len(labels)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else float("nan"),
        "average_precision": (
            float(average_precision_score(labels, probabilities))
            if has_both_classes
            else float("nan")
        ),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_threshold(frame: pd.DataFrame) -> tuple[float, dict]:
    grouped = {
        name: (
            group["label"].to_numpy(dtype=np.int64),
            group["probability"].to_numpy(dtype=np.float64),
        )
        for name, group in frame.groupby("condition", sort=True)
    }
    if not grouped:
        raise ValueError("Calibration predictions are empty")
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_condition_scores: dict[str, float] = {}
    for threshold in np.linspace(0.01, 0.99, 981):
        scores: dict[str, float] = {}
        for name, (labels, probabilities) in grouped.items():
            predictions = probabilities >= threshold
            true_positive_rate = float(
                np.sum((labels == 1) & predictions) / max(np.sum(labels == 1), 1)
            )
            true_negative_rate = float(
                np.sum((labels == 0) & ~predictions) / max(np.sum(labels == 0), 1)
            )
            scores[name] = (true_positive_rate + true_negative_rate) / 2
        mean_score = float(np.mean(list(scores.values())))
        worst_score = float(np.min(list(scores.values())))
        key = (mean_score, worst_score, -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_condition_scores = scores
    return best_threshold, {
        "objective": "mean condition-balanced accuracy; ties use worst condition then proximity to 0.5",
        "conditions": sorted(grouped),
        "mean_balanced_accuracy": float(np.mean(list(best_condition_scores.values()))),
        "worst_balanced_accuracy": float(np.min(list(best_condition_scores.values()))),
        "by_condition": best_condition_scores,
    }


def comparison_rows(frame: pd.DataFrame, calibrated_threshold: float) -> list[dict]:
    rows: list[dict] = []
    for condition, group in frame.groupby("condition", sort=False):
        for threshold_name, threshold in (("default_0.5", 0.5), ("calibrated", calibrated_threshold)):
            rows.append(
                {
                    "condition": condition,
                    "threshold_name": threshold_name,
                    **threshold_metrics(group, threshold),
                }
            )
    return rows


def select_error_cases(
    frame: pd.DataFrame, threshold: float, per_type: int
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["prediction"] = (annotated["probability"] >= threshold).astype(np.int64)
    annotated["error_type"] = np.where(
        (annotated["label"] == 0) & (annotated["prediction"] == 1),
        "false_positive",
        np.where(
            (annotated["label"] == 1) & (annotated["prediction"] == 0),
            "false_negative",
            "correct",
        ),
    )
    selected: list[pd.DataFrame] = []
    for condition, group in annotated.groupby("condition", sort=False):
        false_positives = group[group["error_type"] == "false_positive"].sort_values(
            "probability", ascending=False
        )
        false_negatives = group[group["error_type"] == "false_negative"].sort_values(
            "probability", ascending=True
        )
        # WildFake contains repeated content under different collection folders.
        # The hash-like basename is used as a content key so the report shows
        # diverse examples instead of the same image several times.
        false_positives = false_positives.assign(
            content_key=false_positives["path"].map(lambda value: Path(value).name.lower())
        ).drop_duplicates("content_key").head(per_type)
        false_negatives = false_negatives.assign(
            content_key=false_negatives["path"].map(lambda value: Path(value).name.lower())
        ).drop_duplicates("content_key").head(per_type)
        for error_group in (false_positives, false_negatives):
            ranked = error_group.copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            selected.append(ranked)
    if not selected:
        return annotated.iloc[0:0]
    return pd.concat(selected, ignore_index=True)[
        [
            "condition",
            "error_type",
            "rank",
            "path",
            "dataset",
            "source_class",
            "label",
            "prediction",
            "probability",
        ]
    ]


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def build_report(
    threshold: float,
    calibration_summary: dict,
    robustness_rows: list[dict],
    target_rows: list[dict],
    error_cases: pd.DataFrame,
    calibration_path: Path,
    target_path: Path,
) -> str:
    lines = [
        "# WildFake held-out robustness and error analysis",
        "",
        "## Evaluation policy",
        "",
        "WildFake remained held out: it was not used to train the model or choose the threshold. "
        f"The single threshold `{threshold:.3f}` was selected on `{calibration_path.name}` by "
        "maximising mean balanced accuracy across the five internal validation conditions. "
        f"It was then frozen and applied to `{target_path.name}`.",
        "",
        "## Headline robustness",
        "",
        "| Condition | ROC AUC | Average precision | Accuracy @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in robustness_rows:
        lines.append(
            f"| {row['condition']} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} | "
            f"{row['accuracy_at_0_5']:.4f} | {row['recall_at_0_5']:.4f} | {row['f1_at_0_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Threshold comparison on WildFake",
            "",
            "| Condition | Threshold | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target_rows:
        lines.append(
            f"| {row['condition']} | {row['threshold']:.3f} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['specificity']:.4f} | {row['f1']:.4f} | {row['fp']} | {row['fn']} |"
        )
    worst = min(robustness_rows, key=lambda row: row["roc_auc"])
    clean_calibrated = next(
        row
        for row in target_rows
        if row["condition"] == "clean" and row["threshold_name"] == "calibrated"
    )
    blur_calibrated = next(
        row
        for row in target_rows
        if row["condition"] == "blur_2.0" and row["threshold_name"] == "calibrated"
    )
    clean_default = next(
        row
        for row in target_rows
        if row["condition"] == "clean" and row["threshold_name"] == "default_0.5"
    )
    blur_default = next(
        row
        for row in target_rows
        if row["condition"] == "blur_2.0" and row["threshold_name"] == "default_0.5"
    )
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            f"- The weakest ranking condition is `{worst['condition']}` with ROC AUC "
            f"{worst['roc_auc']:.4f}; this is consistent with strong blur removing or distorting "
            "high-frequency forensic cues.",
            f"- At the frozen calibrated threshold, clean false positives/false negatives are "
            f"{clean_calibrated['fp']}/{clean_calibrated['fn']}; under `blur_2.0` they become "
            f"{blur_calibrated['fp']}/{blur_calibrated['fn']}.",
            f"- Calibration improves clean balanced accuracy from {clean_default['balanced_accuracy']:.4f} "
            f"to {clean_calibrated['balanced_accuracy']:.4f}, but under `blur_2.0` it changes from "
            f"{blur_default['balanced_accuracy']:.4f} to {blur_calibrated['balanced_accuracy']:.4f}. "
            "The lower threshold recovers more synthetic images but also increases false positives, so it is not "
            "a universal fix for severe blur.",
            "- JPEG AUC remains high even when fixed-threshold recall falls, indicating score calibration "
            "shift rather than a complete loss of ranking ability.",
            "- The submission interface should continue to emit confidence scores. The calibrated threshold is "
            "for binary demo decisions and is not baked into the score output.",
            "- Representative high-confidence errors are listed below and fully exported in `error_cases.csv`.",
            "",
        ]
    )
    for (condition, error_type), group in error_cases.groupby(
        ["condition", "error_type"], sort=False
    ):
        lines.append(f"### {condition}: {error_type.replace('_', ' ')}")
        lines.append("")
        for row in group.head(5).itertuples(index=False):
            lines.append(f"- `{row.path}` — score {row.probability:.4f}")
        lines.append("")
    lines.extend(
        [
            "## Calibration details",
            "",
            f"- Objective: {calibration_summary['objective']}",
            f"- Mean internal balanced accuracy: {format_metric(calibration_summary['mean_balanced_accuracy'])}",
            f"- Worst internal balanced accuracy: {format_metric(calibration_summary['worst_balanced_accuracy'])}",
            "",
            "The WildFake demonstration split is a reference benchmark only and does not contribute to the final score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    calibration_path = resolve(args.calibration_predictions)
    target_path = resolve(args.target_predictions)
    full_evaluation_path = resolve(args.full_evaluation)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = read_predictions(calibration_path)
    target = read_predictions(target_path)
    threshold, calibration_summary = choose_threshold(calibration)

    calibration_rows = comparison_rows(calibration, threshold)
    target_rows = comparison_rows(target, threshold)
    error_cases = select_error_cases(target, threshold, args.error_cases_per_type)

    with full_evaluation_path.open("r", encoding="utf-8") as handle:
        full_evaluation = json.load(handle)
    robustness_rows = []
    for condition, metrics in full_evaluation["conditions"].items():
        overall = metrics["overall"]
        robustness_rows.append(
            {
                "condition": condition,
                "count": int(overall["count"]),
                "roc_auc": float(overall["roc_auc"]),
                "average_precision": float(overall["average_precision"]),
                "accuracy_at_0_5": float(overall["accuracy"]),
                "precision_at_0_5": float(overall["precision"]),
                "recall_at_0_5": float(overall["recall"]),
                "f1_at_0_5": float(overall["f1"]),
                "fp_at_0_5": int(overall["fp"]),
                "fn_at_0_5": int(overall["fn"]),
            }
        )

    pd.DataFrame(robustness_rows).to_csv(output_dir / "robustness_table.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "calibration_threshold_comparison.csv", index=False
    )
    pd.DataFrame(target_rows).to_csv(output_dir / "wildfake_threshold_comparison.csv", index=False)
    error_cases.to_csv(output_dir / "error_cases.csv", index=False)

    payload = {
        "policy": {
            "threshold_selected_on": str(calibration_path),
            "target_evaluated_on": str(target_path),
            "wildfake_used_for_threshold_selection": False,
        },
        "calibrated_threshold": threshold,
        "calibration": calibration_summary,
        "full_robustness": full_evaluation["robustness"],
        "target_threshold_comparison": target_rows,
        "representative_error_count": int(len(error_cases)),
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    report = build_report(
        threshold,
        calibration_summary,
        robustness_rows,
        target_rows,
        error_cases,
        calibration_path,
        target_path,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"threshold={threshold:.3f}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
