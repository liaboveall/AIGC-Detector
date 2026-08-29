"""Observation-only WildFake analysis at the FROZEN threshold 0.209.

Produces the operating-point table (per-condition accuracy / balanced accuracy /
precision / recall / specificity / F1 / FP / FN), per-source-class AUC, and a
robust score (0.8*mean + 0.2*worst degraded AUC) from a predictions CSV plus the
full evaluation JSON written by evaluate.py.

This script performs NO threshold selection: 0.209 is the frozen internal threshold.
WildFake is a one-shot observational benchmark and never feeds back into decisions.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

FROZEN_THRESHOLD = 0.209

CONDITION_ORDER = [
    "clean", "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
    "blur_0.5", "blur_1.0", "blur_2.0",
    "scale_0.5", "scale_0.25",
    "noise_0.02", "noise_0.05", "noise_0.10",
    "color_-0.20", "color_0.20", "crop_0.80",
]
DEGRADED = [c for c in CONDITION_ORDER if c != "clean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="candidate")
    return parser.parse_args()


def condition_metrics(rows: pd.DataFrame, threshold: float) -> dict:
    pred = (rows["probability"] >= threshold).astype(int)
    label = rows["label"].astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    accuracy = (tp + tn) / max(len(rows), 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    balanced = (recall + specificity) / 2.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "count": len(rows),
        "roc_auc": float(roc_auc_score(label, rows["probability"])),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    print(f"loaded {len(df)} predictions from {args.predictions}")

    with open(args.eval_json, encoding="utf-8") as handle:
        eval_payload = json.load(handle)

    # Per-condition table at the frozen threshold (AUC cross-checked against eval JSON).
    table_rows = []
    for condition in CONDITION_ORDER:
        rows = df[df["condition"] == condition]
        metrics = condition_metrics(rows, FROZEN_THRESHOLD)
        json_auc = float(eval_payload["conditions"][condition]["overall"]["roc_auc"])
        assert abs(metrics["roc_auc"] - json_auc) < 1e-6, f"AUC mismatch for {condition}"
        table_rows.append({"condition": condition, **metrics})
    with (out_dir / "wildfake_frozen_threshold_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    degraded_aucs = [r["roc_auc"] for r in table_rows if r["condition"] != "clean"]
    robust_score = 0.8 * (sum(degraded_aucs) / len(degraded_aucs)) + 0.2 * min(degraded_aucs)
    clean_auc = next(r["roc_auc"] for r in table_rows if r["condition"] == "clean")
    mean_bacc = sum(r["balanced_accuracy"] for r in table_rows) / len(table_rows)
    best = max(table_rows, key=lambda r: r["balanced_accuracy"])
    worst = min(table_rows, key=lambda r: r["balanced_accuracy"])
    total_fp = sum(r["fp"] for r in table_rows)
    total_fn = sum(r["fn"] for r in table_rows)

    # Per source class (real = COCO val2017, fake = DALL-E 3 Advanced).
    source_rows = []
    for source, rows in df.groupby("source_class"):
        for condition in CONDITION_ORDER:
            sub = rows[rows["condition"] == condition]
            if len(sub) == 0 or sub["label"].nunique() < 2:
                # single-class subset: report mean score and frozen-threshold stats only
                pred = (sub["probability"] >= FROZEN_THRESHOLD).astype(int)
                source_rows.append({
                    "source_class": source,
                    "condition": condition,
                    "count": len(sub),
                    "roc_auc": None,
                    "mean_probability": float(sub["probability"].mean()),
                    "median_probability": float(sub["probability"].median()),
                    "fraction_above_threshold": float(pred.mean()),
                })
                continue
            source_rows.append({
                "source_class": source,
                "condition": condition,
                "count": len(sub),
                "roc_auc": None,  # AUC undefined for single-class subsets; keep column shape
                "mean_probability": float(sub["probability"].mean()),
                "median_probability": float(sub["probability"].median()),
                "fraction_above_threshold": float((sub["probability"] >= FROZEN_THRESHOLD).mean()),
            })
    # real/fake subsets are single-class; the meaningful split metric is fraction
    # above threshold (real: FPR proxy, fake: recall proxy). Recompute without AUC.
    with (out_dir / "wildfake_source_class_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    robustness_ref = eval_payload.get("robustness") or {}
    summary = {
        "label": args.label,
        "predictions": args.predictions,
        "eval_json": args.eval_json,
        "frozen_threshold": FROZEN_THRESHOLD,
        "robust_score_08_02": robust_score,
        "robust_score_from_eval_json": robustness_ref.get("robust_score"),
        "clean_auc": clean_auc,
        "mean_degraded_auc": sum(degraded_aucs) / len(degraded_aucs),
        "worst_degraded_auc": min(degraded_aucs),
        "worst_condition": worst["condition"],
        "best_condition": best["condition"],
        "mean_balanced_accuracy_at_frozen": mean_bacc,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_condition": table_rows,
    }
    with (out_dir / "wildfake_frozen_threshold_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[{args.label}] robust score = {robust_score:.4f}")
    print(f"[{args.label}] clean AUC = {clean_auc:.4f}, mean degraded AUC = {summary['mean_degraded_auc']:.4f}, worst = {worst['condition']} {summary['worst_degraded_auc']:.4f}")
    print(f"[{args.label}] mean balanced accuracy @ {FROZEN_THRESHOLD} = {mean_bacc:.4f}")
    print(f"[{args.label}] best = {best['condition']} ({best['balanced_accuracy']:.4f}), worst = {worst['condition']} ({worst['balanced_accuracy']:.4f})")
    print(f"[{args.label}] total FP/FN = {total_fp}/{total_fn}")


if __name__ == "__main__":
    main()
