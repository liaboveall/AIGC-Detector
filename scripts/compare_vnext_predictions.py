"""Compare aligned predictions with generator-macro robustness metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Grouped bootstrap replicates. Use 0 only for non-gating diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-macro-gain", type=float, default=0.005)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_aligned(baseline_path: Path, candidate_path: Path, manifest_path: Path) -> pd.DataFrame:
    keys = ["path", "condition"]
    static = ["dataset", "source_class", "label"]
    baseline = pd.read_csv(baseline_path).rename(columns={"probability": "baseline_probability"})
    candidate = pd.read_csv(candidate_path).rename(columns={"probability": "candidate_probability"})
    if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("Prediction files contain duplicate path/condition keys")
    merged = baseline.merge(candidate, on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("Baseline and candidate predictions are not perfectly aligned")
    for column in static:
        left, right = f"{column}_base", f"{column}_candidate"
        if not (merged[left].astype(str) == merged[right].astype(str)).all():
            raise ValueError(f"Static prediction column differs: {column}")
        merged[column] = merged[left]
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if manifest["path"].duplicated().any():
        raise ValueError("Development manifest contains duplicate paths")
    group_column = "content_group" if "content_group" in manifest.columns else "duplicate_group"
    if group_column not in manifest.columns:
        raise ValueError("Development manifest must contain content_group or duplicate_group")
    groups = manifest[["path", group_column]].rename(columns={group_column: "content_group"})
    merged = merged.merge(groups, on="path", how="left", validate="many_to_one")
    if merged["content_group"].isna().any():
        raise ValueError("Predictions contain paths absent from the development manifest")
    merged["label"] = merged["label"].astype(int)
    return merged[keys + static + ["content_group", "baseline_probability", "candidate_probability"]]


def safe_auc(frame: pd.DataFrame, probability: str, weights: np.ndarray | None = None) -> float:
    labels = frame["label"].to_numpy(dtype=np.int64)
    if len(np.unique(labels)) != 2:
        return math.nan
    return float(
        roc_auc_score(
            labels,
            frame[probability].to_numpy(dtype=np.float64),
            sample_weight=weights,
        )
    )


def robust(values: dict[str, float]) -> float:
    degraded = [value for condition, value in values.items() if condition != "clean" and math.isfinite(value)]
    if not degraded:
        return math.nan
    return 0.8 * float(np.mean(degraded)) + 0.2 * float(np.min(degraded))


def summarise(
    frame: pd.DataFrame,
    probability: str,
    row_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    conditions = sorted(frame["condition"].unique(), key=lambda value: (value != "clean", value))
    strata: dict[str, dict[str, Any]] = {}
    for dataset in sorted(frame["dataset"].unique()):
        dataset_frame = frame.loc[frame["dataset"] == dataset]
        generators = sorted(dataset_frame.loc[dataset_frame["label"] == 1, "source_class"].unique())
        for generator in generators:
            mask = (frame["dataset"] == dataset) & (
                (frame["label"] == 0) | (frame["source_class"] == generator)
            )
            contrast = frame.loc[mask]
            contrast_weights = row_weights[np.flatnonzero(mask.to_numpy())] if row_weights is not None else None
            aucs: dict[str, float] = {}
            for condition in conditions:
                condition_mask = contrast["condition"] == condition
                condition_frame = contrast.loc[condition_mask]
                weights = (
                    contrast_weights[np.flatnonzero(condition_mask.to_numpy())]
                    if contrast_weights is not None
                    else None
                )
                aucs[condition] = safe_auc(condition_frame, probability, weights)
            key = f"{dataset}:{generator}"
            strata[key] = {
                "condition_aucs": aucs,
                "robust_score": robust(aucs),
                "worst_degraded_auc": min(value for name, value in aucs.items() if name != "clean"),
            }
    robust_values = [entry["robust_score"] for entry in strata.values()]
    worst_values = [entry["worst_degraded_auc"] for entry in strata.values()]
    return {
        "macro_robust_score": float(np.mean(robust_values)),
        "worst_generator_robust_score": float(np.min(robust_values)),
        "worst_generator_condition_auc": float(np.min(worst_values)),
        "strata": strata,
    }


def bootstrap_deltas(frame: pd.DataFrame, count: int, seed: int) -> dict[str, Any] | None:
    if count <= 0:
        return None
    rng = np.random.default_rng(seed)
    dataset_groups = {
        dataset: sorted(group["content_group"].unique())
        for dataset, group in frame.groupby("dataset", sort=True)
    }
    deltas = []
    for _ in range(count):
        group_counts: dict[tuple[str, str], int] = {}
        for dataset, groups in dataset_groups.items():
            sampled = rng.choice(groups, size=len(groups), replace=True)
            values, counts = np.unique(sampled, return_counts=True)
            group_counts.update(
                {(dataset, str(value)): int(sample_count) for value, sample_count in zip(values, counts)}
            )
        weights = np.fromiter(
            (
                group_counts.get((str(dataset), str(content_group)), 0)
                for dataset, content_group in zip(frame["dataset"], frame["content_group"], strict=True)
            ),
            dtype=np.float64,
            count=len(frame),
        )
        base = summarise(frame, "baseline_probability", weights)
        candidate = summarise(frame, "candidate_probability", weights)
        deltas.append(candidate["macro_robust_score"] - base["macro_robust_score"])
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "replicates": count,
        "seed": seed,
        "mean_delta": float(array.mean()),
        "ci95_lower": float(np.quantile(array, 0.025)),
        "ci95_upper": float(np.quantile(array, 0.975)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    frame = load_aligned(resolve(args.baseline), resolve(args.candidate), resolve(args.manifest))
    baseline = summarise(frame, "baseline_probability")
    candidate = summarise(frame, "candidate_probability")
    delta = {
        key: candidate[key] - baseline[key]
        for key in (
            "macro_robust_score",
            "worst_generator_robust_score",
            "worst_generator_condition_auc",
        )
    }
    bootstrap = bootstrap_deltas(frame, args.bootstrap, args.seed)
    gates = {
        "macro_robust_gain": delta["macro_robust_score"] >= args.min_macro_gain,
        "worst_generator_non_regression": delta["worst_generator_robust_score"] >= 0.0,
        "worst_generator_condition_non_regression": delta["worst_generator_condition_auc"] >= 0.0,
        "bootstrap_ci_positive": bootstrap is not None and bootstrap["ci95_lower"] > 0.0,
    }
    payload = {
        "baseline_predictions": str(resolve(args.baseline)),
        "candidate_predictions": str(resolve(args.candidate)),
        "manifest": str(resolve(args.manifest)),
        "rows": len(frame),
        "images": int(frame["path"].nunique()),
        "conditions": int(frame["condition"].nunique()),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "bootstrap": bootstrap,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"delta": delta, "bootstrap": bootstrap, "gates": gates, "accepted": payload["accepted"]}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
