"""Create an aligned logit-space blend of two prediction CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="alpha=0 prediction CSV")
    parser.add_argument("--candidate", required=True, help="alpha=1 prediction CSV")
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    keys = ["path", "condition"]
    static = ["dataset", "source_class", "label"]
    baseline = pd.read_csv(resolve(args.baseline)).rename(columns={"probability": "p0"})
    candidate = pd.read_csv(resolve(args.candidate)).rename(columns={"probability": "p1"})
    if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("prediction inputs contain duplicate path/condition keys")
    merged = baseline.merge(candidate, on=keys, suffixes=("_0", "_1"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("prediction inputs are not perfectly aligned")
    output = merged[keys].copy()
    for column in static:
        left, right = f"{column}_0", f"{column}_1"
        if not (merged[left].astype(str) == merged[right].astype(str)).all():
            raise ValueError(f"static prediction column differs: {column}")
        output[column] = merged[left]
    epsilon = 1e-6
    p0 = np.clip(merged["p0"].to_numpy(dtype=np.float64), epsilon, 1.0 - epsilon)
    p1 = np.clip(merged["p1"].to_numpy(dtype=np.float64), epsilon, 1.0 - epsilon)
    logit0 = np.log(p0) - np.log1p(-p0)
    logit1 = np.log(p1) - np.log1p(-p1)
    blended_logit = (1.0 - args.alpha) * logit0 + args.alpha * logit1
    output["probability"] = 1.0 / (1.0 + np.exp(-blended_logit))
    output = output[["path", "dataset", "source_class", "label", "condition", "probability"]]
    destination = resolve(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"wrote {destination} rows={len(output):,} alpha={args.alpha:.3f}")


if __name__ == "__main__":
    main()
