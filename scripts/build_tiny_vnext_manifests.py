"""Build source/label/generator-balanced Tiny vNext train and dev manifests."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = PROJECT_ROOT / "Dataset" / "manifests"
OLD_SOURCE_WEIGHTS = {
    "CommunityForensics-Small": 0.40,
    "GenImage": 0.20,
    "SID_Set": 0.20,
}
MODERN_WEIGHT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-train", type=int, default=280_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def sample_rows(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Cannot sample from an empty frame")
    return frame.sample(n=count, replace=len(frame) < count, random_state=seed)


def label_balanced(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count % 2:
        raise ValueError(f"Label-balanced count must be even, got {count}")
    parts = []
    for label in (0, 1):
        group = frame.loc[frame["binary_label"].astype(int) == label]
        parts.append(sample_rows(group, count // 2, seed + label))
    return pd.concat(parts, ignore_index=True)


def allocate(total: int, names: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(names))
    return {name: base + int(index < remainder) for index, name in enumerate(sorted(names))}


def modern_balanced(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count % 2:
        raise ValueError(f"Modern count must be even, got {count}")
    half = count // 2
    real = frame.loc[frame["binary_label"].astype(int) == 0].copy()
    fake = frame.loc[frame["binary_label"].astype(int) == 1].copy()

    real["_stratum"] = real["dataset"].astype(str) + ":real"
    fake["_stratum"] = fake["dataset"].astype(str) + ":" + fake["source_class"].astype(str)
    parts = []
    for offset, (subset, target) in enumerate(((real, half), (fake, half))):
        strata = sorted(subset["_stratum"].unique())
        quotas = allocate(target, strata)
        for index, stratum in enumerate(strata):
            group = subset.loc[subset["_stratum"] == stratum].drop(columns="_stratum")
            parts.append(sample_rows(group, quotas[stratum], seed + offset * 1000 + index))
    return pd.concat(parts, ignore_index=True)


def manifest_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "unique_paths": int(frame["path"].nunique()),
        "by_dataset": {str(k): int(v) for k, v in frame["dataset"].value_counts().sort_index().items()},
        "by_label": {
            str(k): int(v) for k, v in frame["binary_label"].astype(int).value_counts().sort_index().items()
        },
        "by_dataset_label": {
            f"{dataset}:{int(label)}": int(value)
            for (dataset, label), value in frame.groupby(["dataset", "binary_label"]).size().items()
        },
        "modern_fake_strata": dict(
            sorted(
                Counter(
                    f"{row.dataset}:{row.source_class}"
                    for row in frame.itertuples()
                    if row.dataset in {"SuSy", "MS-COCOAI"} and int(row.binary_label) == 1
                ).items()
            )
        ),
    }


def main() -> None:
    args = parse_args()
    total = int(args.total_train)
    if total <= 0 or total % 10:
        raise ValueError("--total-train must be positive and divisible by 10")
    seed = int(args.seed)

    replay = pd.read_csv(MANIFEST_ROOT / "replay_balanced_560000.csv", keep_default_na=False)
    susy_train = pd.read_csv(MANIFEST_ROOT / "susy_vnext_train.csv", keep_default_na=False)
    cocoai_train = pd.read_csv(MANIFEST_ROOT / "cocoai_vnext_train.csv", keep_default_na=False)
    modern = pd.concat([susy_train, cocoai_train], ignore_index=True)

    train_parts = []
    for offset, (dataset, weight) in enumerate(OLD_SOURCE_WEIGHTS.items()):
        target = round(total * weight)
        source = replay.loc[replay["dataset"] == dataset]
        train_parts.append(label_balanced(source, target, seed + offset * 100))
    modern_target = round(total * MODERN_WEIGHT)
    train_parts.append(modern_balanced(modern, modern_target, seed + 1000))
    train = pd.concat(train_parts, ignore_index=True)
    if len(train) != total:
        raise RuntimeError(f"Training row count mismatch: expected={total} actual={len(train)}")
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train["role"] = "tiny_vnext_train"
    train["allowed_for_training"] = True

    susy_dev = pd.read_csv(MANIFEST_ROOT / "susy_vnext_dev.csv", keep_default_na=False)
    cocoai_dev = pd.read_csv(MANIFEST_ROOT / "cocoai_vnext_dev.csv", keep_default_na=False)
    dev = pd.concat([susy_dev, cocoai_dev], ignore_index=True)
    if dev["sha256"].duplicated().any():
        duplicates = dev.loc[dev["sha256"].duplicated(keep=False), "path"].head(10).tolist()
        raise RuntimeError(f"Exact duplicate in combined modern development manifest: {duplicates}")
    dev = dev.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Treat the independent sources as one modern benchmark so generators from
    # either source are contrasted against the shared, exact-deduplicated real
    # pool (MS COCOAI real rows largely overlap historical CF COCO images).
    dev["dataset"] = "TinyVNext-Modern"
    dev["role"] = "tiny_vnext_development"
    dev["allowed_for_training"] = False

    train_path = MANIFEST_ROOT / f"tiny_vnext_train_balanced_{total}.csv"
    dev_path = MANIFEST_ROOT / "tiny_vnext_modern_dev.csv"
    train.to_csv(train_path, index=False)
    dev.to_csv(dev_path, index=False)
    train_hashes = set(train["sha256"].astype(str))
    dev_hashes = set(dev["sha256"].astype(str))
    overlap = sorted(train_hashes & dev_hashes)
    if overlap:
        raise RuntimeError(f"Tiny vNext train/dev exact overlap: {overlap[:10]}")

    summary = {
        "seed": seed,
        "target_total": total,
        "source_weights": {**OLD_SOURCE_WEIGHTS, "modern_combined": MODERN_WEIGHT},
        "train": manifest_summary(train),
        "development": manifest_summary(dev),
        "train_dev_exact_sha_overlap": overlap,
    }
    audit_path = PROJECT_ROOT / "outputs" / "tiny_vnext" / "data" / "manifest_summary.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {train_path} rows={len(train)}")
    print(f"wrote {dev_path} rows={len(dev)}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
