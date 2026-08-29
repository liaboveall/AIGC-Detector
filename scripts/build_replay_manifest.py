"""Build the source-balanced replay-only training manifest (target 560,000 rows).

Composition: CommunityForensics-Small 50% (140,000 real + 140,000 fake),
GenImage 25% (70,000 + 70,000), SID_Set 25% (70,000 + 70,000).
If any source/label pool is short, every source is downsampled proportionally
to the minimum availability (keeping 50/25/25 source ratio and 1:1 labels).

Triple assertions (fail fast):
  1. SHA256 sets disjoint from validation_modern_combined_selection_12000.csv
     and validation_modern_combined_confirmation_16000.csv.
  2. CF fake generators in the replay manifest are disjoint from the fake
     generator sets of communityforensics_selection_6000.csv and
     communityforensics_confirmation_6000.csv.
  3. path and sha256 unique inside the output manifest.

Deterministic: fixed seed, stable-hash ordering (same scheme as
scripts/build_modern_generator_manifests.py); reruns are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from build_modern_generator_manifests import (
    deterministic_rows,
    read_manifest,
    round_robin_generators,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MANIFEST_ROOT = DATASET_ROOT / "manifests"
AUDIT_ROOT = DATASET_ROOT / "audit"

CF_TRAIN_INPUT = MANIFEST_ROOT / "communityforensics_train_balanced.csv"
MULTISOURCE_INPUT = MANIFEST_ROOT / "training_multisource.csv"
SELECTION_12000 = MANIFEST_ROOT / "validation_modern_combined_selection_12000.csv"
CONFIRMATION_16000 = MANIFEST_ROOT / "validation_modern_combined_confirmation_16000.csv"
CF_SELECTION = MANIFEST_ROOT / "communityforensics_selection_6000.csv"
CF_CONFIRMATION = MANIFEST_ROOT / "communityforensics_confirmation_6000.csv"
REPORT_PATH = AUDIT_ROOT / "replay_manifest_report.json"
SUMMARY_PATH = AUDIT_ROOT / "replay_manifest_summary.txt"

REQUIRED_OUTPUT_COLUMNS = {
    "path",
    "dataset",
    "source_class",
    "binary_label",
    "allowed_for_training",
    "sha256",
    "generator",
}

RATIOS = {"CommunityForensics-Small": 0.5, "GenImage": 0.25, "SID_Set": 0.25}
TOTAL_TARGET = 560_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-balanced replay-only training manifest."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cf-per-label", type=int, default=140_000)
    parser.add_argument("--other-per-label", type=int, default=70_000)
    parser.add_argument("--max-per-generator", type=int, default=10_000)
    args = parser.parse_args()
    if args.cf_per_label < 1 or args.other_per_label < 1:
        parser.error("per-label targets must be positive")
    if args.max_per_generator < 1:
        parser.error("--max-per-generator must be positive")
    return args


def assert_unique(frame: pd.DataFrame, name: str) -> None:
    for column in ("path", "sha256"):
        duplicates = int(frame[column].duplicated().sum())
        if duplicates:
            raise ValueError(f"{name} contains {duplicates} duplicate {column} values")


def sha_set(path: Path) -> set[str]:
    frame = pd.read_csv(path, keep_default_na=False, usecols=["sha256"])
    return set(frame["sha256"])


def fake_generators(path: Path) -> set[str]:
    frame = pd.read_csv(
        path, keep_default_na=False, usecols=["binary_label", "generator"]
    )
    frame = frame.loc[frame["binary_label"].astype(int).eq(1)]
    return {str(name) for name in frame["generator"] if str(name).strip()}


def main() -> None:
    args = parse_args()

    cf = read_manifest(CF_TRAIN_INPUT)
    multisource = read_manifest(MULTISOURCE_INPUT)
    assert_unique(cf, "communityforensics_train_balanced")
    assert_unique(multisource, "training_multisource")
    if not multisource["allowed_for_training"].astype(str).str.lower().eq("true").all():
        raise AssertionError("training_multisource.csv has rows not allowed for training")
    if not cf["allowed_for_training"].astype(str).str.lower().eq("true").all():
        raise AssertionError("communityforensics_train_balanced.csv has rows not allowed for training")

    pools = {
        ("CommunityForensics-Small", 0): cf.loc[cf["binary_label"].eq(0)],
        ("CommunityForensics-Small", 1): cf.loc[cf["binary_label"].eq(1)],
        ("GenImage", 0): multisource.loc[
            multisource["dataset"].eq("GenImage") & multisource["binary_label"].eq(0)
        ],
        ("GenImage", 1): multisource.loc[
            multisource["dataset"].eq("GenImage") & multisource["binary_label"].eq(1)
        ],
        ("SID_Set", 0): multisource.loc[
            multisource["dataset"].eq("SID_Set") & multisource["binary_label"].eq(0)
        ],
        ("SID_Set", 1): multisource.loc[
            multisource["dataset"].eq("SID_Set") & multisource["binary_label"].eq(1)
        ],
    }
    availability = {
        f"{source}:label_{label}": int(len(pool)) for (source, label), pool in pools.items()
    }

    # Availability precheck: downsample proportionally if any pool is short.
    # CF fakes are sampled via generator round-robin with a per-generator cap,
    # so their effective capacity is capped per generator as well.
    capacities = {
        key: len(pool) for key, pool in pools.items()
    }
    cf_fake_pool = pools[("CommunityForensics-Small", 1)]
    cf_fake_generator_count = int(
        cf_fake_pool.loc[
            cf_fake_pool["generator"].astype(str).str.strip().ne(""), "generator"
        ].nunique()
    )
    capacities[("CommunityForensics-Small", 1)] = min(
        len(cf_fake_pool), cf_fake_generator_count * args.max_per_generator
    )
    scale = 1.0
    limiting_pool = None
    for source, ratio in RATIOS.items():
        per_label = TOTAL_TARGET * ratio / 2
        for label in (0, 1):
            capacity = capacities[(source, label)] / per_label
            if capacity < scale:
                scale = capacity
                limiting_pool = f"{source}:label_{label}"
    scale = min(1.0, scale)
    targets = {
        (source, label): int((TOTAL_TARGET * ratio / 2) * scale)
        for source, ratio in RATIOS.items()
        for label in (0, 1)
    }
    downsampled = scale < 1.0
    for (source, label), target in targets.items():
        if capacities[(source, label)] < target:
            raise AssertionError(
                f"Pool {source}/label={label} has capacity {capacities[(source, label)]:,} rows, "
                f"target {target:,} after scaling — logic error"
            )

    # Sampling. CF fakes: generator round-robin with per-generator cap.
    # CF reals and GenImage/SID pools: fixed-seed deterministic ordering.
    sampled: dict[tuple[str, int], pd.DataFrame] = {}
    sampled[("CommunityForensics-Small", 1)] = round_robin_generators(
        pools[("CommunityForensics-Small", 1)],
        targets[("CommunityForensics-Small", 1)],
        args.seed,
        "replay:cf:fake",
        cap_per_generator=args.max_per_generator,
    )
    for key in (("CommunityForensics-Small", 0), ("GenImage", 0), ("GenImage", 1),
                ("SID_Set", 0), ("SID_Set", 1)):
        source, label = key
        ordered = deterministic_rows(
            pools[key], args.seed, f"replay:{source.lower()}:{label}"
        )
        sampled[key] = ordered.head(targets[key]).copy()

    parts = []
    for key in sorted(sampled):
        part = sampled[key].copy()
        part["replay_source"] = key[0]
        part["replay_label"] = key[1]
        parts.append(part)
    columns: list[str] = []
    for part in parts:
        for column in part.columns:
            if column not in columns:
                columns.append(column)
    replay = pd.concat(
        [part.reindex(columns=columns, fill_value="") for part in parts],
        ignore_index=False,
    )
    missing = REQUIRED_OUTPUT_COLUMNS - set(replay.columns)
    if missing:
        raise AssertionError(f"Output manifest is missing columns: {sorted(missing)}")
    order = replay["path"].map(
        lambda path: hashlib.sha256(
            f"replay_manifest:{args.seed}:{path}".encode("utf-8")
        ).hexdigest()
    )
    replay = (
        replay.assign(_order=order)
        .sort_values("_order")
        .drop(columns=["_order", "replay_source", "replay_label"])
        .reset_index(drop=True)
    )

    # ---- Triple assertions ----
    assert_unique(replay, "replay manifest")

    selection_sha = sha_set(SELECTION_12000)
    confirmation_sha = sha_set(CONFIRMATION_16000)
    replay_sha = set(replay["sha256"])
    overlap_selection = replay_sha & selection_sha
    overlap_confirmation = replay_sha & confirmation_sha
    if overlap_selection or overlap_confirmation:
        raise AssertionError(
            "Replay manifest leaks into validation sets: "
            f"{len(overlap_selection)} selection / {len(overlap_confirmation)} confirmation SHA256 overlaps"
        )

    replay_cf_fake_generators = {
        str(name)
        for name in replay.loc[
            replay["dataset"].eq("CommunityForensics-Small")
            & replay["binary_label"].eq(1),
            "generator",
        ]
        if str(name).strip()
    }
    validation_generators = fake_generators(CF_SELECTION) | fake_generators(CF_CONFIRMATION)
    generator_overlap = replay_cf_fake_generators & validation_generators
    if generator_overlap:
        raise AssertionError(
            f"CF fake generator leakage with validation side: {sorted(generator_overlap)}"
        )

    label_counts = replay.groupby("binary_label").size().to_dict()
    source_counts = replay.groupby("dataset").size().to_dict()
    per_source_label = {
        f"{source}:label_{label}": int(count)
        for (source, label), count in replay.groupby(["dataset", "binary_label"]).size().items()
    }
    total = len(replay)
    expected = sum(targets.values())
    if total != expected:
        raise AssertionError(f"Expected {expected:,} rows (sum of per-pool targets), got {total:,}")
    if label_counts.get(0, 0) != label_counts.get(1, 0):
        raise AssertionError(f"Label imbalance: {label_counts}")

    # ---- Output ----
    output_name = f"replay_balanced_{total}.csv"
    output_path = MANIFEST_ROOT / output_name
    replay.to_csv(output_path, index=False)

    report = {
        "seed": args.seed,
        "sampling": {
            "cf_fake": (
                "generator round-robin with deterministic stable-hash ordering, "
                f"cap {args.max_per_generator} rows per generator"
            ),
            "cf_real_and_genimage_sid": "fixed-seed deterministic stable-hash ordering, head(target)",
        },
        "inputs": {
            "cf_train": str(CF_TRAIN_INPUT.relative_to(PROJECT_ROOT)),
            "multisource_train": str(MULTISOURCE_INPUT.relative_to(PROJECT_ROOT)),
            "validation_selection": str(SELECTION_12000.relative_to(PROJECT_ROOT)),
            "validation_confirmation": str(CONFIRMATION_16000.relative_to(PROJECT_ROOT)),
            "cf_selection": str(CF_SELECTION.relative_to(PROJECT_ROOT)),
            "cf_confirmation": str(CF_CONFIRMATION.relative_to(PROJECT_ROOT)),
        },
        "availability": availability,
        "downsampled": downsampled,
        "downsample_scale": scale,
        "limiting_pool": limiting_pool,
        "targets_per_source_label": {
            f"{source}:label_{label}": targets[(source, label)]
            for source in RATIOS
            for label in (0, 1)
        },
        "output_manifest": str(output_path.relative_to(PROJECT_ROOT)),
        "manifest_rows": total,
        "label_counts": {str(label): int(count) for label, count in sorted(label_counts.items())},
        "source_counts": {source: int(count) for source, count in sorted(source_counts.items())},
        "per_source_label": per_source_label,
        "ratios": {
            source: round(count / total, 6) for source, count in source_counts.items()
        },
        "cf_fake_generator_count": len(replay_cf_fake_generators),
        "cf_fake_per_generator_max": int(
            replay.loc[
                replay["dataset"].eq("CommunityForensics-Small")
                & replay["binary_label"].eq(1),
                "generator",
            ]
            .value_counts()
            .max()
        ),
        "assertions": {
            "path_and_sha_unique": True,
            "sha_disjoint_from_selection_12000": True,
            "sha_disjoint_from_confirmation_16000": True,
            "cf_generators_disjoint_from_validation": True,
            "label_balance_1_to_1": True,
        },
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Replay manifest build summary",
        "=============================",
        f"seed: {args.seed}",
        f"output: {output_path.relative_to(PROJECT_ROOT)}",
        f"total rows: {total:,}",
        f"downsampled: {downsampled}"
        + (f" (scale={scale:.6f}, limiting pool: {limiting_pool})" if downsampled else ""),
        "",
        "rows by source x label:",
    ]
    lines += [f"  {key}: {count:,}" for key, count in sorted(per_source_label.items())]
    lines += [
        "",
        "source ratios:",
    ]
    lines += [f"  {source}: {count / total:.4f}" for source, count in sorted(source_counts.items())]
    lines += [
        "",
        "availability (input pools):",
    ]
    lines += [f"  {key}: {count:,}" for key, count in sorted(availability.items())]
    lines += [
        "",
        f"CF fake generators used: {len(replay_cf_fake_generators)} "
        f"(max {report['cf_fake_per_generator_max']:,} rows per generator)",
        "",
        "assertions: ALL PASSED",
        "  - path/sha256 unique within manifest",
        "  - sha256 disjoint from selection_12000 and confirmation_16000",
        "  - CF fake generators disjoint from CF selection/confirmation validation sets",
        "  - label balance 1:1",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
