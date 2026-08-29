from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MANIFEST_ROOT = DATASET_ROOT / "manifests"
AUDIT_ROOT = DATASET_ROOT / "audit"

CF_INPUT = MANIFEST_ROOT / "communityforensics_small_train.csv"
EXISTING_TRAIN = MANIFEST_ROOT / "training_multisource.csv"
LEGACY_SELECTION = MANIFEST_ROOT / "validation_selection_6000.csv"
LEGACY_CONFIRMATION = MANIFEST_ROOT / "validation_confirmation_10000.csv"

CF_ASSIGNMENTS = MANIFEST_ROOT / "communityforensics_generator_assignments.csv"
CF_TRAIN = MANIFEST_ROOT / "communityforensics_train_balanced.csv"
CF_SELECTION = MANIFEST_ROOT / "communityforensics_selection_6000.csv"
CF_CONFIRMATION = MANIFEST_ROOT / "communityforensics_confirmation_6000.csv"
COMBINED_TRAIN = MANIFEST_ROOT / "training_modern_generators.csv"
COMBINED_SELECTION = MANIFEST_ROOT / "validation_modern_combined_selection_12000.csv"
COMBINED_CONFIRMATION = MANIFEST_ROOT / "validation_modern_combined_confirmation_16000.csv"
REPORT_PATH = AUDIT_ROOT / "modern_generator_manifest_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build format-balanced, generator-disjoint modern training manifests."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection-per-class", type=int, default=3000)
    parser.add_argument("--confirmation-per-class", type=int, default=3000)
    parser.add_argument("--max-train-per-generator", type=int, default=64)
    args = parser.parse_args()
    if args.selection_per_class < 1 or args.confirmation_per_class < 1:
        parser.error("selection/confirmation class sizes must be positive")
    if args.max_train_per_generator < 1:
        parser.error("--max-train-per-generator must be positive")
    return args


def stable_value(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def normalized_generator(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "path",
        "dataset",
        "source_class",
        "binary_label",
        "allowed_for_training",
        "sha256",
        "generator",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["binary_label"] = frame["binary_label"].astype(int)
    suffix_formats = (
        frame["path"]
        .map(lambda value: Path(str(value)).suffix.lower())
        .map(
            {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".webp": "WEBP",
                ".bmp": "BMP",
                ".tif": "TIFF",
                ".tiff": "TIFF",
            }
        )
        .fillna("UNKNOWN")
    )
    if "image_format" not in frame.columns:
        frame["image_format"] = suffix_formats
    else:
        blank = frame["image_format"].astype(str).str.strip().eq("")
        frame.loc[blank, "image_format"] = suffix_formats.loc[blank]
    return frame


def assert_unique(frame: pd.DataFrame, name: str) -> None:
    for column in ("path", "sha256"):
        duplicates = int(frame[column].duplicated().sum())
        if duplicates:
            raise ValueError(f"{name} contains {duplicates} duplicate {column} values")


def generator_assignments(
    fake: pd.DataFrame,
    existing_train: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    metadata = (
        fake.groupby("generator", sort=True)
        .agg(
            source_config=("source_config", "first"),
            architecture=("architecture", "first"),
            image_count=("path", "size"),
            png_count=("image_format", lambda values: int((values == "PNG").sum())),
            jpeg_count=("image_format", lambda values: int((values == "JPEG").sum())),
        )
        .reset_index()
    )
    if len(metadata) != fake["generator"].nunique():
        raise AssertionError("Generator metadata aggregation failed")

    existing_names = {
        normalized_generator(name)
        for name in existing_train.loc[
            existing_train["binary_label"].eq(1), "generator"
        ]
        if str(name).strip()
    }
    metadata["existing_name_overlap"] = metadata["generator"].map(
        lambda name: normalized_generator(name) in existing_names
    )
    metadata["assignment"] = ""
    metadata.loc[metadata["existing_name_overlap"], "assignment"] = "train"

    unassigned = metadata.loc[~metadata["existing_name_overlap"]].copy()
    for (source_config, architecture), group in unassigned.groupby(
        ["source_config", "architecture"], sort=True
    ):
        indices = sorted(
            group.index,
            key=lambda index: stable_value(
                f"generator:{source_config}:{architecture}:{metadata.at[index, 'generator']}",
                seed,
            ),
        )
        count = len(indices)
        if count >= 10:
            selection_count = max(1, round(count * 0.10))
            confirmation_count = max(1, round(count * 0.10))
        elif count >= 3:
            selection_count = 1
            confirmation_count = 1
        else:
            selection_count = 0
            confirmation_count = 0
        if selection_count + confirmation_count >= count:
            selection_count = confirmation_count = 0

        selection_indices = indices[:selection_count]
        confirmation_indices = indices[
            selection_count : selection_count + confirmation_count
        ]
        train_indices = indices[selection_count + confirmation_count :]
        metadata.loc[selection_indices, "assignment"] = "selection"
        metadata.loc[confirmation_indices, "assignment"] = "confirmation"
        metadata.loc[train_indices, "assignment"] = "train"

    if (metadata["assignment"] == "").any():
        raise AssertionError("Some generators were not assigned")
    return metadata.sort_values(["assignment", "source_config", "architecture", "generator"])


def deterministic_rows(frame: pd.DataFrame, seed: int, namespace: str) -> pd.DataFrame:
    order = frame["path"].map(lambda path: stable_value(f"{namespace}:{path}", seed))
    return frame.assign(_order=order).sort_values("_order").drop(columns="_order")


def round_robin_generators(
    frame: pd.DataFrame,
    target: int,
    seed: int,
    namespace: str,
    cap_per_generator: int | None = None,
) -> pd.DataFrame:
    if target < 0:
        raise ValueError("target must be non-negative")
    groups: dict[str, list[int]] = {}
    for generator, group in frame.groupby("generator", sort=True):
        ordered = deterministic_rows(group, seed, f"{namespace}:{generator}")
        indices = ordered.index.tolist()
        if cap_per_generator is not None:
            indices = indices[:cap_per_generator]
        groups[str(generator)] = indices

    generator_order = sorted(
        groups,
        key=lambda name: stable_value(f"{namespace}:generator:{name}", seed),
    )
    selected: list[int] = []
    depth = 0
    while len(selected) < target:
        added = 0
        for generator in generator_order:
            values = groups[generator]
            if depth < len(values):
                selected.append(values[depth])
                added += 1
                if len(selected) == target:
                    break
        if added == 0:
            break
        depth += 1
    if len(selected) < target:
        raise ValueError(
            f"Only {len(selected):,} rows available for target {target:,} in {namespace}"
        )
    return frame.loc[selected].copy()


def sample_fake_holdout(
    pool: pd.DataFrame,
    target: int,
    seed: int,
    namespace: str,
) -> pd.DataFrame:
    return round_robin_generators(pool, target, seed, namespace)


def take_real_matching_formats(
    real_pool: pd.DataFrame,
    fake_sample: pd.DataFrame,
    seed: int,
    namespace: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[pd.DataFrame] = []
    remaining = real_pool.copy()
    format_counts = fake_sample["image_format"].value_counts().to_dict()
    for image_format, count in sorted(format_counts.items()):
        candidates = remaining.loc[remaining["image_format"].eq(image_format)]
        if len(candidates) < count:
            raise ValueError(
                f"Need {count:,} real {image_format} rows for {namespace}, have {len(candidates):,}"
            )
        sample = deterministic_rows(candidates, seed, f"{namespace}:real:{image_format}").head(count)
        selected.append(sample)
        remaining = remaining.drop(index=sample.index)
    return pd.concat(selected, ignore_index=False), remaining


def sample_balanced_training(
    fake_pool: pd.DataFrame,
    real_pool: pd.DataFrame,
    cap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fake_parts: list[pd.DataFrame] = []
    real_parts: list[pd.DataFrame] = []
    for image_format, fake_format_pool in fake_pool.groupby("image_format", sort=True):
        real_format_pool = real_pool.loc[real_pool["image_format"].eq(image_format)]
        capped_capacity = sum(
            min(len(group), cap)
            for _, group in fake_format_pool.groupby("generator", sort=True)
        )
        target = min(capped_capacity, len(real_format_pool))
        if target == 0:
            continue
        fake_sample = round_robin_generators(
            fake_format_pool,
            target,
            seed,
            f"train:fake:{image_format}",
            cap_per_generator=cap,
        )
        real_sample = deterministic_rows(
            real_format_pool, seed, f"train:real:{image_format}"
        ).head(target)
        fake_parts.append(fake_sample)
        real_parts.append(real_sample)
    if not fake_parts:
        raise ValueError("No format-matched training rows were selected")
    return pd.concat(fake_parts, ignore_index=False), pd.concat(real_parts, ignore_index=False)


def prepare_rows(frame: pd.DataFrame, role: str, allowed: bool) -> pd.DataFrame:
    result = frame.copy()
    result["role"] = role
    result["allowed_for_training"] = str(allowed)
    return result


def union_columns(frames: Iterable[pd.DataFrame]) -> list[str]:
    columns: list[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
    return columns


def concatenate(frames: list[pd.DataFrame], seed: int, namespace: str) -> pd.DataFrame:
    columns = union_columns(frames)
    normalized = [frame.reindex(columns=columns, fill_value="") for frame in frames]
    result = pd.concat(normalized, ignore_index=True)
    order = result["path"].map(lambda path: stable_value(f"{namespace}:{path}", seed))
    return result.assign(_order=order).sort_values("_order").drop(columns="_order").reset_index(drop=True)


def cramers_v(frame: pd.DataFrame) -> float:
    table = pd.crosstab(frame["binary_label"], frame["image_format"])
    values = table.to_numpy(dtype=float)
    total = values.sum()
    if total == 0 or min(values.shape) < 2:
        return 0.0
    expected = values.sum(axis=1, keepdims=True) @ values.sum(axis=0, keepdims=True) / total
    chi_square = float(((values - expected) ** 2 / expected).sum())
    return (chi_square / (total * min(values.shape[0] - 1, values.shape[1] - 1))) ** 0.5


def format_table(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.groupby(["binary_label", "image_format"]).size()
    return {f"label_{label}:{image_format}": int(count) for (label, image_format), count in counts.items()}


def write_manifest(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cf = read_manifest(CF_INPUT)
    existing_train = read_manifest(EXISTING_TRAIN)
    legacy_selection = read_manifest(LEGACY_SELECTION)
    legacy_confirmation = read_manifest(LEGACY_CONFIRMATION)
    assert_unique(cf, "CommunityForensics input")
    assert_unique(existing_train, "existing train")

    fake = cf.loc[cf["binary_label"].eq(1)].copy()
    real = cf.loc[cf["binary_label"].eq(0)].copy()
    assignments = generator_assignments(fake, existing_train, args.seed)
    assignment_map = assignments.set_index("generator")["assignment"]
    fake["generator_assignment"] = fake["generator"].map(assignment_map)
    if fake["generator_assignment"].isna().any():
        raise AssertionError("Fake rows with no generator assignment")

    selection_fake = sample_fake_holdout(
        fake.loc[fake["generator_assignment"].eq("selection")],
        args.selection_per_class,
        args.seed,
        "selection",
    )
    confirmation_fake = sample_fake_holdout(
        fake.loc[fake["generator_assignment"].eq("confirmation")],
        args.confirmation_per_class,
        args.seed,
        "confirmation",
    )
    selection_real, remaining_real = take_real_matching_formats(
        real, selection_fake, args.seed, "selection"
    )
    confirmation_real, remaining_real = take_real_matching_formats(
        remaining_real, confirmation_fake, args.seed, "confirmation"
    )

    train_fake, train_real = sample_balanced_training(
        fake.loc[fake["generator_assignment"].eq("train")],
        remaining_real,
        args.max_train_per_generator,
        args.seed,
    )

    cf_train = concatenate(
        [
            prepare_rows(train_real, "train", True),
            prepare_rows(train_fake, "train", True),
        ],
        args.seed,
        "cf_train",
    )
    cf_selection = concatenate(
        [
            prepare_rows(selection_real, "cross_generator_selection", False),
            prepare_rows(selection_fake, "cross_generator_selection", False),
        ],
        args.seed,
        "cf_selection",
    )
    cf_confirmation = concatenate(
        [
            prepare_rows(confirmation_real, "cross_generator_confirmation", False),
            prepare_rows(confirmation_fake, "cross_generator_confirmation", False),
        ],
        args.seed,
        "cf_confirmation",
    )

    combined_train = concatenate([existing_train, cf_train], args.seed, "combined_train")
    combined_selection = concatenate(
        [legacy_selection, cf_selection], args.seed, "combined_selection"
    )
    combined_confirmation = concatenate(
        [legacy_confirmation, cf_confirmation], args.seed, "combined_confirmation"
    )

    outputs = {
        "cf_train": cf_train,
        "cf_selection": cf_selection,
        "cf_confirmation": cf_confirmation,
        "combined_train": combined_train,
        "combined_selection": combined_selection,
        "combined_confirmation": combined_confirmation,
    }
    for name, frame in outputs.items():
        assert_unique(frame, name)
    train_sha = set(combined_train["sha256"])
    selection_sha = set(combined_selection["sha256"])
    confirmation_sha = set(combined_confirmation["sha256"])
    if train_sha & selection_sha or train_sha & confirmation_sha or selection_sha & confirmation_sha:
        raise AssertionError("Train/selection/confirmation SHA leakage detected")

    cf_train_generators = set(train_fake["generator"])
    cf_selection_generators = set(selection_fake["generator"])
    cf_confirmation_generators = set(confirmation_fake["generator"])
    if (
        cf_train_generators & cf_selection_generators
        or cf_train_generators & cf_confirmation_generators
        or cf_selection_generators & cf_confirmation_generators
    ):
        raise AssertionError("CommunityForensics generator leakage detected")

    assignments.to_csv(CF_ASSIGNMENTS, index=False)
    write_manifest(CF_TRAIN, cf_train)
    write_manifest(CF_SELECTION, cf_selection)
    write_manifest(CF_CONFIRMATION, cf_confirmation)
    write_manifest(COMBINED_TRAIN, combined_train)
    write_manifest(COMBINED_SELECTION, combined_selection)
    write_manifest(COMBINED_CONFIRMATION, combined_confirmation)

    report = {
        "seed": args.seed,
        "inputs": {
            "communityforensics": str(CF_INPUT.relative_to(PROJECT_ROOT)),
            "existing_train": str(EXISTING_TRAIN.relative_to(PROJECT_ROOT)),
            "legacy_selection": str(LEGACY_SELECTION.relative_to(PROJECT_ROOT)),
            "legacy_confirmation": str(LEGACY_CONFIRMATION.relative_to(PROJECT_ROOT)),
        },
        "parameters": {
            "selection_per_class": args.selection_per_class,
            "confirmation_per_class": args.confirmation_per_class,
            "max_train_per_generator": args.max_train_per_generator,
        },
        "generator_assignments": {
            key: int(value)
            for key, value in assignments["assignment"].value_counts().sort_index().items()
        },
        "existing_generator_names_forced_to_train": assignments.loc[
            assignments["existing_name_overlap"], "generator"
        ].tolist(),
        "manifests": {
            name: {
                "rows": len(frame),
                "labels": {
                    str(label): int(count)
                    for label, count in frame["binary_label"].value_counts().sort_index().items()
                },
                "formats": format_table(frame) if "image_format" in frame else {},
                "format_cramers_v": cramers_v(frame) if "image_format" in frame else None,
            }
            for name, frame in outputs.items()
        },
        "cf_generator_counts": {
            "train": len(cf_train_generators),
            "selection": len(cf_selection_generators),
            "confirmation": len(cf_confirmation_generators),
        },
        "leakage_checks": {
            "path_and_sha_unique_per_manifest": True,
            "sha_disjoint_train_selection_confirmation": True,
            "cf_generators_disjoint_train_selection_confirmation": True,
        },
        "unused_cf_rows": int(
            len(cf)
            - len(cf_train)
            - len(cf_selection)
            - len(cf_confirmation)
        ),
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
