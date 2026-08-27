from __future__ import annotations

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MANIFEST_ROOT = DATASET_ROOT / "manifests"
AUDIT_ROOT = DATASET_ROOT / "audit"
SEED = 2026
COLUMN_ORDER = [
    "path",
    "dataset",
    "official_split",
    "role",
    "source_class",
    "source_label",
    "binary_label",
    "mask_path",
    "allowed_for_training",
    "archive_crc32",
    "uncompressed_bytes",
    "sha256",
    "duplicate_group",
    "generator",
    "source_md5",
    "source_image_path",
    "source_config",
    "width",
    "height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deduplicated SID + GenImage manifests.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sid-validation-per-class", type=int, default=2_000)
    args = parser.parse_args()
    if args.workers < 1 or args.sid_validation_per_class < 1:
        parser.error("worker and validation counts must be positive")
    return args


def load_manifest(name: str) -> pd.DataFrame:
    return pd.read_csv(MANIFEST_ROOT / name, keep_default_na=False)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    defaults = {
        "generator": "",
        "source_md5": "",
        "source_image_path": "",
        "source_config": "",
        "width": "",
        "height": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    for column in COLUMN_ORDER:
        if column not in frame.columns:
            frame[column] = ""
    # Pandas 3 preserves strict Arrow/integer dtypes from CSV inference. The
    # merged schema intentionally mixes blank strings with numeric metadata,
    # so use object columns before filling missing values.
    return frame[COLUMN_ORDER].astype(object)


def inspect_file(item: tuple[int, str]) -> tuple[int, str, str, int, int, int]:
    index, relative_path = item
    path = DATASET_ROOT / relative_path
    blob = path.read_bytes()
    sha256 = hashlib.sha256(blob).hexdigest()
    md5 = hashlib.md5(blob, usedforsecurity=False).hexdigest()
    with Image.open(io.BytesIO(blob)) as image:
        width, height = image.size
        image.verify()
    return index, sha256, md5, len(blob), width, height


def fill_hashes(frame: pd.DataFrame, description: str, workers: int) -> pd.DataFrame:
    frame = normalize_columns(frame)
    missing = (
        frame["sha256"].astype(str).str.strip().eq("")
        | frame["source_md5"].astype(str).str.strip().eq("")
        | frame["width"].astype(str).str.strip().eq("")
        | frame["height"].astype(str).str.strip().eq("")
    )
    items = [(int(index), str(frame.at[index, "path"])) for index in frame.index[missing]]
    if not items:
        return frame
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(inspect_file, items),
                total=len(items),
                desc=description,
                unit="image",
                dynamic_ncols=True,
            )
        )
    result_frame = pd.DataFrame(
        results,
        columns=["index", "sha256", "source_md5", "byte_count", "width", "height"],
    ).set_index("index")
    selected = result_frame.index
    recorded_sizes = frame.loc[selected, "uncompressed_bytes"].astype(str).str.strip()
    recorded_numeric = pd.to_numeric(recorded_sizes, errors="coerce")
    mismatched = recorded_numeric.notna() & recorded_numeric.ne(result_frame["byte_count"])
    if mismatched.any():
        index = mismatched[mismatched].index[0]
        raise RuntimeError(
            f"Size mismatch for {frame.at[index, 'path']}: "
            f"{frame.at[index, 'uncompressed_bytes']} != {result_frame.at[index, 'byte_count']}"
        )
    frame.loc[selected, "uncompressed_bytes"] = result_frame["byte_count"].astype(str)
    frame.loc[selected, "sha256"] = result_frame["sha256"]
    frame.loc[selected, "duplicate_group"] = result_frame["sha256"]
    frame.loc[selected, "source_md5"] = result_frame["source_md5"]
    frame.loc[selected, "source_image_path"] = frame.loc[selected, "path"]
    frame.loc[selected, "width"] = result_frame["width"].astype(str)
    frame.loc[selected, "height"] = result_frame["height"].astype(str)
    return frame


def load_or_build_hashed_manifest(
    source_name: str,
    cache_name: str,
    description: str,
    workers: int,
) -> pd.DataFrame:
    cache_path = MANIFEST_ROOT / cache_name
    if cache_path.exists():
        cached = load_manifest(cache_name)
        complete = all(
            ~cached[column].astype(str).str.strip().eq("").any()
            for column in ("sha256", "source_md5", "width", "height")
        )
        if complete:
            print(f"reuse hashed manifest: {cache_name}")
            return normalize_columns(cached)
    hashed = fill_hashes(load_manifest(source_name), description, workers)
    hashed.to_csv(cache_path, index=False)
    print(f"wrote hashed manifest: {cache_name}")
    return hashed


def distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.groupby(["dataset", "source_class", "binary_label"]).size()
    return {
        f"{dataset}/{source_class}/label_{label}": int(count)
        for (dataset, source_class, label), count in counts.items()
    }


def deduplicate(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, int]:
    label_counts = frame.groupby("sha256")["binary_label"].nunique()
    conflicts = label_counts[label_counts > 1]
    if not conflicts.empty:
        examples = conflicts.index[:5].tolist()
        raise RuntimeError(f"{name} has duplicate images with conflicting labels: {examples}")
    duplicate_count = int(frame.duplicated("sha256", keep="first").sum())
    return frame.drop_duplicates("sha256", keep="first").copy(), duplicate_count


def sample_sid_validation(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    sid = frame.loc[frame["dataset"].eq("SID_Set")]
    parts = []
    for offset, (source_class, group) in enumerate(sid.groupby("source_class", sort=True)):
        if len(group) < per_class:
            raise RuntimeError(
                f"SID validation has only {len(group)} {source_class} rows; need {per_class}"
            )
        parts.append(group.sample(n=per_class, random_state=SEED + offset))
    return pd.concat(parts, ignore_index=True)


def smoke_sample(frame: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    group_columns = ["dataset", "source_class", "generator"]
    for offset, (_, group) in enumerate(frame.groupby(group_columns, sort=True)):
        parts.append(
            group.sample(n=min(per_group, len(group)), random_state=seed + offset)
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    sid_train = load_or_build_hashed_manifest(
        "training_main.csv",
        "training_main_hashed.csv",
        "hash SID train",
        args.workers,
    )
    sid_validation = load_or_build_hashed_manifest(
        "validation_pool.csv",
        "validation_pool_hashed.csv",
        "hash SID validation",
        args.workers,
    )
    genimage_train = normalize_columns(load_manifest("genimage_train.csv"))
    glide_validation = normalize_columns(load_manifest("genimage_glide_validation.csv"))

    # SID rows are ordered first so a same-label exact duplicate keeps the
    # project's original training example and drops the GenImage copy.
    train_before = pd.concat([sid_train, genimage_train], ignore_index=True)
    train, train_duplicates = deduplicate(train_before, "training data")
    train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Keep GLIDE first inside validation so the held-out generator is preserved
    # if an identical same-label image also appears in SID validation.
    validation_before = pd.concat([glide_validation, sid_validation], ignore_index=True)
    validation_full, validation_duplicates = deduplicate(
        validation_before, "validation data"
    )
    train_hashes = set(train["sha256"])
    overlap_mask = validation_full["sha256"].isin(train_hashes)
    train_validation_overlap_removed = int(overlap_mask.sum())
    validation_full = validation_full.loc[~overlap_mask].copy()

    sid_checkpoint = sample_sid_validation(
        validation_full, args.sid_validation_per_class
    )
    glide_checkpoint = validation_full.loc[
        validation_full["dataset"].eq("GenImage")
        & validation_full["generator"].eq("GLIDE")
    ]
    validation = (
        pd.concat([sid_checkpoint, glide_checkpoint], ignore_index=True)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )

    if set(train["sha256"]) & set(validation["sha256"]):
        raise RuntimeError("Training/checkpoint-validation SHA-256 leakage remains")
    if not train["allowed_for_training"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    ).all():
        raise RuntimeError("Merged training manifest contains forbidden rows")
    if validation["allowed_for_training"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    ).any():
        raise RuntimeError("Merged validation manifest contains training-allowed rows")

    outputs = {
        "training_multisource.csv": train,
        "validation_multisource_full.csv": validation_full,
        "validation_multisource.csv": validation,
        "training_multisource_smoke.csv": smoke_sample(train, 100, SEED + 100),
        "validation_multisource_smoke.csv": smoke_sample(
            validation, 100, SEED + 200
        ),
    }
    for filename, frame in outputs.items():
        frame[COLUMN_ORDER].to_csv(MANIFEST_ROOT / filename, index=False)

    report = {
        "seed": SEED,
        "train_rows_before_deduplication": int(len(train_before)),
        "train_exact_duplicates_removed": train_duplicates,
        "train_rows": int(len(train)),
        "validation_rows_before_deduplication": int(len(validation_before)),
        "validation_exact_duplicates_removed": validation_duplicates,
        "train_validation_overlap_removed": train_validation_overlap_removed,
        "validation_full_rows": int(len(validation_full)),
        "checkpoint_validation_rows": int(len(validation)),
        "unique_train_sha256": int(train["sha256"].nunique()),
        "unique_validation_sha256": int(validation["sha256"].nunique()),
        "train_validation_sha256_overlap": int(
            len(set(train["sha256"]) & set(validation["sha256"]))
        ),
        "train_distribution": distribution(train),
        "checkpoint_validation_distribution": distribution(validation),
        "outputs": {name: int(len(frame)) for name, frame in outputs.items()},
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "multisource_manifest_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
