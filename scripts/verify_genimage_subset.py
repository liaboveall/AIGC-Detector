from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MANIFEST_ROOT = DATASET_ROOT / "manifests"
SUBSET_ROOT = DATASET_ROOT / "GenImage_subset"
STATE_PATH = SUBSET_ROOT / "_state" / "download_state.jsonl"
REPORT_PATH = DATASET_ROOT / "audit" / "genimage_integrity_report.json"

EXPECTED_TRAIN = {
    ("ImageNet", 0): 140_000,
    ("Midjourney", 1): 20_000,
    ("SD1.4", 1): 20_000,
    ("SD1.5", 1): 20_000,
    ("ADM", 1): 20_000,
    ("BigGAN", 1): 20_000,
    ("VQDM", 1): 20_000,
    ("Wukong", 1): 20_000,
}
EXPECTED_VALIDATION = {("GLIDE", 0): 5_000, ("GLIDE", 1): 5_000}
EXPECTED_TARGETS = {
    "train_fake_midjourney": 20_000,
    "train_fake_sd14": 20_000,
    "train_fake_sd15": 20_000,
    "train_fake_adm": 20_000,
    "train_fake_biggan": 20_000,
    "train_fake_vqdm": 20_000,
    "train_fake_wukong": 20_000,
    "train_real_imagenet": 140_000,
    "validation_glide_fake": 5_000,
    "validation_glide_real": 5_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the exported GenImage subset.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-reported-errors", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_error(errors: list[str], message: str, limit: int) -> None:
    if len(errors) < limit:
        errors.append(message)


def check_file(row: dict[str, str]) -> tuple[str, list[str]]:
    relative_path = row["path"]
    path = DATASET_ROOT / relative_path
    errors: list[str] = []
    try:
        resolved = path.resolve()
        resolved.relative_to(DATASET_ROOT.resolve())
    except (OSError, ValueError):
        return relative_path, ["path escapes Dataset root"]
    if not path.is_file():
        return relative_path, ["file missing"]
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return relative_path, [f"read failed: {exc}"]

    expected_size = int(row["uncompressed_bytes"])
    if len(blob) != expected_size:
        errors.append(f"size mismatch: {len(blob)} != {expected_size}")
    sha256 = hashlib.sha256(blob).hexdigest()
    if sha256 != row["sha256"].strip().lower():
        errors.append("SHA-256 mismatch")
    md5 = hashlib.md5(blob, usedforsecurity=False).hexdigest()
    if md5 != row["source_md5"].strip().lower():
        errors.append("MD5 mismatch")
    try:
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            image.verify()
        if width != int(row["width"]) or height != int(row["height"]):
            errors.append(
                f"dimensions mismatch: {width}x{height} != {row['width']}x{row['height']}"
            )
    except Exception as exc:
        errors.append(f"image verification failed: {exc}")
    return relative_path, errors


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    train_path = MANIFEST_ROOT / "genimage_train.csv"
    validation_path = MANIFEST_ROOT / "genimage_glide_validation.csv"
    train = read_csv(train_path)
    validation = read_csv(validation_path)
    rows = train + validation

    if len(train) != 280_000:
        append_error(errors, f"train row count: {len(train)} != 280000", args.max_reported_errors)
    if len(validation) != 10_000:
        append_error(
            errors,
            f"validation row count: {len(validation)} != 10000",
            args.max_reported_errors,
        )
    if len(rows) != 290_000:
        append_error(errors, f"total row count: {len(rows)} != 290000", args.max_reported_errors)

    train_distribution = Counter(
        (row["generator"], int(row["binary_label"])) for row in train
    )
    validation_distribution = Counter(
        (row["generator"], int(row["binary_label"])) for row in validation
    )
    if dict(train_distribution) != EXPECTED_TRAIN:
        append_error(
            errors,
            f"unexpected train distribution: {dict(train_distribution)}",
            args.max_reported_errors,
        )
    if dict(validation_distribution) != EXPECTED_VALIDATION:
        append_error(
            errors,
            f"unexpected validation distribution: {dict(validation_distribution)}",
            args.max_reported_errors,
        )

    for row in train:
        if not truthy(row["allowed_for_training"]):
            append_error(errors, f"train row forbidden: {row['path']}", args.max_reported_errors)
        if row["role"] != "train":
            append_error(errors, f"bad train role: {row['path']}", args.max_reported_errors)
    for row in validation:
        if truthy(row["allowed_for_training"]):
            append_error(
                errors,
                f"validation row allowed for training: {row['path']}",
                args.max_reported_errors,
            )
        if row["role"] != "cross_generator_validation":
            append_error(
                errors, f"bad validation role: {row['path']}", args.max_reported_errors
            )

    paths = [row["path"] for row in rows]
    sha256_values = [row["sha256"].strip().lower() for row in rows]
    md5_values = [row["source_md5"].strip().lower() for row in rows]
    for name, values in (("path", paths), ("SHA-256", sha256_values), ("MD5", md5_values)):
        duplicates = len(values) - len(set(values))
        if duplicates:
            append_error(errors, f"duplicate {name} values: {duplicates}", args.max_reported_errors)

    train_sha = {row["sha256"].strip().lower() for row in train}
    validation_sha = {row["sha256"].strip().lower() for row in validation}
    leakage = train_sha & validation_sha
    if leakage:
        append_error(
            errors,
            f"train/validation SHA-256 leakage: {len(leakage)}",
            args.max_reported_errors,
        )

    state_count = 0
    state_targets: Counter[str] = Counter()
    state_paths: set[str] = set()
    with STATE_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                append_error(
                    errors,
                    f"invalid state JSON at line {line_number}: {exc}",
                    args.max_reported_errors,
                )
                continue
            state_count += 1
            state_targets[entry["target"]] += 1
            state_paths.add(entry["path"])
    if state_count != 290_000:
        append_error(errors, f"state row count: {state_count} != 290000", args.max_reported_errors)
    if dict(state_targets) != EXPECTED_TARGETS:
        append_error(
            errors,
            f"unexpected state target distribution: {dict(state_targets)}",
            args.max_reported_errors,
        )
    if state_paths != set(paths):
        append_error(
            errors,
            f"state/manifest path mismatch: {len(state_paths ^ set(paths))}",
            args.max_reported_errors,
        )

    actual_files = {
        path.relative_to(DATASET_ROOT).as_posix()
        for path in SUBSET_ROOT.rglob("*")
        if path.is_file() and "_state" not in path.parts
    }
    expected_files = set(paths)
    missing = expected_files - actual_files
    orphaned = actual_files - expected_files
    if missing:
        append_error(errors, f"missing files from disk: {len(missing)}", args.max_reported_errors)
    if orphaned:
        append_error(errors, f"orphaned image files: {len(orphaned)}", args.max_reported_errors)

    file_error_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(check_file, rows)
        for relative_path, file_errors in tqdm(
            results,
            total=len(rows),
            desc="verify",
            unit="image",
            dynamic_ncols=True,
        ):
            if file_errors:
                file_error_count += 1
                for message in file_errors:
                    append_error(
                        errors,
                        f"{relative_path}: {message}",
                        args.max_reported_errors,
                    )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "total_rows": len(rows),
        "unique_paths": len(set(paths)),
        "unique_sha256": len(set(sha256_values)),
        "unique_md5": len(set(md5_values)),
        "state_rows": state_count,
        "actual_image_files": len(actual_files),
        "missing_files": len(missing),
        "orphaned_files": len(orphaned),
        "train_validation_sha256_overlap": len(leakage),
        "files_with_integrity_errors": file_error_count,
        "train_distribution": {
            f"{generator}:label_{label}": count
            for (generator, label), count in sorted(train_distribution.items())
        },
        "validation_distribution": {
            f"{generator}:label_{label}": count
            for (generator, label), count in sorted(validation_distribution.items())
        },
        "errors": errors,
        "reported_error_limit": args.max_reported_errors,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
