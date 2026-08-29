from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# Configure the Hub before importing huggingface_hub.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = DATASET_ROOT / "CommunityForensics_Small"
IMAGE_ROOT = OUTPUT_ROOT / "images"
STATE_ROOT = OUTPUT_ROOT / "_state"
SHARD_MANIFEST_ROOT = STATE_ROOT / "shard_manifests"
STAGING_ROOT = OUTPUT_ROOT / "_staging"
COMPLETED_PATH = STATE_ROOT / "completed_shards.jsonl"
LEAKAGE_CACHE_PATH = STATE_ROOT / "wildfake_hashes.csv"
MASTER_MANIFEST_PATH = DATASET_ROOT / "manifests" / "communityforensics_small_all.csv"
TRAIN_MANIFEST_PATH = DATASET_ROOT / "manifests" / "communityforensics_small_train.csv"
HOLDOUT_MANIFEST_PATH = DATASET_ROOT / "manifests" / "communityforensics_small_holdout.csv"
REPORT_PATH = DATASET_ROOT / "audit" / "communityforensics_small_download_report.json"

REPOSITORY_ID = "OwensLab/CommunityForensics-Small"
REPOSITORY_REVISION = "6c539a534c07917307c381f5af4053c6091b5278"
LICENSE_ID = "CC-BY-NC-SA-4.0"
LICENSE_URL = (
    "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/"
    "blob/main/README.md#licensing-information"
)

MANIFEST_COLUMNS = [
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
    "image_format",
    "image_mode",
    "architecture",
    "real_source",
    "prompt_sha256",
    "source_repository",
    "source_revision",
    "source_license",
    "source_shard",
]

PROTECTED_MANIFESTS = (
    DATASET_ROOT / "manifests" / "training_multisource.csv",
    DATASET_ROOT / "manifests" / "validation_multisource_full.csv",
)
WILDFAKE_MANIFEST = DATASET_ROOT / "manifests" / "wildfake_demo.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and export the complete usable CommunityForensics-Small dataset "
            "one Parquet shard at a time."
        )
    )
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--revision", default=REPOSITORY_REVISION)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-free-gib", type=float, default=40.0)
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep successfully exported Parquet shards (roughly doubles disk use).",
    )
    parser.add_argument(
        "--only-shard",
        help="Export one shard filename, for example HFCF_small_117.parquet.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Process at most this many unfinished shards; useful for smoke tests.",
    )
    args = parser.parse_args()
    if not args.accept_license:
        parser.error(
            f"Review {LICENSE_ID} at {LICENSE_URL}, then pass --accept-license"
        )
    if args.max_retries < 1 or args.batch_size < 1:
        parser.error("--max-retries and --batch-size must be positive")
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be positive")
    return args


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    sample = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(sample.getdata())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{bits:016x}"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_completed_shards() -> set[str]:
    completed: set[str] = set()
    if not COMPLETED_PATH.exists():
        return completed
    with COMPLETED_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "complete":
                completed.add(str(record["filename"]))
    return completed


def append_completion(record: dict[str, Any]) -> None:
    COMPLETED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPLETED_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def protected_sha256_values() -> set[str]:
    values: set[str] = set()
    for manifest_path in PROTECTED_MANIFESTS:
        for row in load_csv(manifest_path):
            digest = row.get("sha256", "").strip().lower()
            if digest:
                values.add(digest)
    return values


def load_or_build_wildfake_hashes() -> tuple[set[str], set[str]]:
    expected_rows = load_csv(WILDFAKE_MANIFEST)
    cached_rows = load_csv(LEAKAGE_CACHE_PATH)
    if len(cached_rows) == len(expected_rows) and {
        row["path"] for row in cached_rows
    } == {row["path"] for row in expected_rows}:
        return (
            {row["sha256"] for row in cached_rows},
            {row["dhash"] for row in cached_rows},
        )

    print(f"hashing {len(expected_rows):,} protected WildFake images", flush=True)
    cache: list[dict[str, str]] = []
    for index, row in enumerate(expected_rows, start=1):
        relative_path = row["path"]
        path = DATASET_ROOT / relative_path
        digest = sha256_file(path)
        with Image.open(path) as image:
            dhash = difference_hash(image)
        cache.append({"path": relative_path, "sha256": digest, "dhash": dhash})
        if index % 1000 == 0:
            print(f"wildfake hashes: {index:,}/{len(expected_rows):,}", flush=True)

    LEAKAGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = LEAKAGE_CACHE_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "dhash"])
        writer.writeheader()
        writer.writerows(cache)
    os.replace(temporary, LEAKAGE_CACHE_PATH)
    return ({row["sha256"] for row in cache}, {row["dhash"] for row in cache})


def numeric_shard_key(filename: str) -> tuple[int, str]:
    match = re.search(r"_(\d+)\.(?:parquet|csv)$", filename)
    return (int(match.group(1)) if match else 10**9, filename)


def repository_shards(revision: str) -> tuple[list[tuple[str, int]], str]:
    info = HfApi().dataset_info(
        REPOSITORY_ID,
        revision=revision,
        files_metadata=True,
    )
    files = [
        (item.rfilename, int(item.size or 0))
        for item in info.siblings
        if item.rfilename.startswith("data/") and item.rfilename.endswith(".parquet")
    ]
    return sorted(files, key=lambda item: numeric_shard_key(item[0])), str(info.sha)


def free_gib() -> float:
    return shutil.disk_usage(DATASET_ROOT).free / (1024**3)


def image_suffix(image_format: str) -> str:
    return {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tif",
    }.get(image_format.upper(), ".img")


def download_shard(filename: str, revision: str, retries: int) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPOSITORY_ID,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=STAGING_ROOT,
                )
            )
        except Exception as exc:  # network failures are resumable
            last_error = exc
            delay = min(60, 2 ** min(attempt, 6))
            print(
                f"download retry {attempt}/{retries} for {filename}: {exc}; "
                f"waiting {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Failed to download {filename}") from last_error


def existing_export_hashes() -> set[str]:
    values: set[str] = set()
    for path in SHARD_MANIFEST_ROOT.glob("*.csv"):
        values.update(row["sha256"] for row in load_csv(path) if row.get("sha256"))
    return values


def export_shard(
    parquet_path: Path,
    source_filename: str,
    revision: str,
    known_sha256: set[str],
    wildfake_sha256: set[str],
    wildfake_dhash: set[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    shard_name = Path(source_filename).stem
    parquet_file = pq.ParquetFile(parquet_path)
    required = {
        "image_name",
        "image_data",
        "model_name",
        "nsfw_flag",
        "subset",
        "split",
        "label",
    }
    missing = required - set(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(f"{source_filename} is missing columns: {sorted(missing)}")

    columns = [
        name
        for name in (
            "image_name",
            "format",
            "mode",
            "image_data",
            "model_name",
            "nsfw_flag",
            "prompt",
            "real_source",
            "subset",
            "split",
            "label",
            "architecture",
        )
        if name in parquet_file.schema_arrow.names
    ]
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        for source in batch.to_pylist():
            counts["source_rows"] += 1
            if bool(source.get("nsfw_flag", False)):
                counts["excluded_nsfw"] += 1
                continue
            label = int(source["label"])
            if label not in (0, 1):
                counts["excluded_invalid_label"] += 1
                continue
            raw = source.get("image_data")
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                counts["excluded_missing_image"] += 1
                continue
            data = bytes(raw)
            digest = sha256_bytes(data)
            if digest in known_sha256:
                counts["excluded_exact_duplicate"] += 1
                continue

            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    width, height = image.size
                    actual_format = str(image.format or source.get("format") or "UNKNOWN")
                    actual_mode = str(image.mode)
                    dhash = difference_hash(image)
            except Exception:
                counts["excluded_invalid_image"] += 1
                continue

            if digest in wildfake_sha256 or dhash in wildfake_dhash:
                counts["excluded_wildfake_overlap"] += 1
                continue

            official_split = str(source.get("split") or "train").strip().lower()
            allowed = official_split == "train"
            role = "train" if allowed else "source_holdout"
            source_class = "full_synthetic" if label == 1 else "real"
            suffix = image_suffix(actual_format)
            relative_path = Path("CommunityForensics_Small") / "images" / (
                "fake" if label else "real"
            ) / shard_name / f"{digest}{suffix}"
            destination = DATASET_ROOT / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != len(data):
                    raise RuntimeError(f"Existing export has wrong size: {destination}")
            else:
                temporary = destination.with_suffix(destination.suffix + ".part")
                with temporary.open("wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)

            prompt = str(source.get("prompt") or "")
            rows.append(
                {
                    "path": relative_path.as_posix(),
                    "dataset": "CommunityForensics-Small",
                    "official_split": official_split,
                    "role": role,
                    "source_class": source_class,
                    "source_label": label,
                    "binary_label": label,
                    "mask_path": "",
                    "allowed_for_training": str(allowed),
                    "archive_crc32": "",
                    "uncompressed_bytes": len(data),
                    "sha256": digest,
                    "duplicate_group": digest,
                    "generator": str(source.get("model_name") or "unknown"),
                    "source_md5": "",
                    "source_image_path": str(source.get("image_name") or ""),
                    "source_config": str(source.get("subset") or ""),
                    "width": width,
                    "height": height,
                    "image_format": actual_format,
                    "image_mode": actual_mode,
                    "architecture": str(source.get("architecture") or ""),
                    "real_source": str(source.get("real_source") or ""),
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else "",
                    "source_repository": REPOSITORY_ID,
                    "source_revision": revision,
                    "source_license": LICENSE_ID,
                    "source_shard": source_filename,
                }
            )
            known_sha256.add(digest)
            counts[f"exported_label_{label}"] += 1
    return rows, counts


def rebuild_master_manifests() -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(SHARD_MANIFEST_ROOT.glob("*.csv"), key=lambda p: numeric_shard_key(p.name)):
        rows.extend(load_csv(path))
    rows.sort(key=lambda row: (row["binary_label"], row["generator"], row["path"]))
    train = [row for row in rows if row["allowed_for_training"].lower() == "true"]
    holdout = [row for row in rows if row["allowed_for_training"].lower() != "true"]
    write_csv(MASTER_MANIFEST_PATH, rows)
    write_csv(TRAIN_MANIFEST_PATH, train)
    write_csv(HOLDOUT_MANIFEST_PATH, holdout)

    generator_counts = Counter(
        row["generator"] for row in train if row["binary_label"] == "1"
    )
    return {
        "all_rows": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_real": sum(row["binary_label"] == "0" for row in train),
        "train_fake": sum(row["binary_label"] == "1" for row in train),
        "train_fake_generators": len(generator_counts),
        "top_train_fake_generators": generator_counts.most_common(25),
    }


def main() -> None:
    args = parse_args()
    for path in (IMAGE_ROOT, STATE_ROOT, SHARD_MANIFEST_ROOT, STAGING_ROOT, REPORT_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)

    shards, resolved_revision = repository_shards(args.revision)
    if resolved_revision != args.revision:
        raise RuntimeError(
            f"Resolved revision {resolved_revision} does not match pinned {args.revision}"
        )
    if args.only_shard:
        wanted = f"data/{args.only_shard}" if "/" not in args.only_shard else args.only_shard
        shards = [item for item in shards if item[0] == wanted]
        if not shards:
            raise ValueError(f"Unknown shard: {args.only_shard}")

    completed = load_completed_shards()
    unfinished = [item for item in shards if item[0] not in completed]
    if args.max_shards is not None:
        unfinished = unfinished[: args.max_shards]

    protected = protected_sha256_values()
    wildfake_sha256, wildfake_dhash = load_or_build_wildfake_hashes()
    known_sha256 = protected | wildfake_sha256 | existing_export_hashes()
    total_bytes = sum(size for _, size in shards)
    print(
        f"repository={REPOSITORY_ID} revision={resolved_revision}\n"
        f"shards={len(shards)} unfinished_this_run={len(unfinished)} "
        f"source_size={total_bytes / 1024**3:.1f} GiB free={free_gib():.1f} GiB\n"
        f"protected_sha256={len(protected):,} wildfake_dhash={len(wildfake_dhash):,}",
        flush=True,
    )

    run_counts: Counter[str] = Counter()
    for index, (filename, size) in enumerate(unfinished, start=1):
        if free_gib() < args.min_free_gib + size / (1024**3):
            raise RuntimeError(
                f"Insufficient free space before {filename}: {free_gib():.1f} GiB"
            )
        print(
            f"[{index}/{len(unfinished)}] downloading {filename} "
            f"({size / 1024**3:.2f} GiB), free={free_gib():.1f} GiB",
            flush=True,
        )
        parquet_path = download_shard(filename, resolved_revision, args.max_retries)
        rows, counts = export_shard(
            parquet_path,
            filename,
            resolved_revision,
            known_sha256,
            wildfake_sha256,
            wildfake_dhash,
            args.batch_size,
        )
        shard_manifest = SHARD_MANIFEST_ROOT / f"{Path(filename).stem}.csv"
        write_csv(shard_manifest, rows)
        record = {
            "filename": filename,
            "status": "complete",
            "source_bytes": size,
            "manifest": str(shard_manifest.relative_to(PROJECT_ROOT)),
            "counts": dict(counts),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        append_completion(record)
        run_counts.update(counts)
        if not args.keep_parquet:
            parquet_path.unlink(missing_ok=True)
        print(
            f"[{index}/{len(unfinished)}] complete: {dict(counts)}, "
            f"free={free_gib():.1f} GiB",
            flush=True,
        )

    summary = rebuild_master_manifests()
    report = {
        "repository": REPOSITORY_ID,
        "revision": resolved_revision,
        "license": LICENSE_ID,
        "license_url": LICENSE_URL,
        "available_shards": len(shards),
        "completed_shards": len(load_completed_shards()),
        "source_size_bytes": total_bytes,
        "keep_parquet": args.keep_parquet,
        "free_gib_after": free_gib(),
        "run_counts": dict(run_counts),
        "summary": summary,
        "leakage_policy": {
            "exact_sha256_against": [str(path.relative_to(PROJECT_ROOT)) for path in PROTECTED_MANIFESTS]
            + [str(WILDFAKE_MANIFEST.relative_to(PROJECT_ROOT))],
            "exact_dhash_against": str(WILDFAKE_MANIFEST.relative_to(PROJECT_ROOT)),
            "nsfw_rows_excluded": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
