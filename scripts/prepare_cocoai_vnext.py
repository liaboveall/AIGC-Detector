"""Extract MS COCOAI parquet images and build leakage-audited vNext manifests."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
SOURCE_ROOT = DATASET_ROOT / "MS_COCOAI"
REPOSITORY = "Rajarshi-Roy-research/Defactify_Image_Dataset"
REVISION = "787334f7857fa54f29027a7f09c30e895ad486ef"
SOURCE_MAP = {
    0: ("real", "", 0),
    1: ("sd21", "Stable Diffusion 2.1", 1),
    2: ("sdxl", "Stable Diffusion XL", 1),
    3: ("sd3", "Stable Diffusion 3", 1),
    4: ("dalle_3", "DALL-E 3", 1),
    5: ("midjourney_v6", "Midjourney V6", 1),
}
MANIFEST_COLUMNS = [
    "path", "dataset", "official_split", "role", "source_class", "source_label",
    "binary_label", "mask_path", "allowed_for_training", "archive_crc32",
    "uncompressed_bytes", "sha256", "duplicate_group", "generator", "source_md5",
    "source_image_path", "source_config", "width", "height", "image_format",
    "image_mode", "architecture", "real_source", "prompt_sha256", "source_repository",
    "source_revision", "source_license", "source_shard", "generator_assignment",
    "content_group", "dhash64",
]


def file_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalised_prompt_hash(caption: str) -> str:
    normalised = " ".join(str(caption).strip().lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def dhash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def extension_for(image: Image.Image, original_path: str) -> str:
    suffix = Path(original_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return suffix
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tif",
    }.get(image.format or "", ".png")


def existing_manifest_hashes() -> set[str]:
    values: set[str] = set()
    for manifest in sorted((DATASET_ROOT / "manifests").glob("*.csv")):
        if manifest.name.startswith("cocoai_vnext_"):
            continue
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sha256" not in reader.fieldnames:
                continue
            for row in reader:
                digest = row.get("sha256", "").strip().lower()
                if len(digest) == 64:
                    values.add(digest)
    return values


def extract_split(split: str) -> list[dict[str, Any]]:
    parquet_split = "validation" if split == "val" else split
    shards = sorted((SOURCE_ROOT / "data").glob(f"{parquet_split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found for {parquet_split}")
    rows: list[dict[str, Any]] = []
    image_root = SOURCE_ROOT / "images" / split
    image_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=128):
            for local_index, source_row in enumerate(batch.to_pylist()):
                image_record = source_row["Image"]
                image_bytes = image_record.get("bytes")
                if not image_bytes:
                    raise RuntimeError(f"Missing embedded image bytes: {shard} row={row_offset + local_index}")
                label_a = int(source_row["Label_A"])
                label_b = int(source_row["Label_B"])
                if label_b not in SOURCE_MAP:
                    raise RuntimeError(f"Unknown Label_B={label_b}: {shard}")
                source_class, generator, expected_binary = SOURCE_MAP[label_b]
                if label_a != expected_binary:
                    raise RuntimeError(
                        f"Label mismatch Label_A={label_a} Label_B={label_b}: {shard}"
                    )
                prompt_hash = normalised_prompt_hash(source_row["Caption"])
                digest = file_sha256_bytes(image_bytes)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    image_format = image.format or ""
                    image_mode = image.mode
                    perceptual = dhash64(image)
                    suffix = extension_for(image, str(image_record.get("path") or ""))
                sequence = row_offset + local_index
                target = image_root / source_class / f"{shard.stem}_{sequence:06d}_{digest[:12]}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size != len(image_bytes) or file_sha256_bytes(target.read_bytes()) != digest:
                        raise RuntimeError(f"Existing extracted image differs: {target}")
                else:
                    temporary = target.with_name(target.name + ".partial")
                    temporary.write_bytes(image_bytes)
                    os.replace(temporary, target)
                rows.append(
                    {
                        "path": target.relative_to(DATASET_ROOT).as_posix(),
                        "dataset": "MS-COCOAI",
                        "official_split": parquet_split,
                        "role": "train" if split == "train" else "vnext_development",
                        "source_class": source_class,
                        "source_label": label_b,
                        "binary_label": label_a,
                        "mask_path": "",
                        "allowed_for_training": split == "train",
                        "archive_crc32": "",
                        "uncompressed_bytes": len(image_bytes),
                        "sha256": digest,
                        "duplicate_group": digest,
                        "generator": generator,
                        "source_md5": "",
                        "source_image_path": str(image_record.get("path") or ""),
                        "source_config": source_class,
                        "width": width,
                        "height": height,
                        "image_format": image_format,
                        "image_mode": image_mode,
                        "architecture": "",
                        "real_source": "MS COCO" if label_a == 0 else "",
                        "prompt_sha256": prompt_hash,
                        "source_repository": REPOSITORY,
                        "source_revision": REVISION,
                        "source_license": "unspecified on dataset card; included by user decision",
                        "source_shard": f"data/{shard.name}",
                        "generator_assignment": "train" if split == "train" else "development",
                        "content_group": prompt_hash,
                        "dhash64": perceptual,
                    }
                )
                processed += 1
                if processed % 1000 == 0:
                    print(f"extract split={split} rows={processed}", flush=True)
            row_offset += len(batch)
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_exact(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep one deterministic row per exact image hash."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    for row in rows:
        digest = str(row["sha256"])
        if digest in seen:
            removed.append(str(row["path"]))
            continue
        seen.add(digest)
        kept.append(row)
    return kept, removed


def main() -> None:
    receipt = SOURCE_ROOT / "download_receipt.json"
    if not receipt.is_file() or "completed_at_utc" not in json.loads(receipt.read_text(encoding="utf-8")):
        raise RuntimeError("Pinned train/validation download has not completed verification")

    rows_by_split = {split: extract_split(split) for split in ("train", "val")}
    expected_rows = {"train": 42_000, "val": 9_000}
    for split, expected in expected_rows.items():
        if len(rows_by_split[split]) != expected:
            raise RuntimeError(f"Row count mismatch {split}: expected={expected} actual={len(rows_by_split[split])}")

    train_groups = {row["content_group"] for row in rows_by_split["train"]}
    val_groups = {row["content_group"] for row in rows_by_split["val"]}
    prompt_overlap = sorted(train_groups & val_groups)
    if prompt_overlap:
        # Upstream splits contain a small number of repeated captions. Keep the
        # training groups and remove every matching group from development so
        # model selection cannot benefit from shared semantic content.
        overlapping_groups = set(prompt_overlap)
        rows_by_split["val"] = [
            row for row in rows_by_split["val"] if row["content_group"] not in overlapping_groups
        ]
        print(
            f"removed upstream train/val prompt overlap from dev: groups={len(prompt_overlap)}",
            flush=True,
        )

    known_hashes = existing_manifest_hashes()
    contaminated_rows = {
        row["path"]
        for rows in rows_by_split.values()
        for row in rows
        if row["sha256"] in known_hashes
    }
    contaminated_groups = {
        row["content_group"]
        for rows in rows_by_split.values()
        for row in rows
        if row["path"] in contaminated_rows
    }
    filtered = {
        split: [row for row in rows if row["path"] not in contaminated_rows]
        for split, rows in rows_by_split.items()
    }

    # The upstream parquet files contain a small number of byte-identical rows,
    # including some that cross the official split boundary despite differing
    # captions. Keep one deterministic copy within each split, then treat train
    # as authoritative and remove every exact train hash from development.
    filtered["train"], train_internal_exact_removed = deduplicate_exact(filtered["train"])
    filtered["val"], val_internal_exact_removed = deduplicate_exact(filtered["val"])
    filtered_train_hashes = {row["sha256"] for row in filtered["train"]}
    val_cross_split_exact_removed = [
        row["path"] for row in filtered["val"] if row["sha256"] in filtered_train_hashes
    ]
    filtered["val"] = [
        row for row in filtered["val"] if row["sha256"] not in filtered_train_hashes
    ]

    dhash_locations: dict[str, list[str]] = defaultdict(list)
    for split, rows in filtered.items():
        for row in rows:
            dhash_locations[row["dhash64"]].append(f"{split}:{row['path']}")
    cross_split_dhash = {
        key: locations
        for key, locations in dhash_locations.items()
        if len({location.split(":", 1)[0] for location in locations}) > 1
    }

    write_manifest(DATASET_ROOT / "manifests" / "cocoai_vnext_train.csv", filtered["train"])
    write_manifest(DATASET_ROOT / "manifests" / "cocoai_vnext_dev.csv", filtered["val"])
    summary = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license_status": "unspecified on dataset card; included by user decision",
        "raw_counts": {
            split: {
                "total": len(rows),
                "prompt_groups": len({row["content_group"] for row in rows}),
                "by_source": dict(sorted(Counter(row["source_class"] for row in rows).items())),
                "by_label": dict(sorted(Counter(str(row["binary_label"]) for row in rows).items())),
            }
            for split, rows in rows_by_split.items()
        },
        "exact_duplicate_rows_removed": len(contaminated_rows),
        "prompt_groups_affected_by_exact_duplicates": len(contaminated_groups),
        "upstream_internal_exact_duplicates_removed": {
            "train": len(train_internal_exact_removed),
            "val": len(val_internal_exact_removed),
        },
        "upstream_internal_exact_duplicate_paths_removed": {
            "train": train_internal_exact_removed,
            "val": val_internal_exact_removed,
        },
        "train_val_exact_duplicates_removed_from_dev": len(val_cross_split_exact_removed),
        "train_val_exact_duplicate_paths_removed_from_dev": val_cross_split_exact_removed,
        "filtered_counts": {
            split: {
                "total": len(rows),
                "prompt_groups": len({row["content_group"] for row in rows}),
                "by_source": dict(sorted(Counter(row["source_class"] for row in rows).items())),
                "by_label": dict(sorted(Counter(str(row["binary_label"]) for row in rows).items())),
            }
            for split, rows in filtered.items()
        },
        "train_val_prompt_overlap": prompt_overlap,
        "train_val_prompt_overlap_removed_from_dev": len(prompt_overlap),
        "train_val_dhash_collision_count": len(cross_split_dhash),
        "train_val_dhash_collisions": cross_split_dhash,
    }
    audit_path = SOURCE_ROOT / "audit" / "preparation_summary.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"complete audit={audit_path}", flush=True)
    print(json.dumps(summary["filtered_counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
