"""Extract the approved SuSy subsets and build audited Tiny vNext manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
SUSY_ROOT = DATASET_ROOT / "SuSy"
REPOSITORY = "aminasifar1/SuSy-Dataset"
REVISION = "df5f324e4438cddaaf0de87f231c356b47aa555d"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SELECTED_SOURCES = {
    "coco": {
        "label": 0,
        "source_class": "real",
        "generator": "",
        "license": "CC-BY-4.0 (per SuSy card; upstream COCO attribution required)",
    },
    "dalle-3-images": {
        "label": 1,
        "source_class": "dalle_3",
        "generator": "DALL-E 3",
        "license": "MIT",
    },
    "diffusiondb": {
        "label": 1,
        "source_class": "stable_diffusion_1x",
        "generator": "Stable Diffusion 1.x",
        "license": "CC0-1.0",
    },
    "midjourney-images": {
        "label": 1,
        "source_class": "midjourney_v5_v6",
        "generator": "Midjourney V5/V6",
        "license": "MIT",
    },
    "midjourney_tti": {
        "label": 1,
        "source_class": "midjourney_v1_v2",
        "generator": "Midjourney V1/V2",
        "license": "CC0 links only; upstream image rights unspecified",
    },
    "realisticSDXL": {
        "label": 1,
        "source_class": "sdxl",
        "generator": "Stable Diffusion XL",
        "license": "CreativeML-OpenRAIL-M",
    },
}
EXCLUDED_SOURCES: dict[str, str] = {}
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
    "generator_assignment",
    "content_group",
    "dhash64",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def selected_member(info: zipfile.ZipInfo, split: str) -> tuple[str, Path] | None:
    path = PurePosixPath(info.filename)
    if info.is_dir() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    parts = path.parts
    if len(parts) < 3 or parts[0] != split:
        return None
    source = parts[1]
    if source not in SELECTED_SOURCES:
        return None
    relative = Path(*parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe archive member: {info.filename}")
    return source, relative


def extract_split(split: str, resume: bool) -> dict[str, Any]:
    archive = SUSY_ROOT / "data" / f"{split}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing verified archive: {archive}")
    destination_root = SUSY_ROOT / "extracted"
    destination_root.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    bytes_written = 0
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for index, info in enumerate(infos, start=1):
            selected = selected_member(info, split)
            if selected is None:
                continue
            source, relative = selected
            target = destination_root / relative
            if target.exists():
                if resume and target.stat().st_size == info.file_size:
                    counts[source] += 1
                    continue
                raise FileExistsError(f"Refusing to overwrite existing file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            with bundle.open(info) as source_handle, temporary.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=4 * 1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError(f"Extracted size mismatch: {info.filename}")
            os.replace(temporary, target)
            counts[source] += 1
            bytes_written += info.file_size
            if sum(counts.values()) % 500 == 0:
                print(
                    f"extract split={split} images={sum(counts.values())} "
                    f"written_gib={bytes_written / 2**30:.2f}",
                    flush=True,
                )
    print(f"extracted split={split} counts={dict(sorted(counts.items()))}", flush=True)
    return {"counts": dict(sorted(counts.items())), "bytes_written": bytes_written}


def metadata_expected_counts() -> dict[str, dict[str, int]]:
    payload = json.loads((SUSY_ROOT / "susy_dataset.json").read_text(encoding="utf-8"))
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        result[split] = {source: len(paths) for source, paths in payload[split].items()}
    return result


def existing_manifest_hashes() -> set[str]:
    hashes: set[str] = set()
    for manifest in sorted((DATASET_ROOT / "manifests").glob("*.csv")):
        if manifest.name.startswith("susy_vnext_"):
            continue
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sha256" not in reader.fieldnames:
                continue
            for row in reader:
                value = row.get("sha256", "").strip().lower()
                if len(value) == 64:
                    hashes.add(value)
    return hashes


def build_rows(split: str, crc_by_member: dict[str, int]) -> list[dict[str, Any]]:
    root = SUSY_ROOT / "extracted" / split
    role = "train" if split == "train" else "vnext_development"
    rows: list[dict[str, Any]] = []
    for source, source_meta in SELECTED_SOURCES.items():
        source_root = root / source
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing extracted source directory: {source_root}")
        for image_path in sorted(
            path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            relative_dataset = image_path.relative_to(DATASET_ROOT).as_posix()
            archive_member = f"{split}/{source}/{image_path.relative_to(source_root).as_posix()}"
            digest = file_sha256(image_path)
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format or ""
                image_mode = image.mode
                perceptual = dhash64(image)
            rows.append(
                {
                    "path": relative_dataset,
                    "dataset": "SuSy",
                    "official_split": split,
                    "role": role,
                    "source_class": source_meta["source_class"],
                    "source_label": source,
                    "binary_label": source_meta["label"],
                    "mask_path": "",
                    "allowed_for_training": split == "train",
                    "archive_crc32": f"{crc_by_member[archive_member]:08x}",
                    "uncompressed_bytes": image_path.stat().st_size,
                    "sha256": digest,
                    "duplicate_group": digest,
                    "generator": source_meta["generator"],
                    "source_md5": "",
                    "source_image_path": archive_member,
                    "source_config": source,
                    "width": width,
                    "height": height,
                    "image_format": image_format,
                    "image_mode": image_mode,
                    "architecture": "",
                    "real_source": "COCO" if source == "coco" else "",
                    "prompt_sha256": "",
                    "source_repository": REPOSITORY,
                    "source_revision": REVISION,
                    "source_license": source_meta["license"],
                    "source_shard": f"data/{split}.zip",
                    "generator_assignment": "train" if split == "train" else "development",
                    "content_group": f"{source}:{image_path.stem}",
                    "dhash64": perceptual,
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    expected = metadata_expected_counts()
    extraction: dict[str, Any] = {}
    for split in args.splits:
        extraction[split] = extract_split(split, args.resume)
        actual = extraction[split]["counts"]
        selected_expected = {source: expected[split][source] for source in SELECTED_SOURCES}
        if actual != selected_expected:
            raise RuntimeError(
                f"Selected-source count mismatch for {split}: expected={selected_expected} actual={actual}"
            )

    crc_maps: dict[str, dict[str, int]] = {}
    for split in args.splits:
        with zipfile.ZipFile(SUSY_ROOT / "data" / f"{split}.zip") as bundle:
            crc_maps[split] = {info.filename: info.CRC for info in bundle.infolist()}

    rows_by_split = {split: build_rows(split, crc_maps[split]) for split in args.splits}
    known_hashes = existing_manifest_hashes()
    overlaps_existing = {
        split: sorted(row["path"] for row in rows if row["sha256"] in known_hashes)
        for split, rows in rows_by_split.items()
    }
    for split, overlaps in overlaps_existing.items():
        if overlaps:
            raise RuntimeError(f"Exact SHA-256 overlap with existing manifests in {split}: {overlaps[:10]}")

    if {"train", "val"}.issubset(rows_by_split):
        train_hashes = {row["sha256"] for row in rows_by_split["train"]}
        val_hashes = {row["sha256"] for row in rows_by_split["val"]}
        sha_overlap = sorted(train_hashes & val_hashes)
        if sha_overlap:
            raise RuntimeError(f"SuSy train/val exact overlap: {sha_overlap[:10]}")
    else:
        sha_overlap = []

    dhash_locations: dict[str, list[str]] = defaultdict(list)
    for split, rows in rows_by_split.items():
        for row in rows:
            dhash_locations[row["dhash64"]].append(f"{split}:{row['path']}")
    cross_split_dhash = {
        key: locations
        for key, locations in dhash_locations.items()
        if len({location.split(":", 1)[0] for location in locations}) > 1
    }

    manifest_names = {"train": "susy_vnext_train.csv", "val": "susy_vnext_dev.csv"}
    for split, rows in rows_by_split.items():
        write_manifest(DATASET_ROOT / "manifests" / manifest_names[split], rows)

    summary = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "selected_sources": SELECTED_SOURCES,
        "excluded_sources": EXCLUDED_SOURCES,
        "expected_counts": expected,
        "extraction": extraction,
        "manifest_counts": {
            split: {
                "total": len(rows),
                "by_source": dict(sorted(Counter(row["source_label"] for row in rows).items())),
                "by_label": dict(sorted(Counter(str(row["binary_label"]) for row in rows).items())),
            }
            for split, rows in rows_by_split.items()
        },
        "exact_overlap_with_existing_manifests": overlaps_existing,
        "train_val_exact_sha_overlap": sha_overlap,
        "train_val_dhash_collision_count": len(cross_split_dhash),
        "train_val_dhash_collisions": cross_split_dhash,
    }
    audit_path = SUSY_ROOT / "audit" / "preparation_summary.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"complete audit={audit_path}", flush=True)
    print(json.dumps(summary["manifest_counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
