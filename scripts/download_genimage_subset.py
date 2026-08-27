from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The Hub defaults to 10-second reads, which is too short for multi-hundred-MB
# Arrow shards on ordinary connections. These must be set before importing
# datasets/huggingface_hub.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from datasets import Dataset as ArrowDataset
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = DATASET_ROOT / "GenImage_subset"
STATE_ROOT = OUTPUT_ROOT / "_state"
STATE_PATH = STATE_ROOT / "download_state.jsonl"
PROCESSED_SHARDS_PATH = STATE_ROOT / "processed_shards.jsonl"
CACHE_ROOT = DATASET_ROOT / "_hf_cache"
SHARD_CACHE_ROOT = DATASET_ROOT / "_hf_cache_shards"
MANIFEST_ROOT = DATASET_ROOT / "manifests"
REPOSITORY_ID = "nebula/GenImage-arrow"
REPOSITORY_REVISION = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"
LICENSE_URL = "https://github.com/GenImage-Dataset/GenImage/blob/main/License"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
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
]
CONFIG_DIRECTORIES = {
    "midjourney-train": "Midjourney",
    "sd14-train": "stable_diffusion_v_1_4",
    "sd15-train": "stable_diffusion_v_1_5",
    "adm-train": "ADM",
    "biggan-train": "BigGAN",
    "vqdm-train": "VQDM",
    "wukong-train": "wukong",
    "glide-train": "glide",
}


@dataclass(frozen=True)
class Target:
    key: str
    config: str
    generator: str
    label: int
    requested: int
    relative_dir: str
    role: str
    allowed_for_training: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and export a resumable, deduplicated GenImage subset."
    )
    parser.add_argument("--fake-per-generator", type=int, default=20_000)
    parser.add_argument("--real-count", type=int, default=140_000)
    parser.add_argument("--val-per-class", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Read shards sequentially; intended only for connectivity tests.",
    )
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument(
        "--transport",
        choices=("shard", "stream"),
        default="shard",
        help="Use fast Xet shard downloads by default; stream is a fallback.",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that you reviewed and accept the GenImage license.",
    )
    args = parser.parse_args()
    for name in ("fake_per_generator", "real_count", "val_per_class"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.shuffle_buffer < 1:
        parser.error("--shuffle-buffer must be positive")
    if not args.accept_license:
        parser.error(
            "Review the GenImage non-commercial license, then rerun with "
            f"--accept-license: {LICENSE_URL}"
        )
    return args


def build_targets(args: argparse.Namespace) -> list[Target]:
    generator_configs = [
        ("Midjourney", "midjourney-train"),
        ("SD1.4", "sd14-train"),
        ("SD1.5", "sd15-train"),
        ("ADM", "adm-train"),
        ("BigGAN", "biggan-train"),
        ("VQDM", "vqdm-train"),
        ("Wukong", "wukong-train"),
    ]
    targets = [
        Target(
            key=f"train_fake_{generator.lower().replace('.', '')}",
            config=config,
            generator=generator,
            label=1,
            requested=args.fake_per_generator,
            relative_dir=f"train/fake/{generator}",
            role="train",
            allowed_for_training=True,
        )
        for generator, config in generator_configs
    ]
    targets.extend(
        [
            Target(
                key="train_real_imagenet",
                config="biggan-train",
                generator="ImageNet",
                label=0,
                requested=args.real_count,
                relative_dir="train/real/ImageNet",
                role="train",
                allowed_for_training=True,
            ),
            Target(
                key="validation_glide_fake",
                config="glide-train",
                generator="GLIDE",
                label=1,
                requested=args.val_per_class,
                relative_dir="validation/GLIDE/fake",
                role="cross_generator_validation",
                allowed_for_training=False,
            ),
            Target(
                key="validation_glide_real",
                config="glide-train",
                generator="GLIDE",
                label=0,
                requested=args.val_per_class,
                relative_dir="validation/GLIDE/real",
                role="cross_generator_validation",
                allowed_for_training=False,
            ),
        ]
    )
    return targets


def load_state() -> list[dict[str, Any]]:
    if not STATE_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with STATE_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: ignoring incomplete state line {line_number}", flush=True)
    return entries


def load_processed_shards() -> set[str]:
    if not PROCESSED_SHARDS_PATH.exists():
        return set()
    processed: set[str] = set()
    with PROCESSED_SHARDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                processed.add(value)
    return processed


def extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    raise TypeError(f"Unsupported image payload: {type(value).__name__}")


def inspect_image(blob: bytes) -> tuple[int, int, str | None]:
    with Image.open(io.BytesIO(blob)) as image:
        width, height = image.size
        image_format = image.format
        image.verify()
    return width, height, image_format


def choose_suffix(source_path: str, image_format: str | None) -> str:
    suffix = Path(source_path).suffix.lower()
    if suffix in ALLOWED_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix
    fallback = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }
    return fallback.get(str(image_format).upper(), ".img")


def stable_config_seed(base_seed: int, config: str) -> int:
    offset = int(hashlib.sha256(config.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + offset) % (2**32)


def atomic_write(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != len(blob):
            raise RuntimeError(f"Existing file has unexpected size: {path}")
        return
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(entries, key=lambda item: (item["generator"], item["path"])):
            writer.writerow(entry)
    os.replace(temporary, path)


def write_outputs(entries: list[dict[str, Any]], targets: list[Target]) -> None:
    train_entries = [entry for entry in entries if entry["role"] == "train"]
    validation_entries = [
        entry for entry in entries if entry["role"] == "cross_generator_validation"
    ]
    write_csv(MANIFEST_ROOT / "genimage_train.csv", train_entries)
    write_csv(MANIFEST_ROOT / "genimage_glide_validation.csv", validation_entries)

    counts = Counter(entry["target"] for entry in entries)
    summary = {
        "repository": REPOSITORY_ID,
        "license": LICENSE_URL,
        "state_path": str(STATE_PATH),
        "total_saved": len(entries),
        "train_saved": len(train_entries),
        "validation_saved": len(validation_entries),
        "targets": {
            target.key: {
                "config": target.config,
                "generator": target.generator,
                "label": target.label,
                "requested": target.requested,
                "saved": counts[target.key],
                "complete": counts[target.key] >= target.requested,
            }
            for target in targets
        },
    }
    summary_path = MANIFEST_ROOT / "genimage_download_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, summary_path)


def export_config(
    config: str,
    config_targets: list[Target],
    counts: Counter[str],
    seen_sha256: set[str],
    seen_md5: set[str],
    entries: list[dict[str, Any]],
    state_handle: Any,
    args: argparse.Namespace,
) -> None:
    pending_by_label = {target.label: target for target in config_targets}
    if all(counts[target.key] >= target.requested for target in config_targets):
        print(f"[{config}] already complete", flush=True)
        return

    for attempt in range(1, args.max_retries + 1):
        scanned = 0
        try:
            print(f"[{config}] opening stream (attempt {attempt}/{args.max_retries})", flush=True)
            stream = load_dataset(
                REPOSITORY_ID,
                config,
                split="train",
                streaming=True,
                cache_dir=str(CACHE_ROOT),
            )
            stream = stream.decode(False)
            if not args.no_shuffle:
                stream = stream.shuffle(
                    seed=stable_config_seed(args.seed, config),
                    buffer_size=args.shuffle_buffer,
                )
            for row in stream:
                scanned += 1
                if all(counts[target.key] >= target.requested for target in config_targets):
                    print(f"[{config}] complete after scanning {scanned:,} rows", flush=True)
                    return
                try:
                    label = int(row["label"])
                except (KeyError, TypeError, ValueError):
                    continue
                target = pending_by_label.get(label)
                if target is None or counts[target.key] >= target.requested:
                    continue

                source_md5 = str(row.get("md5") or "").strip().lower()
                if source_md5 and source_md5 in seen_md5:
                    continue
                try:
                    blob = extract_image_bytes(row.get("image"))
                    sha256 = hashlib.sha256(blob).hexdigest()
                    if sha256 in seen_sha256:
                        continue
                    computed_md5 = hashlib.md5(blob, usedforsecurity=False).hexdigest()
                    if source_md5 and computed_md5 != source_md5:
                        print(
                            f"[{config}] warning: MD5 mismatch for {row.get('image_path')}",
                            flush=True,
                        )
                        continue
                    width, height, image_format = inspect_image(blob)
                except Exception as exc:
                    print(
                        f"[{config}] warning: invalid image {row.get('image_path')}: {exc}",
                        flush=True,
                    )
                    continue

                source_path = str(row.get("image_path") or "")
                suffix = choose_suffix(source_path, image_format)
                destination = OUTPUT_ROOT / target.relative_dir / f"{sha256}{suffix}"
                atomic_write(destination, blob)
                relative_path = destination.relative_to(DATASET_ROOT).as_posix()
                entry = {
                    "target": target.key,
                    "path": relative_path,
                    "dataset": "GenImage",
                    "official_split": "train",
                    "role": target.role,
                    "source_class": "full_synthetic" if label == 1 else "real",
                    "source_label": label,
                    "binary_label": label,
                    "mask_path": "",
                    "allowed_for_training": target.allowed_for_training,
                    "archive_crc32": "",
                    "uncompressed_bytes": len(blob),
                    "sha256": sha256,
                    "duplicate_group": sha256,
                    "generator": target.generator,
                    "source_md5": source_md5 or computed_md5,
                    "source_image_path": source_path,
                    "source_config": config,
                    "width": int(row.get("width") or width),
                    "height": int(row.get("height") or height),
                }
                state_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                state_handle.flush()
                entries.append(entry)
                counts[target.key] += 1
                seen_sha256.add(sha256)
                seen_md5.add(entry["source_md5"])

                saved = counts[target.key]
                if saved == 1 or saved % 250 == 0 or saved == target.requested:
                    print(
                        f"[{config}] {target.key}: {saved:,}/{target.requested:,} "
                        f"(scanned {scanned:,})",
                        flush=True,
                    )
            missing = {
                target.key: target.requested - counts[target.key]
                for target in config_targets
                if counts[target.key] < target.requested
            }
            raise RuntimeError(f"stream exhausted before quotas were met: {missing}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt >= args.max_retries:
                raise RuntimeError(f"[{config}] failed after {attempt} attempts") from exc
            delay = min(30, 2 ** min(attempt, 5))
            print(f"[{config}] stream error: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def export_config_sharded(
    config: str,
    config_targets: list[Target],
    repository_files: list[str],
    processed_shards: set[str],
    processed_handle: Any,
    counts: Counter[str],
    seen_sha256: set[str],
    seen_md5: set[str],
    entries: list[dict[str, Any]],
    state_handle: Any,
    args: argparse.Namespace,
) -> None:
    if all(counts[target.key] >= target.requested for target in config_targets):
        print(f"[{config}] already complete", flush=True)
        return

    directory = CONFIG_DIRECTORIES[config]
    prefix = f"data/train/{directory}/"
    shards = sorted(
        filename
        for filename in repository_files
        if filename.startswith(prefix) and filename.endswith(".arrow")
    )
    if not shards:
        raise RuntimeError(f"No Arrow shards found for {config} at {prefix}")
    random.Random(stable_config_seed(args.seed, config)).shuffle(shards)
    pending_by_label = {target.label: target for target in config_targets}
    print(
        f"[{config}] shard mode: {len(shards):,} remote shards, "
        f"{sum(shard in processed_shards for shard in shards):,} already processed",
        flush=True,
    )

    for shard_number, filename in enumerate(shards, start=1):
        if all(counts[target.key] >= target.requested for target in config_targets):
            print(f"[{config}] all quotas complete", flush=True)
            return
        if filename in processed_shards:
            continue
        if shutil.disk_usage(DATASET_ROOT).free < 20 * 1024**3:
            raise RuntimeError("Less than 20 GiB remains on the dataset volume")

        local_path: str | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                print(
                    f"[{config}] downloading shard {shard_number}/{len(shards)}: "
                    f"{Path(filename).name}",
                    flush=True,
                )
                local_path = hf_hub_download(
                    repo_id=REPOSITORY_ID,
                    filename=filename,
                    repo_type="dataset",
                    revision=REPOSITORY_REVISION,
                    cache_dir=str(SHARD_CACHE_ROOT),
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if attempt >= args.max_retries:
                    raise RuntimeError(f"Failed to download shard {filename}") from exc
                delay = min(30, 2 ** min(attempt, 5))
                print(
                    f"[{config}] shard error: {exc}; retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        if local_path is None:
            raise RuntimeError(f"No local path returned for {filename}")

        dataset = ArrowDataset.from_file(local_path).with_format(None)
        saved_before = sum(counts[target.key] for target in config_targets)
        for row_index in range(len(dataset)):
            if all(counts[target.key] >= target.requested for target in config_targets):
                break
            row = dataset[row_index]
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError):
                continue
            target = pending_by_label.get(label)
            if target is None or counts[target.key] >= target.requested:
                continue

            source_md5 = str(row.get("md5") or "").strip().lower()
            if source_md5 and source_md5 in seen_md5:
                continue
            try:
                blob = extract_image_bytes(row.get("image"))
                sha256 = hashlib.sha256(blob).hexdigest()
                if sha256 in seen_sha256:
                    continue
                computed_md5 = hashlib.md5(blob, usedforsecurity=False).hexdigest()
                if source_md5 and computed_md5 != source_md5:
                    print(
                        f"[{config}] warning: MD5 mismatch for {row.get('image_path')}",
                        flush=True,
                    )
                    continue
                width, height, image_format = inspect_image(blob)
            except Exception as exc:
                print(
                    f"[{config}] warning: invalid image {row.get('image_path')}: {exc}",
                    flush=True,
                )
                continue

            source_path = str(row.get("image_path") or "")
            suffix = choose_suffix(source_path, image_format)
            destination = OUTPUT_ROOT / target.relative_dir / f"{sha256}{suffix}"
            atomic_write(destination, blob)
            entry = {
                "target": target.key,
                "path": destination.relative_to(DATASET_ROOT).as_posix(),
                "dataset": "GenImage",
                "official_split": "train",
                "role": target.role,
                "source_class": "full_synthetic" if label == 1 else "real",
                "source_label": label,
                "binary_label": label,
                "mask_path": "",
                "allowed_for_training": target.allowed_for_training,
                "archive_crc32": "",
                "uncompressed_bytes": len(blob),
                "sha256": sha256,
                "duplicate_group": sha256,
                "generator": target.generator,
                "source_md5": source_md5 or computed_md5,
                "source_image_path": source_path,
                "source_config": config,
                "width": int(row.get("width") or width),
                "height": int(row.get("height") or height),
            }
            state_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            state_handle.flush()
            entries.append(entry)
            counts[target.key] += 1
            seen_sha256.add(sha256)
            seen_md5.add(entry["source_md5"])

            saved = counts[target.key]
            if saved == 1 or saved % 250 == 0 or saved == target.requested:
                print(
                    f"[{config}] {target.key}: {saved:,}/{target.requested:,}",
                    flush=True,
                )

        del dataset
        processed_handle.write(filename + "\n")
        processed_handle.flush()
        processed_shards.add(filename)
        saved_after = sum(counts[target.key] for target in config_targets)
        print(
            f"[{config}] shard extracted: +{saved_after - saved_before:,} images",
            flush=True,
        )

    missing = {
        target.key: target.requested - counts[target.key]
        for target in config_targets
        if counts[target.key] < target.requested
    }
    if missing:
        raise RuntimeError(f"All shards exhausted before quotas were met: {missing}")


def main() -> None:
    args = parse_args()
    targets = build_targets(args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SHARD_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    free_gib = shutil.disk_usage(DATASET_ROOT).free / (1024**3)
    print(f"free space on dataset volume: {free_gib:.1f} GiB", flush=True)
    if free_gib < 80:
        print("warning: less than 80 GiB free; the selected export may not fit", flush=True)

    entries = load_state()
    counts: Counter[str] = Counter(entry["target"] for entry in entries)
    seen_sha256 = {str(entry["sha256"]) for entry in entries}
    seen_md5 = {str(entry.get("source_md5") or "") for entry in entries}
    seen_md5.discard("")

    print(f"resuming from {len(entries):,} completed images", flush=True)
    grouped: dict[str, list[Target]] = defaultdict(list)
    for target in targets:
        grouped[target.config].append(target)

    interrupted = False
    try:
        with STATE_PATH.open("a", encoding="utf-8", buffering=1) as state_handle:
            if args.transport == "shard":
                repository_files = HfApi().list_repo_files(
                    REPOSITORY_ID,
                    repo_type="dataset",
                    revision=REPOSITORY_REVISION,
                )
                processed_shards = load_processed_shards()
                with PROCESSED_SHARDS_PATH.open(
                    "a", encoding="utf-8", buffering=1
                ) as processed_handle:
                    for config, config_targets in grouped.items():
                        export_config_sharded(
                            config,
                            config_targets,
                            repository_files,
                            processed_shards,
                            processed_handle,
                            counts,
                            seen_sha256,
                            seen_md5,
                            entries,
                            state_handle,
                            args,
                        )
            else:
                for config, config_targets in grouped.items():
                    export_config(
                        config,
                        config_targets,
                        counts,
                        seen_sha256,
                        seen_md5,
                        entries,
                        state_handle,
                        args,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("download interrupted; saved progress is resumable", flush=True)
    finally:
        write_outputs(entries, targets)

    print("\ncurrent totals:", flush=True)
    for target in targets:
        print(f"  {target.key}: {counts[target.key]:,}/{target.requested:,}", flush=True)
    print(f"  all saved: {len(entries):,}", flush=True)
    if interrupted:
        raise SystemExit(130)
    if any(counts[target.key] < target.requested for target in targets):
        raise SystemExit("download stopped before all quotas were met")
    print("GenImage subset download complete", flush=True)


if __name__ == "__main__":
    main()
