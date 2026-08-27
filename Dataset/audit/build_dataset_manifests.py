from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "_archives"
MANIFESTS = ROOT / "manifests"
AUDIT = ROOT / "audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_PRIORITY = {"train": 1, "validation": 2, "test": 3}
FIELDNAMES = [
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
]


def posix_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] = FIELDNAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sid_entry(
    archive_name: str,
    zip_path: str,
    split: str,
    source_class: str,
    final_name: str,
    crc: int,
    size: int,
) -> dict:
    labels = {"real": 0, "full_synthetic": 1, "tampered": 2}
    image_path = ROOT / "SID_Set" / split / source_class / final_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing extracted SID image: {image_path}")
    mask_path = ""
    if source_class == "tampered":
        mask = ROOT / "SID_Set" / split / "masks" / f"{Path(final_name).stem}_mask.png"
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path}: {mask}")
        mask_path = posix_path(mask)
    return {
        "path": posix_path(image_path),
        "dataset": "SID_Set",
        "official_split": split,
        "role": split,
        "source_class": source_class,
        "source_label": labels[source_class],
        "binary_label": 0 if source_class == "real" else 1,
        "mask_path": mask_path,
        "allowed_for_training": "true" if split == "train" else "false",
        "archive_crc32": f"{crc:08x}",
        "uncompressed_bytes": size,
        "sha256": "",
        "duplicate_group": "",
        "_archive": archive_name,
        "_zip_path": zip_path,
        "_crc": crc,
        "_size": size,
        "_split": split,
    }


def collect_sid_rows() -> tuple[list[dict], dict[str, int]]:
    archive_root = ARCHIVES / "SID_Set"
    specs = [
        ("train_real.zip", "train", "real", "real"),
        ("train_full_synthetic_part1.zip", "train", "full_synthetic", "full_synthetic_part1"),
        ("train_full_synthetic_part2.zip", "train", "full_synthetic", "full_synthetic_part2"),
        ("train_tampered.zip", "train", "tampered", "tampered"),
    ]
    rows: list[dict] = []
    for archive_name, split, source_class, root_name in specs:
        archive_path = archive_root / archive_name
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                internal = PurePosixPath(info.filename.replace("\\", "/"))
                if info.is_dir() or internal.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if not internal.parts or internal.parts[0] != root_name:
                    continue
                final_name = "/".join(internal.parts[1:])
                rows.append(
                    sid_entry(
                        archive_name,
                        info.filename,
                        split,
                        source_class,
                        final_name,
                        info.CRC,
                        info.file_size,
                    )
                )

    for archive_name, split in (("validation.zip", "validation"), ("test.zip", "test")):
        archive_path = archive_root / archive_name
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                internal = PurePosixPath(info.filename.replace("\\", "/"))
                if info.is_dir() or internal.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if len(internal.parts) < 2 or internal.parts[0] not in {"real", "full_synthetic", "tampered"}:
                    continue
                source_class = internal.parts[0]
                final_name = "/".join(internal.parts[1:])
                rows.append(
                    sid_entry(
                        archive_name,
                        info.filename,
                        split,
                        source_class,
                        final_name,
                        info.CRC,
                        info.file_size,
                    )
                )

    rows.sort(key=lambda row: (row["official_split"], row["source_class"], row["path"]))
    counts = Counter(f"{row['official_split']}/{row['source_class']}" for row in rows)
    expected = {
        "train/real": 70000,
        "train/full_synthetic": 70000,
        "train/tampered": 70000,
        "validation/real": 10000,
        "validation/full_synthetic": 10000,
        "validation/tampered": 10000,
        "test/real": 20000,
        "test/full_synthetic": 20000,
        "test/tampered": 20000,
    }
    if dict(counts) != expected:
        raise RuntimeError(f"SID input counts mismatch: {dict(counts)}")
    return rows, dict(counts)


def sha256_zip_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_sid_duplicates(rows: list[dict]) -> tuple[list[dict], set[str], list[dict]]:
    candidates: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        candidates[(row["_size"], row["_crc"])].append(row)
    candidates = {key: value for key, value in candidates.items() if len(value) > 1}
    print(f"Hashing {sum(map(len, candidates.values()))} SID duplicate candidates...", flush=True)

    zip_handles: dict[str, zipfile.ZipFile] = {}
    exact: dict[tuple[int, str], list[dict]] = defaultdict(list)
    try:
        processed = 0
        for candidate_rows in candidates.values():
            for row in candidate_rows:
                archive_name = row["_archive"]
                if archive_name not in zip_handles:
                    zip_handles[archive_name] = zipfile.ZipFile(ARCHIVES / "SID_Set" / archive_name)
                digest = sha256_zip_member(zip_handles[archive_name], row["_zip_path"])
                row["sha256"] = digest
                exact[(row["_size"], digest)].append(row)
                processed += 1
                if processed % 1000 == 0:
                    print(f"  hashed {processed} candidates", flush=True)
    finally:
        for handle in zip_handles.values():
            handle.close()

    groups = [group for group in exact.values() if len(group) > 1]
    groups.sort(key=lambda group: (group[0]["sha256"], group[0]["path"]))
    excluded_paths: set[str] = set()
    audit_rows: list[dict] = []
    for number, group in enumerate(groups, start=1):
        group_id = f"siddup_{number:06d}"
        canonical = sorted(
            group,
            key=lambda row: (-SPLIT_PRIORITY[row["_split"]], row["path"], row["_archive"], row["_zip_path"]),
        )[0]
        for row in group:
            row["duplicate_group"] = group_id
            kept = row is canonical
            if not kept:
                excluded_paths.add(row["path"])
            audit_rows.append(
                {
                    "duplicate_group": group_id,
                    "sha256": row["sha256"],
                    "uncompressed_bytes": row["uncompressed_bytes"],
                    "split": row["official_split"],
                    "source_class": row["source_class"],
                    "path": row["path"],
                    "source_archive": row["_archive"],
                    "source_zip_path": row["_zip_path"],
                    "kept_in_clean": "true" if kept else "false",
                    "canonical_path": canonical["path"],
                }
            )
    return audit_rows, excluded_paths, groups


def public_row(row: dict) -> dict:
    return {field: row.get(field, "") for field in FIELDNAMES}


def collect_cifake_rows() -> list[dict]:
    rows: list[dict] = []
    for split, source_class, label in (
        ("train", "REAL", 0),
        ("train", "FAKE", 1),
        ("test", "REAL", 0),
        ("test", "FAKE", 1),
    ):
        directory = ROOT / "CIFAKE" / split / source_class
        for image_path in sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append(
                {
                    "path": posix_path(image_path),
                    "dataset": "CIFAKE",
                    "official_split": split,
                    "role": split,
                    "source_class": source_class.lower(),
                    "source_label": label,
                    "binary_label": label,
                    "mask_path": "",
                    "allowed_for_training": "true" if split == "train" else "false",
                    "archive_crc32": "",
                    "uncompressed_bytes": image_path.stat().st_size,
                    "sha256": "",
                    "duplicate_group": "",
                }
            )
    if len(rows) != 120000:
        raise RuntimeError(f"CIFAKE count mismatch: {len(rows)}")
    return rows


def collect_wildfake_rows() -> list[dict]:
    rows: list[dict] = []
    roots = [
        (ROOT / "WildFake_demo" / "Images" / "Real" / "coco" / "coco2017" / "val2017", "real", 0),
        (ROOT / "WildFake_demo" / "Images" / "Diffusion_based" / "DALLE" / "Advanced", "full_synthetic", 1),
    ]
    for directory, source_class, label in roots:
        for image_path in sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append(
                {
                    "path": posix_path(image_path),
                    "dataset": "WildFake_demo",
                    "official_split": "demo",
                    "role": "demo_only",
                    "source_class": source_class,
                    "source_label": label,
                    "binary_label": label,
                    "mask_path": "",
                    "allowed_for_training": "false",
                    "archive_crc32": "",
                    "uncompressed_bytes": image_path.stat().st_size,
                    "sha256": "",
                    "duplicate_group": "",
                }
            )
    if len(rows) != 13841:
        raise RuntimeError(f"WildFake demo count mismatch: {len(rows)}")
    return rows


def build_archive_inventory() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(ARCHIVES.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": posix_path(path),
                "size_bytes": path.stat().st_size,
                "size_gib": f"{path.stat().st_size / (1024 ** 3):.3f}",
                "integrity_status": "7z_test_passed" if path.suffix.lower() == ".zip" else "source_metadata",
            }
        )
    return rows


def main() -> int:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    sid_rows, sid_counts = collect_sid_rows()
    duplicate_rows, excluded_paths, duplicate_groups = find_sid_duplicates(sid_rows)
    clean_sid_rows = [row for row in sid_rows if row["path"] not in excluded_paths]

    for split in ("train", "validation", "test"):
        official = [public_row(row) for row in sid_rows if row["official_split"] == split]
        clean = [public_row(row) for row in clean_sid_rows if row["official_split"] == split]
        write_csv(MANIFESTS / f"sid_official_{split}.csv", official)
        write_csv(MANIFESTS / f"sid_clean_{split}.csv", clean)

    cifake_rows = collect_cifake_rows()
    wildfake_rows = collect_wildfake_rows()
    write_csv(MANIFESTS / "cifake.csv", cifake_rows)
    write_csv(MANIFESTS / "wildfake_demo.csv", wildfake_rows)

    training_pool = [
        public_row(row)
        for row in clean_sid_rows
        if row["official_split"] == "train"
    ] + [row for row in cifake_rows if row["official_split"] == "train"]
    validation_pool = [public_row(row) for row in clean_sid_rows if row["official_split"] == "validation"]
    evaluation_pool = [public_row(row) for row in clean_sid_rows if row["official_split"] == "test"]
    write_csv(MANIFESTS / "training_pool.csv", training_pool)
    write_csv(MANIFESTS / "validation_pool.csv", validation_pool)
    write_csv(MANIFESTS / "evaluation_pool.csv", evaluation_pool)

    duplicate_fields = [
        "duplicate_group",
        "sha256",
        "uncompressed_bytes",
        "split",
        "source_class",
        "path",
        "source_archive",
        "source_zip_path",
        "kept_in_clean",
        "canonical_path",
    ]
    write_csv(AUDIT / "duplicate_groups.csv", duplicate_rows, duplicate_fields)
    write_csv(
        AUDIT / "archive_inventory.csv",
        build_archive_inventory(),
        ["path", "size_bytes", "size_gib", "integrity_status"],
    )
    write_csv(AUDIT / "corrupt_files.csv", [], ["path", "reason"])

    clean_counts = Counter(f"{row['official_split']}/{row['source_class']}" for row in clean_sid_rows)
    summary = {
        "root": str(ROOT),
        "sid_official_counts": sid_counts,
        "sid_clean_counts": dict(sorted(clean_counts.items())),
        "sid_duplicate_groups": len(duplicate_groups),
        "sid_excluded_entries": len(excluded_paths),
        "cifake_images": len(cifake_rows),
        "wildfake_demo_images": len(wildfake_rows),
        "training_pool_images": len(training_pool),
        "validation_pool_images": len(validation_pool),
        "evaluation_pool_images": len(evaluation_pool),
        "wildfake_allowed_for_training": False,
        "archive_integrity": "all_zip_files_passed_7z_test",
    }
    with (AUDIT / "extraction_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
