from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Dataset" / "manifests" / "validation_multisource.csv"
DEFAULT_SELECTION = PROJECT_ROOT / "Dataset" / "manifests" / "validation_selection_6000.csv"
DEFAULT_CONFIRMATION = PROJECT_ROOT / "Dataset" / "manifests" / "validation_confirmation_10000.csv"
DEFAULT_REPORT = PROJECT_ROOT / "Dataset" / "audit" / "robust_validation_split_report.json"

SELECTION_TARGETS = {
    "genimage_glide_real": 1875,
    "genimage_glide_fake": 1875,
    "sid_real": 750,
    "sid_full_synthetic": 750,
    "sid_tampered": 750,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed, mutually exclusive robust-validation selection and confirmation splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--confirmation-output", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def stratum(row: dict[str, str]) -> str:
    dataset = row["dataset"]
    source_class = row["source_class"]
    generator = row["generator"]

    if dataset == "GenImage" and generator == "GLIDE" and source_class == "real":
        return "genimage_glide_real"
    if dataset == "GenImage" and generator == "GLIDE" and source_class == "full_synthetic":
        return "genimage_glide_fake"
    if dataset == "SID_Set" and source_class in {"real", "full_synthetic", "tampered"}:
        return f"sid_{source_class}"
    raise ValueError(
        "Unexpected validation stratum: "
        f"dataset={dataset!r}, generator={generator!r}, source_class={source_class!r}, path={row['path']!r}"
    )


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def count_strata(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = Counter(stratum(row) for row in rows)
    return {name: counts.get(name, 0) for name in SELECTION_TARGETS}


def duplicate_count(rows: list[dict[str, str]], column: str) -> int:
    values = [row[column] for row in rows]
    return len(values) - len(set(values))


def row_digest(fieldnames: list[str], rows: Iterable[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        digest.update("\x1f".join(row[name] for name in fieldnames).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    selection_path = args.selection_output.resolve()
    confirmation_path = args.confirmation_output.resolve()
    report_path = args.report_output.resolve()

    fieldnames, rows = read_manifest(input_path)
    required = {"path", "dataset", "source_class", "generator", "sha256"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(rows) != 16000:
        raise ValueError(f"Expected 16000 input rows, found {len(rows)}")
    if any(not row["path"] or not row["sha256"] for row in rows):
        raise ValueError("Every row must have non-empty path and sha256 fields")
    if duplicate_count(rows, "path") or duplicate_count(rows, "sha256"):
        raise ValueError("Input manifest contains duplicate path or sha256 values")

    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in SELECTION_TARGETS}
    for row in rows:
        grouped[stratum(row)].append(row)

    selection: list[dict[str, str]] = []
    confirmation: list[dict[str, str]] = []
    for offset, (name, target) in enumerate(SELECTION_TARGETS.items()):
        group = list(grouped[name])
        if len(group) < target:
            raise ValueError(f"Stratum {name} needs {target} selection rows, found {len(group)}")
        random.Random(args.seed + offset).shuffle(group)
        selection.extend(group[:target])
        confirmation.extend(group[target:])

    random.Random(args.seed + 100).shuffle(selection)
    random.Random(args.seed + 200).shuffle(confirmation)

    selection_paths = {row["path"] for row in selection}
    confirmation_paths = {row["path"] for row in confirmation}
    selection_sha = {row["sha256"] for row in selection}
    confirmation_sha = {row["sha256"] for row in confirmation}
    path_overlap = selection_paths & confirmation_paths
    sha_overlap = selection_sha & confirmation_sha

    if len(selection) != 6000 or len(confirmation) != 10000:
        raise AssertionError(f"Unexpected output sizes: {len(selection)} and {len(confirmation)}")
    if len(selection) + len(confirmation) != len(rows):
        raise AssertionError("Output row counts do not reconstruct the input manifest")
    if path_overlap or sha_overlap:
        raise AssertionError("Selection and confirmation splits overlap by path or sha256")
    if selection_paths | confirmation_paths != {row["path"] for row in rows}:
        raise AssertionError("Output path union differs from the input manifest")
    if selection_sha | confirmation_sha != {row["sha256"] for row in rows}:
        raise AssertionError("Output sha256 union differs from the input manifest")
    if count_strata(selection) != SELECTION_TARGETS:
        raise AssertionError(f"Selection strata differ from targets: {count_strata(selection)}")

    write_manifest(selection_path, fieldnames, selection)
    write_manifest(confirmation_path, fieldnames, confirmation)

    report = {
        "policy": (
            "The 6,000-row selection split may be used for checkpoint selection and one allowed "
            "hyperparameter retry. The 10,000-row confirmation split remains sealed until a candidate is fixed."
        ),
        "seed": args.seed,
        "input": {
            "path": str(input_path),
            "rows": len(rows),
            "columns": fieldnames,
            "strata": count_strata(rows),
            "duplicate_paths": duplicate_count(rows, "path"),
            "duplicate_sha256": duplicate_count(rows, "sha256"),
            "content_digest_sha256": row_digest(fieldnames, rows),
        },
        "selection": {
            "path": str(selection_path),
            "rows": len(selection),
            "strata": count_strata(selection),
            "duplicate_paths": duplicate_count(selection, "path"),
            "duplicate_sha256": duplicate_count(selection, "sha256"),
            "content_digest_sha256": row_digest(fieldnames, selection),
        },
        "confirmation": {
            "path": str(confirmation_path),
            "rows": len(confirmation),
            "strata": count_strata(confirmation),
            "duplicate_paths": duplicate_count(confirmation, "path"),
            "duplicate_sha256": duplicate_count(confirmation, "sha256"),
            "content_digest_sha256": row_digest(fieldnames, confirmation),
        },
        "audit": {
            "rows_sum_matches_input": len(selection) + len(confirmation) == len(rows),
            "path_overlap_count": len(path_overlap),
            "sha256_overlap_count": len(sha_overlap),
            "path_union_matches_input": selection_paths | confirmation_paths == {row["path"] for row in rows},
            "sha256_union_matches_input": selection_sha | confirmation_sha == {row["sha256"] for row in rows},
            "columns_preserved": True,
            "selection_targets_met": count_strata(selection) == SELECTION_TARGETS,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
