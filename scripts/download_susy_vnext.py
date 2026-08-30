"""Download and verify the pinned SuSy archives used by Tiny vNext.

The downloader deliberately does not extract archives or touch the sealed test
split.  Extraction and manifest creation are separate audited steps so that a
partial download can never be mistaken for a prepared dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "aminasifar1/SuSy-Dataset"
REVISION = "df5f324e4438cddaaf0de87f231c356b47aa555d"
FILES = {
    "train": {
        "filename": "data/train.zip",
        "size": 15_189_567_725,
        "sha256": "5c349404c2f2fff72348ace1c450b92e5bdf1a9d6508a5c6cc38bbfe002ecdc8",
    },
    "val": {
        "filename": "data/val.zip",
        "size": 4_725_427_667,
        "sha256": "c41beef0811e5dd70db3923290ee1fc57a73eb312bef9de5ad573850578775ff",
    },
    "test": {
        "filename": "data/test.zip",
        "size": 6_020_979_522,
        "sha256": "1cf4d744686274ce7b7624f83cb04247792d6ad48d1f7d4e944faa880ebfe242",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(FILES),
        default=["train", "val"],
        help="Download train/val by default. The sealed test split is opt-in.",
    )
    parser.add_argument("--output-dir", default="Dataset/SuSy")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    receipt: dict[str, object] = {
        "repository": REPO_ID,
        "revision": REVISION,
        "requested_splits": list(args.splits),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    receipt_path = output_dir / "download_receipt.json"
    for split in args.splits:
        expected = FILES[split]
        print(f"download split={split} file={expected['filename']} revision={REVISION}", flush=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=str(expected["filename"]),
                revision=REVISION,
                local_dir=output_dir,
            )
        )
        actual_size = downloaded.stat().st_size
        if actual_size != expected["size"]:
            raise RuntimeError(
                f"Size mismatch for {downloaded}: expected={expected['size']} actual={actual_size}"
            )
        print(f"verify sha256 split={split} bytes={actual_size}", flush=True)
        actual_hash = sha256_file(downloaded)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {downloaded}: expected={expected['sha256']} actual={actual_hash}"
            )
        receipt["files"][split] = {
            "path": str(downloaded),
            "size": actual_size,
            "sha256": actual_hash,
            "verified": True,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"verified split={split} sha256={actual_hash}", flush=True)

    receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"complete receipt={receipt_path}", flush=True)


if __name__ == "__main__":
    main()
