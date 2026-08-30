"""Download and verify pinned MS COCOAI train/validation parquet shards."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
REVISION = "787334f7857fa54f29027a7f09c30e895ad486ef"
FILES = {
    "data/train-00000-of-00007.parquet": (448_176_166, "d18283f316c9a34b6dfae6ddd5b9e6f382a8da0c5928a1ba434f34b6493c2706"),
    "data/train-00001-of-00007.parquet": (445_126_447, "b8366ca6c318fa9b24f6b25ed109f288a95ac583e67f6738824e9b1f504e455a"),
    "data/train-00002-of-00007.parquet": (449_640_886, "1caa42eafb07ee509ea84ba7c700360e19a677cbde97040e4c3d1a3ec0460744"),
    "data/train-00003-of-00007.parquet": (456_221_402, "2a14cdc625b1a5fa7e60498512e6b2555a243620471ed5278224286d1bce42c3"),
    "data/train-00004-of-00007.parquet": (458_610_596, "0bfabd173ccad9622a04cdee858540fd88de4ac24102ad390cfd27c8aa81a594"),
    "data/train-00005-of-00007.parquet": (453_342_171, "704f67e5bca246850994a1543fbb0cae5b17ee9c4740d24fb12c2da29345aacc"),
    "data/train-00006-of-00007.parquet": (448_412_083, "c537c8a772ad70e048b52aa3c785ad1474ec77b9e2455cccda048e12f16fc5a2"),
    "data/validation-00000-of-00002.parquet": (333_255_675, "f759e276b9dbb6965a427ec54384f608bcef9336983d77b9d2c70affef35ab8c"),
    "data/validation-00001-of-00002.parquet": (344_988_959, "429ee1ff44d0c9e913c624532c0014cafc55ea6a9cc48a8ac8d820ba435beae5"),
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output_dir = PROJECT_ROOT / "Dataset" / "MS_COCOAI"
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "repository": REPO_ID,
        "revision": REVISION,
        "license_status": "dataset card/API does not declare a license; included by user decision",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    receipt_path = output_dir / "download_receipt.json"
    for filename, (expected_size, expected_hash) in FILES.items():
        print(f"download file={filename}", flush=True)
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                revision=REVISION,
                local_dir=output_dir,
            )
        )
        size = path.stat().st_size
        if size != expected_size:
            raise RuntimeError(f"Size mismatch for {path}: expected={expected_size} actual={size}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {path}: expected={expected_hash} actual={digest}")
        receipt["files"][filename] = {
            "path": str(path),
            "size": size,
            "sha256": digest,
            "verified": True,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"verified file={filename} bytes={size}", flush=True)
    receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"complete receipt={receipt_path}", flush=True)


if __name__ == "__main__":
    main()
