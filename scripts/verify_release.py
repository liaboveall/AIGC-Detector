"""Verify the frozen release asset and the directory-to-JSON inference contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.config import PROJECT_ROOT, project_path


DEFAULT_CHECKPOINT = "weights/aigc-detector-adapter-v2.pt"
EXPECTED_SHA256 = "C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC"
EXPECTED_TOTAL_PARAMETERS = 28_018_018


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_parameter_count(state: dict[str, torch.Tensor]) -> int:
    # ConvNeXt layer-scale parameters are named ``gamma`` rather than weight/bias.
    # This checkpoint has no non-parameter running-stat buffers, so every state tensor
    # contributes to the frozen architecture's parameter total.
    return sum(tensor.numel() for tensor in state.values())


def main() -> None:
    args = parse_args()
    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Release checkpoint not found: {checkpoint_path}")

    actual_sha256 = sha256_file(checkpoint_path)
    expected_sha256 = args.expected_sha256.strip().upper()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch: expected {expected_sha256}, observed {actual_sha256}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(payload) < {"config", "model_state"}:
        raise RuntimeError("Checkpoint must contain config and model_state")
    if not bool(payload["config"].get("adapter", {}).get("enabled", False)):
        raise RuntimeError("Release checkpoint does not enable the residual adapter")
    parameter_count = checkpoint_parameter_count(payload["model_state"])
    if parameter_count != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError(
            f"Parameter count mismatch: expected {EXPECTED_TOTAL_PARAMETERS:,}, "
            f"observed {parameter_count:,}"
        )
    del payload

    with TemporaryDirectory(prefix="aigc-release-smoke-") as temporary:
        root = Path(temporary)
        input_dir = root / "images"
        input_dir.mkdir()
        rng = np.random.default_rng(2026)
        Image.fromarray(
            rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8), mode="RGB"
        ).save(input_dir / "noise.png")
        Image.fromarray(np.full((96, 96, 3), 128, dtype=np.uint8), mode="RGB").save(
            input_dir / "neutral.jpg", quality=90
        )
        (input_dir / "unreadable.jpg").write_bytes(b"not an image")
        output_path = root / "predictions.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_path),
            "--batch-size",
            "3",
            "--num-workers",
            "0",
            "--device",
            args.device,
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        predictions = json.loads(output_path.read_text(encoding="utf-8"))

    if len(predictions) != 3:
        raise RuntimeError(f"Expected 3 predictions, observed {len(predictions)}")
    if any(set(row) != {"image_path", "pred"} for row in predictions):
        raise RuntimeError("Prediction rows must contain exactly image_path and pred")
    if any(not math.isfinite(float(row["pred"])) for row in predictions):
        raise RuntimeError("All prediction scores must be finite")
    if any(not 0.0 <= float(row["pred"]) <= 1.0 for row in predictions):
        raise RuntimeError("All prediction scores must lie in [0, 1]")
    unreadable = next(row for row in predictions if row["image_path"] == "unreadable.jpg")
    if float(unreadable["pred"]) != 0.5:
        raise RuntimeError("Unreadable-image fallback must be exactly 0.5")

    print(f"release_smoke=PASS checkpoint={checkpoint_path}")
    print(f"sha256={actual_sha256}")
    print(f"parameters={parameter_count:,} predictions={len(predictions)}")


if __name__ == "__main__":
    main()
