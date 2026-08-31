"""Verify the frozen ensemble asset and directory-to-JSON inference contract."""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapter import build_checkpoint_model
from src.ensemble import EnsembleModel, ensemble_parameter_counts


DEFAULT_CHECKPOINT = "weights/aigc-detector-ensemble-vnext.pt"
EXPECTED_ALPHA = 0.50
EXPECTED_MODEL_A_SHA256 = "1AF51D00022B9CD3FABD58D65F01C7F728F6F99C2649AAA86ACDBAA9789EDE44"
EXPECTED_MODEL_B_SHA256 = "F49D423847B26F26FAF4C2558F1A831658F0F92DF22F37F56B3E33BA51264DD5"
EXPECTED_TOTAL_PARAMETERS = 115_585_507


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def manifest_sha256(checkpoint_path: Path) -> str:
    manifest = PROJECT_ROOT / "weights" / "SHA256SUMS.txt"
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, filename = line.split(maxsplit=1)
        if filename.strip().lstrip("*") == checkpoint_path.name:
            return digest.upper()
    raise RuntimeError(f"{checkpoint_path.name} is absent from {manifest}")


def run_predict(checkpoint: Path, input_dir: Path, output: Path, device: str) -> list[dict]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "predict.py"),
        "--checkpoint",
        str(checkpoint),
        "--input-dir",
        str(input_dir),
        "--output",
        str(output),
        "--batch-size",
        "3",
        "--num-workers",
        "0",
        "--device",
        device,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ensemble checkpoint not found: {checkpoint_path}")

    actual_sha256 = sha256_file(checkpoint_path)
    expected_sha256 = (args.expected_sha256 or manifest_sha256(checkpoint_path)).strip().upper()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch: expected {expected_sha256}, observed {actual_sha256}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(payload) < {"config", "model_state", "delivery"}:
        raise RuntimeError("Ensemble checkpoint must contain config, model_state, and delivery")
    ensemble_config = payload["config"].get("ensemble", {})
    if not ensemble_config.get("enabled", False):
        raise RuntimeError("Release checkpoint does not enable the ensemble")
    alpha = float(ensemble_config.get("alpha", -1.0))
    if not math.isclose(alpha, EXPECTED_ALPHA, abs_tol=1e-12):
        raise RuntimeError(f"Alpha mismatch: expected {EXPECTED_ALPHA}, observed {alpha}")
    source_a_sha256 = str(ensemble_config["model_a"].get("source_sha256", "")).upper()
    source_b_sha256 = str(ensemble_config["model_b"].get("source_sha256", "")).upper()
    if source_a_sha256 != EXPECTED_MODEL_A_SHA256 or source_b_sha256 != EXPECTED_MODEL_B_SHA256:
        raise RuntimeError("Embedded source checkpoint hashes do not match the frozen protocol")

    model = build_checkpoint_model(payload["config"], payload["model_state"])
    if not isinstance(model, EnsembleModel):
        raise RuntimeError("Checkpoint did not construct an EnsembleModel")
    counts = ensemble_parameter_counts(model)
    if counts["total"] != EXPECTED_TOTAL_PARAMETERS or counts["trainable"] != 0:
        raise RuntimeError(f"Unexpected ensemble parameter counts: {counts}")
    del model, payload

    with TemporaryDirectory(prefix="aigc-ensemble-release-") as temporary:
        root = Path(temporary)
        input_dir = root / "images"
        input_dir.mkdir()
        rng = np.random.default_rng(2026)
        Image.fromarray(rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8), mode="RGB").save(
            input_dir / "noise.png"
        )
        Image.fromarray(np.full((96, 96, 3), 128, dtype=np.uint8), mode="RGB").save(
            input_dir / "neutral.jpg", quality=90
        )
        (input_dir / "unreadable.jpg").write_bytes(b"not an image")
        first = run_predict(checkpoint_path, input_dir, root / "predictions_first.json", args.device)
        second = run_predict(checkpoint_path, input_dir, root / "predictions_second.json", args.device)

    if first != second:
        raise RuntimeError("Repeated directory inference was not deterministic")
    if len(first) != 3 or any(set(row) != {"image_path", "pred"} for row in first):
        raise RuntimeError("Prediction output does not match the exact image_path,pred contract")
    if any(not math.isfinite(float(row["pred"])) or not 0.0 <= float(row["pred"]) <= 1.0 for row in first):
        raise RuntimeError("All prediction scores must be finite and within [0, 1]")
    unreadable = next(row for row in first if row["image_path"] == "unreadable.jpg")
    if float(unreadable["pred"]) != 0.5:
        raise RuntimeError("Unreadable-image fallback must be exactly 0.5")

    result = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "sha256": actual_sha256,
        "device": args.device,
        "alpha": alpha,
        "source_sha256": {"model_a": source_a_sha256, "model_b": source_b_sha256},
        "parameter_count": counts,
        "predictions": len(first),
        "deterministic_repeated_inference": True,
        "unreadable_fallback": 0.5,
    }
    if args.output:
        output = resolve(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
