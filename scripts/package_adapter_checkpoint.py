"""Create an immutable adapter checkpoint with a fixed inference residual gain."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package an adapter checkpoint with a fixed inference gain."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--residual-gain", required=True, type=float)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    if source == destination:
        raise ValueError("Refusing to overwrite the source checkpoint")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if not 0.0 <= args.residual_gain <= 2.0:
        raise ValueError("residual gain must be in [0, 2]")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    packaged = copy.deepcopy(checkpoint)
    adapter = packaged.get("config", {}).get("adapter", {})
    if not adapter.get("enabled", False):
        raise ValueError("Input checkpoint is not adapter-enabled")
    adapter["residual_gain"] = float(args.residual_gain)
    packaged["delivery"] = {
        "source_checkpoint": str(source),
        "source_sha256": sha256(source),
        "fixed_residual_gain": float(args.residual_gain),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packaged, destination)
    metadata = {
        "checkpoint": str(destination),
        "sha256": sha256(destination),
        **packaged["delivery"],
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
