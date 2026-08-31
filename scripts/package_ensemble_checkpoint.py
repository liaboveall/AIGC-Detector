"""Package a fixed-weight logit ensemble of two existing checkpoints.

Bundles checkpoint A (``alpha=0``) and checkpoint B (``alpha=1``) plus a fixed
blend weight into one self-contained ``.pt`` file that ``predict.py`` and
``evaluate.py`` can load exactly like any other checkpoint -- no code changes
or extra CLI flags are needed at inference time, because
``src.adapter.build_checkpoint_model`` recognises the ``ensemble.enabled``
config key and builds an ``EnsembleModel`` (see ``src/ensemble.py``).

Example (Tiny vNext + Base v1, the pair validated in
``docs/ENSEMBLE_VNEXT.md``)::

    python scripts/package_ensemble_checkpoint.py \
      --checkpoint-a outputs/tiny_vnext/final_candidate/tiny_vnext_seed2026_gain1p60.pt \
      --checkpoint-b outputs/base_v1/primary/best.pt \
      --alpha 0.50 \
      --output weights/aigc-detector-ensemble-vnext.pt
"""
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


def strip_private_keys(config: dict) -> dict:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def state_parameter_count(state: dict) -> int:
    return sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", required=True, help="alpha=0 checkpoint (e.g. Tiny vNext)")
    parser.add_argument("--checkpoint-b", required=True, help="alpha=1 checkpoint (e.g. Base v1)")
    parser.add_argument("--alpha", required=True, type=float, help="weight on checkpoint-b, within [0, 1]")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be within [0, 1]")

    source_a = Path(args.checkpoint_a).resolve()
    source_b = Path(args.checkpoint_b).resolve()
    destination = Path(args.output).resolve()
    if not source_a.is_file() or not source_b.is_file():
        raise FileNotFoundError("both ensemble source checkpoints must exist")
    if destination in (source_a, source_b):
        raise ValueError("Refusing to overwrite a source checkpoint")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    checkpoint_a = torch.load(source_a, map_location="cpu", weights_only=False)
    checkpoint_b = torch.load(source_b, map_location="cpu", weights_only=False)
    for name, checkpoint in (("model_a", checkpoint_a), ("model_b", checkpoint_b)):
        if set(checkpoint) < {"config", "model_state"}:
            raise ValueError(f"{name} checkpoint must contain config and model_state")
    config_a = strip_private_keys(copy.deepcopy(checkpoint_a["config"]))
    config_b = strip_private_keys(copy.deepcopy(checkpoint_b["config"]))
    size_a = int(config_a["data"]["image_size"])
    size_b = int(config_b["data"]["image_size"])
    if size_a != size_b:
        raise ValueError(f"image_size mismatch between sub-checkpoints: model_a={size_a} model_b={size_b}")
    if config_a.get("ensemble", {}).get("enabled") or config_b.get("ensemble", {}).get("enabled"):
        raise ValueError("Nesting an ensemble checkpoint inside another ensemble is not supported")

    sha_a = sha256(source_a)
    sha_b = sha256(source_b)
    parameters_a = state_parameter_count(checkpoint_a["model_state"])
    parameters_b = state_parameter_count(checkpoint_b["model_state"])
    packaged_config = {
        "seed": config_a.get("seed", 2026),
        "device": config_a.get("device", "auto"),
        "data": config_a["data"],
        "ensemble": {
            "enabled": True,
            "alpha": float(args.alpha),
            "model_a": {
                "config": config_a,
                "source_checkpoint": str(source_a),
                "source_sha256": sha_a,
            },
            "model_b": {
                "config": config_b,
                "source_checkpoint": str(source_b),
                "source_sha256": sha_b,
            },
        },
    }
    packaged = {
        "config": packaged_config,
        "model_state": {
            "model_a": checkpoint_a["model_state"],
            "model_b": checkpoint_b["model_state"],
        },
        "delivery": {
            "kind": "fixed_weight_logit_ensemble",
            "alpha": float(args.alpha),
            "model_a_source": str(source_a),
            "model_a_sha256": sha_a,
            "model_a_parameters": parameters_a,
            "model_b_source": str(source_b),
            "model_b_sha256": sha_b,
            "model_b_parameters": parameters_b,
            "total_parameters": parameters_a + parameters_b,
        },
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
