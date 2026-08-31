"""Evaluate residual gains for one frozen-Base repair checkpoint.

All requested gains share the same frozen-backbone and adapter forward pass.
This makes the diagnostic substantially cheaper than launching one full
evaluation per gain and keeps the validation images perfectly aligned.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapter import (  # noqa: E402
    AdapterModel,
    MultiScaleAdapterModel,
    build_checkpoint_model,
)
from src.config import dataset_paths, project_path  # noqa: E402
from src.data import RobustnessImageDataset, make_loader  # noqa: E402
from src.metrics import (  # noqa: E402
    binary_metrics,
    grouped_metrics,
    robustness_summary,
    source_contrast_metrics,
)
from src.transforms import EVAL_SUITES, build_eval_transform  # noqa: E402
from src.utils import get_device, set_seed, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/base_v2/repair_seed2026/best.pt",
    )
    parser.add_argument(
        "--manifest",
        default="validation_modern_combined_selection_12000.csv",
    )
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/base_v2/repair_seed2026/historical_gain_sweep",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def empty_accumulator() -> dict[str, list[Any]]:
    return {
        "probabilities": [],
        "labels": [],
        "source_classes": [],
        "datasets": [],
    }


def candidate_name(gain: float) -> str:
    text = f"{gain:.2f}".replace("-", "neg").replace(".", "p")
    return f"gain_{text}"


def main() -> None:
    args = parse_args()
    if not args.gains:
        raise ValueError("at least one residual gain is required")
    if len(set(args.gains)) != len(args.gains):
        raise ValueError("residual gains must be unique")
    if any(not math.isfinite(gain) or gain < 0.0 or gain > 8.0 for gain in args.gains):
        raise ValueError("residual gains must be finite and within [0, 8]")

    checkpoint_path = project_path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    model = build_checkpoint_model(config, payload["model_state"])
    if not isinstance(model, (AdapterModel, MultiScaleAdapterModel)):
        raise TypeError("repair checkpoint did not build a supported adapter model")
    model.to(device).eval()
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    data_config = config["data"]
    dataset_root, manifest_path = dataset_paths(config, args.manifest)
    conditions = EVAL_SUITES["full"]
    dataset = RobustnessImageDataset(
        dataset_root,
        manifest_path,
        {
            condition: build_eval_transform(int(data_config["image_size"]), condition)
            for condition in conditions
        },
        max_samples=args.max_samples,
        seed=seed,
    )
    num_workers = int(
        data_config["num_workers"] if args.num_workers is None else args.num_workers
    )
    loader = make_loader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=num_workers,
        training=False,
        balanced_sampling=False,
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=True,
    )

    names = {candidate_name(gain): gain for gain in args.gains}
    accumulators = {
        name: {condition: empty_accumulator() for condition in conditions}
        for name in names
    }
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="repair-gain-scan", leave=False):
            labels = batch["label"].float().cpu().tolist()
            source_classes = list(batch["source_class"])
            datasets = list(batch["dataset"])
            for condition in conditions:
                images = batch["images"][condition].to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    _final_logits, base_logits, residual = model.forward_with_residual(images)
                    base_logits = base_logits.flatten()
                    residual = residual.flatten()
                    probabilities = {
                        name: torch.sigmoid(base_logits + gain * residual)
                        .float()
                        .cpu()
                        .tolist()
                        for name, gain in names.items()
                    }
                for name, values in probabilities.items():
                    accumulator = accumulators[name][condition]
                    accumulator["probabilities"].extend(values)
                    accumulator["labels"].extend(labels)
                    accumulator["source_classes"].extend(source_classes)
                    accumulator["datasets"].extend(datasets)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "candidates": {},
    }
    for name, gain in names.items():
        condition_metrics = {}
        for condition, accumulator in accumulators[name].items():
            labels = accumulator["labels"]
            probabilities = accumulator["probabilities"]
            condition_metrics[condition] = {
                "loss": math.nan,
                "overall": binary_metrics(labels, probabilities),
                "by_source_class": grouped_metrics(
                    labels, probabilities, accumulator["source_classes"]
                ),
                "source_contrasts": source_contrast_metrics(
                    labels, probabilities, accumulator["source_classes"]
                ),
                "by_dataset": grouped_metrics(
                    labels, probabilities, accumulator["datasets"]
                ),
            }
        result = {
            "candidate": name,
            "gain": gain,
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "conditions": condition_metrics,
            "robustness": robustness_summary(condition_metrics),
        }
        output_path = output_dir / f"{name}.json"
        write_json(output_path, result)
        summary["candidates"][name] = {
            "gain": gain,
            "robust_score": result["robustness"]["robust_score"],
            "path": str(output_path),
        }
        print(
            f"wrote candidate={name} "
            f"robust={result['robustness']['robust_score']:.6f}"
        )
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
