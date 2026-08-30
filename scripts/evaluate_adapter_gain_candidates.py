"""Evaluate adapter gain candidates in one shared frozen-backbone pass."""
from __future__ import annotations

import argparse
import csv
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

from src.adapter import AdapterModel, ResidualAdapter, build_checkpoint_model  # noqa: E402
from src.config import dataset_paths, project_path  # noqa: E402
from src.data import RobustnessImageDataset, make_loader  # noqa: E402
from src.metrics import binary_metrics, grouped_metrics, robustness_summary, source_contrast_metrics  # noqa: E402
from src.transforms import EVAL_SUITES, build_eval_transform  # noqa: E402
from src.utils import get_device, set_seed, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", default="outputs/adapter_v1/best.pt")
    parser.add_argument("--v2", default="weights/aigc-detector-adapter-v2.pt")
    parser.add_argument("--manifest", default="tiny_vnext_modern_dev.csv")
    parser.add_argument("--v1-gain", type=float, default=0.60)
    parser.add_argument("--v2-alt-gain", type=float, default=0.87)
    parser.add_argument(
        "--candidate",
        help="Optional trained adapter checkpoint to sweep against frozen Adapter v2",
    )
    parser.add_argument(
        "--candidate-gains",
        type=float,
        nargs="+",
        default=[0.60, 0.80, 1.00, 1.20],
    )
    parser.add_argument("--output-dir", default="outputs/tiny_vnext/gain_scan")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def assert_same_base(v1_state: dict[str, torch.Tensor], v2_state: dict[str, torch.Tensor]) -> None:
    v1 = {key[5:]: value for key, value in v1_state.items() if key.startswith("base.")}
    v2 = {key[5:]: value for key, value in v2_state.items() if key.startswith("base.")}
    if set(v1) != set(v2):
        raise ValueError("Adapter checkpoints do not contain the same frozen-base tensors")
    for key in sorted(v1):
        if not torch.equal(v1[key], v2[key]):
            raise ValueError(f"Frozen base differs between adapter checkpoints: {key}")


def adapter_from_checkpoint(payload: dict[str, Any]) -> ResidualAdapter:
    config = payload["config"]["adapter"]
    adapter = ResidualAdapter(int(config["feature_dim"]), int(config["hidden_dim"]))
    state = {
        key.removeprefix("adapter."): value
        for key, value in payload["model_state"].items()
        if key.startswith("adapter.")
    }
    adapter.load_state_dict(state)
    return adapter


def empty_accumulator() -> dict[str, Any]:
    return {
        "probabilities": [],
        "labels": [],
        "source_classes": [],
        "datasets": [],
    }


def main() -> None:
    args = parse_args()
    requested_gains = args.candidate_gains if args.candidate else [args.v1_gain, args.v2_alt_gain]
    if any(not 0.0 <= gain <= 2.0 for gain in requested_gains):
        raise ValueError("Gain values must be within [0, 2]")
    v1_path = project_path(args.candidate or args.v1)
    v2_path = project_path(args.v2)
    v1_payload = torch.load(v1_path, map_location="cpu", weights_only=False)
    v2_payload = torch.load(v2_path, map_location="cpu", weights_only=False)
    assert_same_base(v1_payload["model_state"], v2_payload["model_state"])
    config = v2_payload["config"]
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))

    model = build_checkpoint_model(config, v2_payload["model_state"])
    if not isinstance(model, AdapterModel):
        raise TypeError("v2 checkpoint did not build an AdapterModel")
    v1_adapter = adapter_from_checkpoint(v1_payload)
    model.to(device).eval()
    v1_adapter.to(device).eval()
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    data_config = config["data"]
    _, manifest_path = dataset_paths(config, args.manifest)
    conditions = EVAL_SUITES["full"]
    dataset = RobustnessImageDataset(
        project_path(data_config["dataset_root"]),
        manifest_path,
        {condition: build_eval_transform(int(data_config["image_size"]), condition) for condition in conditions},
        max_samples=args.max_samples,
        seed=seed,
    )
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
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
    if args.candidate:
        candidates = {"release_v2_gain_1p00": ("v2", 1.0)}
        candidates.update(
            {
                f"candidate_gain_{gain:.2f}".replace(".", "p"): ("v1", gain)
                for gain in args.candidate_gains
            }
        )
    else:
        candidates = {
            "v2_gain_1p00": ("v2", 1.0),
            f"v1_gain_{args.v1_gain:.2f}".replace(".", "p"): ("v1", args.v1_gain),
            f"v2_gain_{args.v2_alt_gain:.2f}".replace(".", "p"): ("v2", args.v2_alt_gain),
        }
    accumulators = {
        name: {condition: empty_accumulator() for condition in conditions}
        for name in candidates
    }
    prediction_rows = {name: [] for name in candidates}
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="gain-scan", leave=False):
            labels = batch["label"].float().cpu().tolist()
            paths = list(batch["path"])
            source_classes = list(batch["source_class"])
            datasets = list(batch["dataset"])
            for condition in conditions:
                images = batch["images"][condition].to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    features = model.pooled_features(images)
                    head = model.base.head
                    base_logits = head.fc(head.drop(head.pre_logits(features))).flatten()
                    residuals = {
                        "v1": v1_adapter(features).flatten(),
                        "v2": model.adapter(features).flatten(),
                    }
                    probabilities = {
                        name: torch.sigmoid(base_logits + gain * residuals[version]).float().cpu().tolist()
                        for name, (version, gain) in candidates.items()
                    }
                for name, values in probabilities.items():
                    accumulator = accumulators[name][condition]
                    accumulator["probabilities"].extend(values)
                    accumulator["labels"].extend(labels)
                    accumulator["source_classes"].extend(source_classes)
                    accumulator["datasets"].extend(datasets)
                    prediction_rows[name].extend(
                        {
                            "path": path,
                            "dataset": dataset_name,
                            "source_class": source_class,
                            "label": int(label),
                            "condition": condition,
                            "probability": probability,
                        }
                        for path, dataset_name, source_class, label, probability in zip(
                            paths, datasets, source_classes, labels, values, strict=True
                        )
                    )

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, condition_accumulators in accumulators.items():
        condition_metrics = {}
        for condition, accumulator in condition_accumulators.items():
            labels = accumulator["labels"]
            probabilities = accumulator["probabilities"]
            condition_metrics[condition] = {
                "loss": math.nan,
                "overall": binary_metrics(labels, probabilities),
                "by_source_class": grouped_metrics(labels, probabilities, accumulator["source_classes"]),
                "source_contrasts": source_contrast_metrics(
                    labels, probabilities, accumulator["source_classes"]
                ),
                "by_dataset": grouped_metrics(labels, probabilities, accumulator["datasets"]),
            }
        payload = {
            "candidate": name,
            "v1_checkpoint": str(v1_path),
            "v2_checkpoint": str(v2_path),
            "manifest": str(manifest_path),
            "conditions": condition_metrics,
            "robustness": robustness_summary(condition_metrics),
        }
        write_json(output_dir / f"{name}.json", payload)
        with (output_dir / f"{name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "dataset", "source_class", "label", "condition", "probability"],
            )
            writer.writeheader()
            writer.writerows(prediction_rows[name])
        print(f"wrote candidate={name} robust={payload['robustness']['robust_score']:.6f}")


if __name__ == "__main__":
    main()
