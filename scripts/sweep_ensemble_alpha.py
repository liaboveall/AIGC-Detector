"""Sweep fixed ensemble blend weights in one shared two-backbone pass.

Loads checkpoint A and checkpoint B once, runs both over every requested
condition for one manifest, and then blends the *logits* (not the CSV-level
probabilities ``scripts/blend_predictions.py`` needs) at every requested
alpha in plain Python/NumPy. This is the live-model equivalent of the
``base-v2`` ``outputs/base_v2/blend_upper_bound`` diagnostic, generalised so a
whole alpha sweep only costs two forward passes total instead of one pair per
alpha (mirrors ``scripts/evaluate_adapter_gain_candidates.py``'s "one shared
forward pass" design).

For every alpha, writes two files under ``--output-dir``:

- ``alpha_<a>.json``: the same ``{"conditions": ..., "robustness": ...}``
  schema ``evaluate.py`` produces, so it plugs directly into
  ``scripts/compare_robustness_candidates.py`` for the historical gates.
- ``alpha_<a>_predictions.csv``: the same ``path,dataset,source_class,label,
  condition,probability`` schema the rest of the repository uses, so it
  plugs directly into ``scripts/compare_vnext_predictions.py`` for the
  modern generator-macro gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapter import build_checkpoint_model  # noqa: E402
from src.config import dataset_paths, project_path  # noqa: E402
from src.data import RobustnessImageDataset, make_loader  # noqa: E402
from src.metrics import binary_metrics, grouped_metrics, robustness_summary, source_contrast_metrics  # noqa: E402
from src.transforms import EVAL_SUITES, build_eval_transform  # noqa: E402
from src.utils import get_device, set_seed, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", required=True, help="alpha=0 checkpoint (e.g. Tiny vNext)")
    parser.add_argument("--checkpoint-b", required=True, help="alpha=1 checkpoint (e.g. Base v1)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--suite", choices=sorted(EVAL_SUITES), default="full")
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p").replace("-", "neg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_accumulator() -> dict[str, list[Any]]:
    return {"logits": [], "labels": [], "source_classes": [], "datasets": [], "paths": []}


def main() -> None:
    args = parse_args()
    if not args.alphas:
        raise ValueError("at least one alpha is required")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("alphas must be unique")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("alphas must be within [0, 1]")
    tags = [alpha_tag(alpha) for alpha in args.alphas]
    if len(set(tags)) != len(tags):
        raise ValueError("alphas collide after two-decimal output tagging")

    checkpoint_a_path = project_path(args.checkpoint_a)
    checkpoint_b_path = project_path(args.checkpoint_b)
    if not checkpoint_a_path.is_file() or not checkpoint_b_path.is_file():
        raise FileNotFoundError("both ensemble source checkpoints must exist")
    checkpoint_a_sha256 = sha256(checkpoint_a_path)
    checkpoint_b_sha256 = sha256(checkpoint_b_path)
    payload_a = torch.load(checkpoint_a_path, map_location="cpu", weights_only=False)
    payload_b = torch.load(checkpoint_b_path, map_location="cpu", weights_only=False)
    config_a = payload_a["config"]
    config_b = payload_b["config"]
    size_a = int(config_a["data"]["image_size"])
    size_b = int(config_b["data"]["image_size"])
    if size_a != size_b:
        raise ValueError(f"image_size mismatch: model_a={size_a} model_b={size_b}")

    seed = int(config_a.get("seed", 2026))
    set_seed(seed)
    device = get_device(config_a.get("device", "auto"))

    model_a = build_checkpoint_model(config_a, payload_a["model_state"])
    model_b = build_checkpoint_model(config_b, payload_b["model_state"])
    model_a.to(device).eval()
    model_b.to(device).eval()
    if device.type == "cuda":
        model_a.to(memory_format=torch.channels_last)
        model_b.to(memory_format=torch.channels_last)

    dataset_root, manifest_path = dataset_paths(config_a, args.manifest)
    conditions = EVAL_SUITES[args.suite]
    dataset = RobustnessImageDataset(
        dataset_root,
        manifest_path,
        {condition: build_eval_transform(size_a, condition) for condition in conditions},
        max_samples=args.max_samples,
        seed=seed,
    )
    data_config = config_a["data"]
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
    loader = make_loader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=num_workers,
        training=False,
        balanced_sampling=False,
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
    )

    accumulators = {condition: empty_accumulator() for condition in conditions}
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="ensemble-sweep", leave=False)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            labels = batch["label"].float().cpu().tolist()
            paths = list(batch["path"])
            source_classes = list(batch["source_class"])
            datasets = list(batch["dataset"])
            for condition in conditions:
                images = batch["images"][condition].to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits_a = model_a(images).flatten().float()
                    logits_b = model_b(images).flatten().float()
                accumulator = accumulators[condition]
                accumulator["logits"].append(torch.stack([logits_a, logits_b], dim=1).cpu())
                accumulator["labels"].extend(labels)
                accumulator["source_classes"].extend(source_classes)
                accumulator["datasets"].extend(datasets)
                accumulator["paths"].extend(paths)

    stacked_logits = {
        condition: torch.cat(accumulator["logits"], dim=0) if accumulator["logits"] else torch.zeros(0, 2)
        for condition, accumulator in accumulators.items()
    }

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        condition_metrics: dict[str, Any] = {}
        prediction_rows: list[dict[str, Any]] = []
        for condition in conditions:
            accumulator = accumulators[condition]
            logits = stacked_logits[condition]
            blended = (1.0 - alpha) * logits[:, 0] + alpha * logits[:, 1]
            probabilities = torch.sigmoid(blended).tolist()
            labels = accumulator["labels"]
            condition_metrics[condition] = {
                "loss": None,
                "overall": binary_metrics(labels, probabilities),
                "by_source_class": grouped_metrics(labels, probabilities, accumulator["source_classes"]),
                "source_contrasts": source_contrast_metrics(labels, probabilities, accumulator["source_classes"]),
                "by_dataset": grouped_metrics(labels, probabilities, accumulator["datasets"]),
            }
            prediction_rows.extend(
                {
                    "path": path,
                    "dataset": dataset_name,
                    "source_class": source_class,
                    "label": int(label),
                    "condition": condition,
                    "probability": probability,
                }
                for path, dataset_name, source_class, label, probability in zip(
                    accumulator["paths"],
                    accumulator["datasets"],
                    accumulator["source_classes"],
                    labels,
                    probabilities,
                    strict=True,
                )
            )
        json_payload = {
            "checkpoint_a": str(checkpoint_a_path),
            "checkpoint_a_sha256": checkpoint_a_sha256,
            "checkpoint_b": str(checkpoint_b_path),
            "checkpoint_b_sha256": checkpoint_b_sha256,
            "alpha": alpha,
            "manifest": str(manifest_path),
            "suite": args.suite,
            "seed": seed,
            "device": str(device),
            "max_samples": args.max_samples,
            "max_batches": args.max_batches,
            "images": len(accumulators[conditions[0]]["paths"]),
            "prediction_rows": sum(len(accumulator["paths"]) for accumulator in accumulators.values()),
            "conditions": condition_metrics,
            "robustness": robustness_summary(condition_metrics),
        }
        write_json(output_dir / f"{tag}.json", json_payload)
        predictions_path = output_dir / f"{tag}_predictions.csv"
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["path", "dataset", "source_class", "label", "condition", "probability"]
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
        print(
            f"alpha={alpha:.2f} robust_score={json_payload['robustness']['robust_score']:.6f} "
            f"wrote {output_dir / f'{tag}.json'} and {predictions_path}"
        )


if __name__ == "__main__":
    main()
