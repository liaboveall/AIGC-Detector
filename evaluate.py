from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.adapter import build_checkpoint_model
from src.config import dataset_paths, project_path
from src.data import RobustnessImageDataset, make_loader
from src.engine import evaluate_condition_suite
from src.metrics import robustness_summary
from src.transforms import EVAL_SUITES, build_eval_transform
from src.utils import get_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint under deterministic degradations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--suite", choices=sorted(EVAL_SUITES), default="full")
    parser.add_argument(
        "--conditions",
        help="Comma-separated conditions; overrides --suite (for example clean,blur_2.0)",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output")
    parser.add_argument("--predictions-output")
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    # The shared loader dispatches legacy, adapter, and fixed-ensemble schemas.
    model = build_checkpoint_model(config, checkpoint["model_state"])
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    criterion = torch.nn.BCEWithLogitsLoss()
    data_config = config["data"]
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
    manifest_name = args.manifest or data_config["val_manifest"]
    dataset_root, manifest_path = dataset_paths(config, manifest_name)
    if args.conditions:
        conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
        if not conditions:
            raise ValueError("--conditions did not contain any condition names")
        if len(set(conditions)) != len(conditions):
            raise ValueError("--conditions contains duplicate condition names")
        suite_name = "custom"
    else:
        conditions = EVAL_SUITES[args.suite]
        suite_name = args.suite
    print(f"conditions={','.join(conditions)}")
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
    loader = make_loader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=num_workers,
        training=False,
        balanced_sampling=False,
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
    )
    prediction_rows: list[dict] | None = [] if args.predictions_output else None
    results = evaluate_condition_suite(
        model,
        loader,
        criterion,
        device,
        conditions,
        args.max_batches,
        prediction_rows=prediction_rows,
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "suite": suite_name,
        "conditions": results,
        "robustness": robustness_summary(results),
    }
    output_path = (
        project_path(args.output)
        if args.output
        else checkpoint_path.parent / f"evaluation_{suite_name}.json"
    )
    write_json(output_path, payload)
    print(f"wrote {output_path}")
    if args.predictions_output:
        predictions_path = project_path(args.predictions_output)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["path", "dataset", "source_class", "label", "condition", "probability"]
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows or [])
        print(f"wrote {predictions_path}")


if __name__ == "__main__":
    main()
