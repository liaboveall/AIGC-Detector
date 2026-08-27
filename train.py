from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.config import dataset_paths, load_config, project_path
from src.data import ManifestImageDataset, RobustnessImageDataset, make_loader
from src.engine import evaluate_condition_suite, train_one_epoch
from src.metrics import robustness_summary
from src.model import create_model, parameter_counts
from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the robust AI-image baseline")
    parser.add_argument("--config", default="configs/baseline_smoke.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        help="Initialise model weights from a checkpoint without restoring optimizer state",
    )
    return parser.parse_args()


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * warmup_epochs, 0)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    data_config = config["data"]
    training_config = config["training"]
    epochs = args.epochs or int(training_config["epochs"])
    max_train_batches = args.max_train_batches or training_config.get("max_batches_per_epoch")
    max_val_batches = args.max_val_batches or training_config.get("max_val_batches")
    output_dir = project_path(args.output_dir or config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint_path = project_path(args.init_checkpoint) if args.init_checkpoint else None
    runtime = {
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "num_workers": int(data_config["num_workers"] if args.num_workers is None else args.num_workers),
        "pretrained_override": False if args.no_pretrained or init_checkpoint_path else None,
        "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path else None,
        "output_directory": str(output_dir),
    }
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        resolved = {k: v for k, v in config.items() if not k.startswith("_")}
        resolved["runtime"] = runtime
        yaml.safe_dump(resolved, handle, sort_keys=False)

    dataset_root, train_manifest = dataset_paths(config, data_config["train_manifest"])
    _, val_manifest = dataset_paths(config, data_config["val_manifest"])
    image_size = int(data_config["image_size"])
    train_dataset = ManifestImageDataset(
        dataset_root,
        train_manifest,
        build_train_transform(
            image_size,
            float(data_config["train_degradation_probability"]),
            degradation_kind_weights=data_config.get("train_degradation_kind_weights"),
            blur_weights=data_config.get("train_blur_weights"),
        ),
        training=True,
        max_samples=data_config.get("max_train_samples"),
        seed=seed,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=num_workers,
        training=True,
        balanced_sampling=bool(data_config["balanced_sampling"]),
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
    )
    conditions = list(config["validation"]["conditions"])
    validation_dataset = RobustnessImageDataset(
        dataset_root,
        val_manifest,
        {condition: build_eval_transform(image_size, condition) for condition in conditions},
        max_samples=data_config.get("max_val_samples"),
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=num_workers,
        training=False,
        balanced_sampling=False,
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=True,
    )

    pretrained_override = False if args.no_pretrained or init_checkpoint_path else None
    model = create_model(config["model"], pretrained_override=pretrained_override).to(device)
    initial_checkpoint = None
    initial_score = -math.inf
    if init_checkpoint_path is not None:
        initial_checkpoint = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
        source_model_name = initial_checkpoint.get("config", {}).get("model", {}).get("name")
        target_model_name = config["model"].get("name")
        if source_model_name and source_model_name != target_model_name:
            raise ValueError(
                f"Checkpoint model {source_model_name!r} does not match config model {target_model_name!r}"
            )
        model.load_state_dict(initial_checkpoint["model_state"])
        source_robustness = initial_checkpoint.get("metrics", {}).get("robustness", {})
        source_score = source_robustness.get("robust_score")
        if source_score is not None and math.isfinite(float(source_score)):
            initial_score = float(source_score)
        print(f"initialized_from={init_checkpoint_path} robust_score={initial_score:.4f}")
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    counts = parameter_counts(model)
    print(f"device={device} train={len(train_dataset)} val={len(validation_dataset)}")
    print(f"model={config['model']['name']} parameters={counts['total']:,} trainable={counts['trainable']:,}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation_steps = int(training_config.get("accumulation_steps", 1))
    batch_count = min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader)
    steps_per_epoch = math.ceil(batch_count / accumulation_steps)
    scheduler = cosine_schedule(
        optimizer,
        steps_per_epoch,
        epochs,
        int(training_config.get("warmup_epochs", 0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    writer = SummaryWriter(output_dir / "tensorboard") if config["output"].get("tensorboard", True) else None
    best_score = initial_score
    if initial_checkpoint is not None:
        initial_best = {
            "epoch": 0,
            "model_state": model.state_dict(),
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "metrics": initial_checkpoint.get("metrics", {}),
            "parameter_counts": counts,
            "runtime": runtime,
            "initialized_from": str(init_checkpoint_path),
        }
        save_checkpoint(output_dir / "best.pt", initial_best)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            scheduler,
            accumulation_steps,
            float(training_config.get("gradient_clip", 1.0)),
            max_batches=max_train_batches,
        )
        print(f"epoch={epoch} conditions={','.join(conditions)}")
        condition_metrics = evaluate_condition_suite(
            model,
            validation_loader,
            criterion,
            device,
            conditions,
            max_batches=max_val_batches,
        )
        robustness = robustness_summary(condition_metrics)
        clean_auc = float(condition_metrics["clean"]["overall"]["roc_auc"])
        score = float(robustness["robust_score"])
        epoch_result = {
            "epoch": epoch,
            "runtime": runtime,
            "train": train_metrics,
            "conditions": condition_metrics,
            "robustness": robustness,
        }
        write_json(output_dir / f"metrics_epoch_{epoch:02d}.json", epoch_result)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "clean_auc": clean_auc,
            "mean_degraded_auc": robustness["mean_degraded_auc"],
            "worst_degraded_auc": robustness["worst_degraded_auc"],
            "robust_score": score,
        }
        append_history(output_dir / "history.csv", history_row)
        if writer is not None:
            for key, value in history_row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "metrics": epoch_result,
            "parameter_counts": counts,
            "runtime": runtime,
        }
        save_checkpoint(output_dir / "last.pt", checkpoint)
        if math.isfinite(score) and score > best_score:
            best_score = score
            # The best artifact is inference/evaluation ready and intentionally
            # omits AdamW state, which is roughly twice the model size. `last.pt`
            # retains the full optimizer state for future resume support.
            best_checkpoint = {
                key: value
                for key, value in checkpoint.items()
                if key not in {"optimizer_state", "scheduler_state"}
            }
            save_checkpoint(output_dir / "best.pt", best_checkpoint)
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} clean_auc={clean_auc:.4f} "
            f"robust_score={score:.4f} best={best_score:.4f}"
        )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
