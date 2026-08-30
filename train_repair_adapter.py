"""Train a zero-initialised residual branch that repairs a frozen Base model.

Historical GenImage/SID rows match the accepted Tiny teacher and their labels.
Modern/CommunityForensics rows penalise non-zero residuals, preserving the
already strong Base-v1 prediction.  The Base and teacher are always frozen.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from src.adapter import AdapterModel, adapter_parameter_counts, assert_zero_residual_identity
from src.config import dataset_paths, load_config, project_path
from src.data import ManifestImageDataset, RobustnessImageDataset, make_loader
from src.distill import load_teacher
from src.engine import evaluate_condition_suite
from src.metrics import robustness_summary
from src.model import create_model
from src.repair import detached_metrics, repair_loss_components, routing_masks
from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_v2_repair_seed2026.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--check-frozen-grads", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model: AdapterModel,
    teacher: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    repair_config: dict[str, Any],
    gradient_clip: float,
    max_batches: int | None,
    step_log_path: Path,
    check_frozen_grads: bool,
) -> dict[str, Any]:
    model.train()
    teacher.eval()
    repair_sources = set(repair_config["repair_sources"])
    protect_sources = set(repair_config["protect_sources"])
    adapter_parameters = [parameter for parameter in model.adapter.parameters() if parameter.requires_grad]
    if not adapter_parameters:
        raise RuntimeError("repair adapter has no trainable parameters")

    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in ("loss", "bce", "distill", "protect", "teacher_mae")}
    total_examples = repair_examples = protect_examples = 0
    step_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    amp_enabled = device.type == "cuda"
    frozen_grad_checked = False

    progress = tqdm(loader, desc="base-repair", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        repair_mask, protect_mask = routing_masks(
            list(batch["dataset"]), repair_sources, protect_sources, device
        )

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            final_logits, _base_logits, residual_logits = model.forward_with_residual(images)
            if bool(repair_mask.any()):
                teacher_images = images[repair_mask]
                with torch.no_grad():
                    teacher_logits = teacher(teacher_images).flatten()
            else:
                teacher_logits = final_logits.new_empty((0,))
        components = repair_loss_components(
            final_logits,
            residual_logits,
            labels,
            teacher_logits,
            repair_mask,
            protect_mask,
            bce_weight=float(repair_config["bce_weight"]),
            distill_weight=float(repair_config["distill_weight"]),
            protect_weight=float(repair_config["protect_weight"]),
            temperature=float(repair_config.get("temperature", 1.0)),
        )
        scaler.scale(components["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter_parameters, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if check_frozen_grads and not frozen_grad_checked:
            if any(parameter.grad is not None for parameter in model.base.parameters()):
                raise RuntimeError("frozen Base received gradients")
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise RuntimeError("frozen teacher received gradients")
            frozen_grad_checked = True

        batch_size = labels.numel()
        values = detached_metrics(components)
        total_examples += batch_size
        repair_examples += int(repair_mask.sum().item())
        protect_examples += int(protect_mask.sum().item())
        for name in totals:
            totals[name] += values[name] * batch_size
        step_rows.append(
            {
                "batch": batch_index,
                **values,
                "repair_fraction": float(repair_mask.float().mean()),
                "protect_fraction": float(protect_mask.float().mean()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        progress.set_postfix(
            loss=f"{totals['loss'] / total_examples:.4f}",
            mae=f"{totals['teacher_mae'] / total_examples:.3f}",
            repair=f"{repair_examples / total_examples:.2f}",
        )

    if not total_examples:
        raise RuntimeError("training loader produced no examples")
    with step_log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    elapsed = max(time.perf_counter() - started, 1e-9)
    result: dict[str, Any] = {name: value / total_examples for name, value in totals.items()}
    result.update(
        {
            "repair_fraction": repair_examples / total_examples,
            "protect_fraction": protect_examples / total_examples,
            "examples_per_second": total_examples / elapsed,
            "frozen_grad_check": frozen_grad_checked if check_frozen_grads else None,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    data_config = config["data"]
    repair_config = config["repair_training"]
    training_config = config["training"]
    adapter_config = config["adapter"]
    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    max_train_batches = (
        args.max_train_batches
        if args.max_train_batches is not None
        else training_config.get("max_batches_per_epoch")
    )
    max_val_batches = (
        args.max_val_batches
        if args.max_val_batches is not None
        else training_config.get("max_val_batches")
    )
    num_workers = data_config["num_workers"] if args.num_workers is None else args.num_workers
    output_dir = project_path(args.output_dir or config["output"]["directory"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    warmstart_path = project_path(config["warmstart"]["checkpoint"])
    teacher_path = project_path(repair_config["teacher_checkpoint"])
    runtime = {
        "warmstart_checkpoint": str(warmstart_path),
        "teacher_checkpoint": str(teacher_path),
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "num_workers": int(num_workers),
        "output_directory": str(output_dir),
    }
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    resolved["runtime"] = runtime
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
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
            reencode_probability=float(data_config.get("train_reencode_probability", 0.0)),
            reencode_qualities=data_config.get("train_reencode_qualities", (50, 70, 90)),
            reencode_codecs=data_config.get("train_reencode_codecs", ("jpeg", "webp")),
        ),
        training=True,
        max_samples=data_config.get("max_train_samples"),
        seed=seed,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(num_workers),
        training=True,
        balanced_sampling=bool(data_config["balanced_sampling"]),
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
    )
    conditions = list(config["validation"]["conditions"])
    val_dataset = RobustnessImageDataset(
        dataset_root,
        val_manifest,
        {condition: build_eval_transform(image_size, condition) for condition in conditions},
        max_samples=data_config.get("max_val_samples"),
        seed=seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(num_workers),
        training=False,
        balanced_sampling=False,
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=True,
    )

    warmstart = torch.load(warmstart_path, map_location="cpu", weights_only=False)
    warmstart_name = warmstart.get("config", {}).get("model", {}).get("name")
    if warmstart_name and warmstart_name != config["model"]["name"]:
        raise ValueError(f"warmstart model {warmstart_name!r} != {config['model']['name']!r}")
    base = create_model(config["model"], pretrained_override=False)
    base.load_state_dict(warmstart["model_state"])
    del warmstart
    model = AdapterModel(
        base,
        feature_dim=int(adapter_config["feature_dim"]),
        hidden_dim=int(adapter_config["hidden_dim"]),
        residual_gain=float(adapter_config.get("residual_gain", 1.0)),
    )
    teacher = load_teacher(teacher_path, device)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    zero_diff = assert_zero_residual_identity(model, device)
    counts = adapter_parameter_counts(model)
    if counts["trainable"] != counts["adapter_branch"]:
        raise RuntimeError("only the residual repair branch may be trainable")
    print(
        f"params base_frozen={counts['base_frozen']:,} adapter={counts['adapter_branch']:,} "
        f"total={counts['total']:,} zero_diff={zero_diff:.3e}"
    )
    print(f"device={device} train={len(train_dataset):,} val={len(val_dataset):,}")

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(repair_config["learning_rate"]),
        weight_decay=float(repair_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.BCEWithLogitsLoss()
    checkpoint_config = {key: value for key, value in config.items() if not key.startswith("_")}
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            scaler,
            device,
            repair_config,
            float(repair_config.get("gradient_clip", 1.0)),
            max_train_batches,
            output_dir / f"train_steps_epoch_{epoch:02d}.csv",
            args.check_frozen_grads,
        )
        print(f"epoch={epoch} conditions={','.join(conditions)}")
        condition_metrics = evaluate_condition_suite(
            model,
            val_loader,
            criterion,
            device,
            conditions,
            max_batches=max_val_batches,
        )
        robustness = robustness_summary(condition_metrics)
        score = float(robustness["robust_score"])
        epoch_result = {
            "epoch": epoch,
            "runtime": runtime,
            "train": train_metrics,
            "conditions": condition_metrics,
            "robustness": robustness,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "zero_init_identity_max_abs_diff": zero_diff,
        }
        write_json(output_dir / f"metrics_epoch_{epoch:02d}.json", epoch_result)
        append_csv(
            output_dir / "history.csv",
            {
                "epoch": epoch,
                **train_metrics,
                "clean_auc": condition_metrics["clean"]["overall"]["roc_auc"],
                "robust_score": score,
            },
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": checkpoint_config,
            "metrics": epoch_result,
            "parameter_counts": counts,
            "runtime": runtime,
        }
        save_checkpoint(output_dir / f"epoch_{epoch:02d}.pt", checkpoint)
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", checkpoint)
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"teacher_mae={train_metrics['teacher_mae']:.4f} "
            f"robust={score:.6f} best={best_score:.6f}"
        )


if __name__ == "__main__":
    main()
