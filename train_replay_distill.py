"""Replay + distillation training entry point (task #5).

Student either hot-starts from a same-architecture checkpoint or starts from
the pretrained backbone named by the config, with stem/stages.0/1 frozen. A
frozen teacher checkpoint provides temperature-scaled binary KD on selected
sources; teacher and student architectures may differ.
All experiment artifacts go to outputs/replay_distill_v1/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.config import dataset_paths, load_config, project_path
from src.data import ManifestImageDataset, RobustnessImageDataset, make_loader
from src.distill import (
    build_param_groups,
    freeze_prefixes,
    group_parameter_counts,
    load_teacher,
    per_source_robust_scores,
    train_one_epoch_distill,
)
from src.engine import evaluate_condition_suite
from src.metrics import robustness_summary
from src.model import create_model
from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay + distillation training (task #5)")
    parser.add_argument("--config", default="configs/replay_distill_v1.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--check-teacher-grad",
        action="store_true",
        help="Smoke aid: verify all teacher gradients stay None after the first optimizer step",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable the per-epoch old-domain drop guard (smoke runs only)",
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


def load_reference_scores(early_stop_config: dict[str, Any]) -> dict[str, float]:
    reference_path = project_path(early_stop_config["reference_metrics"])
    with reference_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return per_source_robust_scores(payload["conditions"], list(early_stop_config["sources"]))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    data_config = config["data"]
    training_config = config["training"]
    distill_config = config["distillation"]
    warmstart_config = config["warmstart"]
    early_stop_config = config.get("early_stop", {})
    epochs = args.epochs or int(training_config["epochs"])
    max_train_batches = args.max_train_batches or training_config.get("max_batches_per_epoch")
    max_val_batches = args.max_val_batches or training_config.get("max_val_batches")
    output_dir = project_path(args.output_dir or config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    warmstart_value = warmstart_config.get("checkpoint")
    warmstart_path = project_path(warmstart_value) if warmstart_value else None
    teacher_path = project_path(distill_config["teacher_checkpoint"])
    early_stop_enabled = bool(early_stop_config.get("enabled", False)) and not args.no_early_stop
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
    runtime = {
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "num_workers": num_workers,
        "warmstart_checkpoint": str(warmstart_path) if warmstart_path else None,
        "teacher_checkpoint": str(teacher_path),
        "early_stop_enabled": early_stop_enabled,
        "output_directory": str(output_dir),
    }
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

    # Student: either hot-start from a same-architecture checkpoint or use the
    # pretrained backbone requested by the config, then freeze shallow layers.
    if warmstart_path is not None:
        student = create_model(config["model"], pretrained_override=False)
        warmstart = torch.load(warmstart_path, map_location="cpu", weights_only=False)
        warmstart_model_name = warmstart.get("config", {}).get("model", {}).get("name")
        if warmstart_model_name and warmstart_model_name != config["model"].get("name"):
            raise ValueError(
                f"Warmstart checkpoint model {warmstart_model_name!r} does not match "
                f"config model {config['model'].get('name')!r}"
            )
        student.load_state_dict(warmstart["model_state"])
        del warmstart
    else:
        if not bool(config["model"].get("pretrained", False)):
            raise ValueError("No warmstart checkpoint requires model.pretrained: true")
        student = create_model(config["model"])
    frozen_prefixes = list(distill_config.get("frozen_prefixes", ("stem.", "stages.0.", "stages.1.")))
    frozen_count = freeze_prefixes(student, frozen_prefixes)
    counts = group_parameter_counts(student)
    if counts["frozen"] != frozen_count:
        raise RuntimeError("Freeze bookkeeping mismatch")
    student.to(device)
    if device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)

    # Teacher: architecture comes from its own checkpoint and may differ from
    # the student (for example Tiny teacher -> Base student).
    teacher = load_teacher(teacher_path, device)

    print(
        f"student params: total={counts['total']:,} frozen={counts['frozen']:,} "
        f"(stem+stages.0+stages.1) head={counts['head']:,} stages_2_3={counts['stages_2_3']:,} "
        f"trainable={counts['trainable']:,}"
    )
    teacher_total = sum(p.numel() for p in teacher.parameters())
    teacher_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"teacher params: total={teacher_total:,} trainable={teacher_trainable:,} (must be 0)")
    print(f"device={device} train={len(train_dataset)} val={len(validation_dataset)}")
    print(f"frozen_prefixes={frozen_prefixes}")

    parameter_groups = build_param_groups(
        student,
        head_learning_rate=float(distill_config["head_learning_rate"]),
        stages_learning_rate=float(distill_config["stages_learning_rate"]),
    )
    for group in parameter_groups:
        group_size = sum(p.numel() for p in group["params"])
        print(f"param_group={group['name']} lr={group['lr']:.2e} params={group_size:,}")
    optimizer = torch.optim.AdamW(
        parameter_groups,
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
    criterion = torch.nn.BCEWithLogitsLoss()
    temperature = float(distill_config["temperature"])
    kd_weight = float(distill_config["kd_weight"])
    kd_sources = list(distill_config["kd_sources"])
    reference_scores = load_reference_scores(early_stop_config) if early_stop_enabled else None
    if reference_scores is not None:
        formatted = ", ".join(f"{source}={score:.6f}" for source, score in reference_scores.items())
        print(f"early_stop_reference: {formatted} max_drop={early_stop_config['max_source_drop']}")

    best_score = -math.inf
    stopped_early = False
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch_distill(
            student,
            teacher,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            scheduler,
            accumulation_steps,
            float(training_config.get("gradient_clip", 1.0)),
            temperature,
            kd_weight,
            kd_sources,
            max_batches=max_train_batches,
            step_log_path=output_dir / f"train_steps_epoch_{epoch:02d}.csv",
            check_teacher_grad=args.check_teacher_grad,
        )
        peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        if args.check_teacher_grad:
            print(f"epoch={epoch} teacher_grad_all_none={train_metrics.get('teacher_grad_all_none')}")
        print(f"epoch={epoch} conditions={','.join(conditions)}")
        condition_metrics = evaluate_condition_suite(
            student,
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
            "peak_gpu_memory_bytes": peak_memory_bytes,
        }
        write_json(output_dir / f"metrics_epoch_{epoch:02d}.json", epoch_result)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "bce": train_metrics["bce"],
            "kd": train_metrics["kd"],
            "kd_fraction": train_metrics["kd_fraction"],
            "examples_per_second": train_metrics["examples_per_second"],
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
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "metrics": epoch_result,
            "parameter_counts": counts,
            "runtime": runtime,
        }
        save_checkpoint(output_dir / "last.pt", checkpoint)
        if config["output"].get("save_epoch_checkpoints", False):
            epoch_checkpoint = {
                key: value
                for key, value in checkpoint.items()
                if key not in {"optimizer_state", "scheduler_state"}
            }
            save_checkpoint(output_dir / f"epoch_{epoch:02d}.pt", epoch_checkpoint)
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_checkpoint = {
                key: value
                for key, value in checkpoint.items()
                if key not in {"optimizer_state", "scheduler_state"}
            }
            save_checkpoint(output_dir / "best.pt", best_checkpoint)
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} bce={train_metrics['bce']:.4f} "
            f"kd={train_metrics['kd']:.4f} kd_fraction={train_metrics['kd_fraction']:.3f} "
            f"ex/s={train_metrics['examples_per_second']:.1f} "
            f"clean_auc={clean_auc:.4f} robust_score={score:.4f} best={best_score:.4f}"
        )
        # Per-epoch early stop: abort before the next epoch if an old-domain
        # per-source robust score dropped too far below the baseline.
        if early_stop_enabled and epoch < epochs:
            current_scores = per_source_robust_scores(
                condition_metrics, list(early_stop_config["sources"])
            )
            max_allowed_drop = float(early_stop_config["max_source_drop"])
            drops = {
                source: reference_scores[source] - current_scores[source]
                for source in current_scores
            }
            violation = {
                source: drop for source, drop in drops.items() if drop > max_allowed_drop
            }
            guard_payload = {
                "epoch": epoch,
                "reference": reference_scores,
                "current": current_scores,
                "drops": drops,
                "max_allowed_drop": max_allowed_drop,
                "violation": violation,
                "triggered": bool(violation),
            }
            write_json(output_dir / f"early_stop_check_epoch_{epoch:02d}.json", guard_payload)
            for source, drop in drops.items():
                print(
                    f"early_stop_check epoch={epoch} {source}: "
                    f"current={current_scores[source]:.6f} reference={reference_scores[source]:.6f} "
                    f"drop={drop:+.6f} limit={max_allowed_drop}"
                )
            if violation:
                stopped_early = True
                print(f"EARLY STOP after epoch {epoch}: {violation}")
                break
    if writer is not None:
        writer.close()
    print(f"done stopped_early={stopped_early} best={best_score:.4f}")


if __name__ == "__main__":
    main()
