"""Preregistered Base v3 phased training entry point."""
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

from src.base_v3 import (
    build_phase_param_groups,
    cosine_schedule,
    make_quota_loader,
    set_phase_trainability,
    train_one_epoch_base_v3,
)
from src.config import dataset_paths, load_config, project_path
from src.data import ManifestImageDataset, RobustnessImageDataset, make_loader
from src.distill import load_teacher, per_source_robust_scores
from src.engine import evaluate_condition_suite
from src.metrics import robustness_summary
from src.model import create_model, parameter_counts
from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train preregistered Base v3")
    parser.add_argument("--config", default="configs/base_v3.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-epochs", type=int, help="Smoke-only global epoch cap")
    parser.add_argument(
        "--no-divergence-stop",
        action="store_true",
        help="Smoke-only: do not apply source metrics computed from truncated validation",
    )
    return parser.parse_args()


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_reference(path: str | Path) -> dict[str, float]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return per_source_robust_scores(payload["conditions"], ("GenImage", "SID_Set"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    data_config = config["data"]
    training_config = config["training"]
    distill_config = config["distillation"]
    output_dir = project_path(args.output_dir or config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
    max_train_batches = args.max_train_batches or training_config.get("max_batches_per_epoch")
    max_val_batches = args.max_val_batches or training_config.get("max_val_batches")
    phases = list(training_config["phases"])
    total_declared_epochs = sum(int(phase["epochs"]) for phase in phases)
    max_epochs = min(args.max_epochs or total_declared_epochs, total_declared_epochs)
    runtime = {
        "output_directory": str(output_dir),
        "device": str(device),
        "num_workers": num_workers,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_epochs": max_epochs,
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
            degradation_kind_weights=data_config["train_degradation_kind_weights"],
            blur_weights=data_config["train_blur_weights"],
            reencode_probability=float(data_config["train_reencode_probability"]),
            reencode_qualities=data_config["train_reencode_qualities"],
            reencode_codecs=data_config["train_reencode_codecs"],
        ),
        training=True,
        max_samples=data_config.get("max_train_samples"),
        seed=seed,
    )
    train_loader, quota_sampler = make_quota_loader(
        train_dataset,
        num_workers=num_workers,
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
        num_workers=num_workers,
        training=False,
        balanced_sampling=False,
        seed=seed,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=True,
    )

    if not bool(config["model"].get("pretrained", False)):
        raise ValueError("Base v3 must start from declared pretrained weights")
    student = create_model(config["model"])
    teacher = load_teacher(project_path(config["teacher"]["checkpoint"]), device)
    student.to(device)
    if device.type == "cuda":
        student.to(memory_format=torch.channels_last)
    student_feature_dim = int(student.head.fc.in_features)
    teacher_feature_dim = int(config["teacher"]["feature_dim"])
    projection = torch.nn.Linear(student_feature_dim, teacher_feature_dim, bias=False).to(device)
    torch.nn.init.orthogonal_(projection.weight)
    reference = load_reference(training_config["reference_metrics"])
    print(
        f"Base v3 device={device} train_rows={len(train_dataset):,} "
        f"quota_batches={len(quota_sampler):,} val_rows={len(val_dataset):,}"
    )
    print(
        f"student_total={parameter_counts(student)['total']:,} teacher_total="
        f"{sum(parameter.numel() for parameter in teacher.parameters()):,} "
        f"projection={sum(parameter.numel() for parameter in projection.parameters()):,}"
    )
    print(f"reference GenImage={reference['GenImage']:.6f} SID_Set={reference['SID_Set']:.6f}")

    writer = SummaryWriter(output_dir / "tensorboard") if config["output"].get("tensorboard", True) else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    accumulation_steps = int(training_config["accumulation_steps"])
    batch_count = min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader)
    global_epoch = 0
    stopped_for_divergence = False
    for phase in phases:
        if global_epoch >= max_epochs:
            break
        phase_epochs = min(int(phase["epochs"]), max_epochs - global_epoch)
        active_prefixes = list(phase["active_prefixes"])
        set_phase_trainability(student, active_prefixes)
        groups = build_phase_param_groups(
            student,
            projection,
            {key: float(value) for key, value in training_config["learning_rates"].items()},
            float(distill_config["projection_learning_rate"]),
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=float(training_config["weight_decay"]))
        optimizer_steps = math.ceil(batch_count / accumulation_steps) * phase_epochs
        scheduler = cosine_schedule(
            optimizer,
            optimizer_steps=optimizer_steps,
            warmup_ratio=float(training_config["warmup_ratio"]),
        )
        print(f"phase={phase['name']} epochs={phase_epochs} active={active_prefixes}")
        for group in groups:
            print(
                f"  group={group['name']} lr={group['lr']:.2e} "
                f"params={sum(parameter.numel() for parameter in group['params']):,}"
            )
        for _ in range(phase_epochs):
            global_epoch += 1
            quota_sampler.set_epoch(global_epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics = train_one_epoch_base_v3(
                student,
                teacher,
                projection,
                train_loader,
                optimizer,
                scheduler,
                device,
                scaler,
                accumulation_steps=accumulation_steps,
                gradient_clip=float(training_config["gradient_clip"]),
                logit_weight=float(distill_config["logit_weight"]),
                logit_delta=float(distill_config["logit_delta"]),
                feature_weight=float(phase["feature_weight"]),
                max_batches=max_train_batches,
            )
            peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            condition_metrics = evaluate_condition_suite(
                student,
                val_loader,
                torch.nn.BCEWithLogitsLoss(),
                device,
                conditions,
                max_batches=max_val_batches,
            )
            robustness = robustness_summary(condition_metrics)
            source_scores = per_source_robust_scores(condition_metrics, ("GenImage", "SID_Set"))
            source_drops = {source: reference[source] - score for source, score in source_scores.items()}
            result = {
                "epoch": global_epoch,
                "phase": phase["name"],
                "train": train_metrics,
                "conditions": condition_metrics,
                "robustness": robustness,
                "source_robust_scores": source_scores,
                "source_robust_drops": source_drops,
                "peak_gpu_memory_bytes": peak_memory,
                "runtime": runtime,
            }
            write_json(output_dir / f"metrics_epoch_{global_epoch:02d}.json", result)
            deployable_checkpoint = {
                "epoch": global_epoch,
                "phase": phase["name"],
                "model_state": student.state_dict(),
                "config": {key: value for key, value in config.items() if not key.startswith("_")},
                "metrics": result,
                "parameter_counts": parameter_counts(student),
                "training_projection_state": projection.state_dict(),
                "runtime": runtime,
            }
            save_checkpoint(output_dir / f"epoch_{global_epoch:02d}.pt", deployable_checkpoint)
            history = {
                "epoch": global_epoch,
                "phase": phase["name"],
                **train_metrics,
                "robust_score": robustness["robust_score"],
                "clean_auc": condition_metrics["clean"]["overall"]["roc_auc"],
                "GenImage_robust": source_scores["GenImage"],
                "SID_Set_robust": source_scores["SID_Set"],
                "peak_gpu_memory_bytes": peak_memory,
            }
            append_csv(output_dir / "history.csv", history)
            if writer is not None:
                for key, value in history.items():
                    if key not in {"epoch", "phase"}:
                        writer.add_scalar(key, value, global_epoch)
            print(
                f"epoch={global_epoch} phase={phase['name']} loss={train_metrics['loss']:.4f} "
                f"ex/s={train_metrics['examples_per_second']:.1f} robust={robustness['robust_score']:.6f} "
                f"GenImage={source_scores['GenImage']:.6f} drop={source_drops['GenImage']:+.6f} "
                f"SID={source_scores['SID_Set']:.6f} drop={source_drops['SID_Set']:+.6f}"
            )
            if not args.no_divergence_stop and global_epoch >= 2 and any(
                drop > float(training_config["divergence_max_source_drop"])
                for drop in source_drops.values()
            ):
                stopped_for_divergence = True
                write_json(
                    output_dir / "divergence_stop.json",
                    {"epoch": global_epoch, "source_drops": source_drops},
                )
                print(f"DIVERGENCE STOP epoch={global_epoch}: {source_drops}")
                break
        if stopped_for_divergence:
            break
    if writer is not None:
        writer.close()
    print(f"done epochs={global_epoch} stopped_for_divergence={stopped_for_divergence}")


if __name__ == "__main__":
    main()
