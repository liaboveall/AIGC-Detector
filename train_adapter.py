"""Residual adapter training entry point.

Frozen ConvNeXt-Tiny base (read-only outputs/multisource_blur_finetune/best.pt)
plus a zero-initialised residual MLP branch; only the ~197k branch parameters
train. Loss routing by manifest `dataset` column:
  - configured adaptation sources -> BCE(final_logit, label)
  - configured preservation sources -> penalty_weight * mean(residual_logit ** 2)

Flow per run:
  epoch 00: save the untouched wrapper, evaluate the 16-condition selection
            set, and REQUIRE item-by-item parity with the pre-registered
            baseline numbers (zero-init correctness); abort on any mismatch.
  epoch N : train, evaluate the configured condition suite, archive predictions,
            run the configured multi-objective gate, and stop or abort according
            to the config. A reproduction should always use a fresh output path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.adapter import (
    AdapterModel,
    adapter_parameter_counts,
    assert_zero_residual_identity,
)
from src.config import PROJECT_ROOT, dataset_paths, load_config, project_path
from src.data import ManifestImageDataset, RobustnessImageDataset, make_loader
from src.distill import per_source_robust_scores
from src.engine import evaluate_condition_suite
from src.metrics import robustness_summary
from src.model import create_model
from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_device, save_checkpoint, set_seed, write_json


CF_SOURCE_DEFAULT = "CommunityForensics-Small"
ALL_SOURCES = ("CommunityForensics-Small", "GenImage", "SID_Set")
PARITY_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual adapter training")
    parser.add_argument("--config", default="configs/adapter_v2.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Set to 1 to skip the epoch-00 baseline-parity evaluation (smoke runs only)",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Explicitly allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch_adapter(
    model: AdapterModel,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    penalty_weight: float,
    bce_sources: list[str],
    old_sources: list[str],
    gradient_clip: float,
    max_batches: int | None = None,
    step_log_path: Path | None = None,
) -> dict[str, Any]:
    """One adapter epoch with the source-routed mixed loss."""
    model.train()
    bce_source_set = set(bce_sources)
    old_source_set = set(old_sources)
    overlap = bce_source_set & old_source_set
    if overlap:
        raise ValueError(f"BCE and preservation source lists overlap: {sorted(overlap)}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters: adapter branch missing?")
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_bce = total_pen = 0.0
    total_examples = bce_examples = 0
    residual_stats: dict[str, dict[str, float]] = {}
    step_rows: list[dict[str, float]] = []
    started = time.perf_counter()
    processed = 0
    amp_enabled = device.type == "cuda"
    progress = tqdm(loader, desc="adapter-train", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        datasets = list(batch["dataset"])
        bce_mask = torch.tensor([d in bce_source_set for d in datasets], dtype=torch.bool, device=device)
        old_mask = torch.tensor([d in old_source_set for d in datasets], dtype=torch.bool, device=device)
        if bool((~(bce_mask | old_mask)).any()):
            routed_sources = bce_source_set | old_source_set
            unrouted = sorted({name for name in datasets if name not in routed_sources})
            raise ValueError(f"Training batch contains unrouted dataset sources: {unrouted}")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            final_logits, _base_logits, residual_logits = model.forward_with_residual(images)
            final_logits = final_logits.flatten()
            residual_logits = residual_logits.flatten()
            if bool(bce_mask.any()):
                bce = F.binary_cross_entropy_with_logits(final_logits[bce_mask], labels[bce_mask])
            else:
                bce = final_logits.new_zeros(())
            if bool(old_mask.any()):
                penalty = residual_logits[old_mask].float().pow(2).mean()
            else:
                penalty = residual_logits.new_zeros(())
            loss = bce + penalty_weight * penalty
        scaler.scale(loss).backward()
        processed += 1
        is_last = max_batches is not None and processed >= max_batches
        if processed % 1 == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = labels.numel()
        total_examples += batch_size
        bce_examples += int(bce_mask.sum())
        total_loss += float(loss.detach()) * batch_size
        total_bce += float(bce.detach()) * batch_size
        total_pen += float(penalty.detach()) * batch_size
        # Residual magnitude bookkeeping (fp32, per dataset).
        residual_abs = residual_logits.detach().float().abs().cpu()
        residual_signed = residual_logits.detach().float().cpu()
        labels_cpu = labels.float().cpu()
        for name, value, signed, label in zip(datasets, residual_abs, residual_signed, labels_cpu):
            entry = residual_stats.setdefault(
                name, {"sum_abs": 0.0, "sum_signed": 0.0, "count": 0, "sum_signed_fake": 0.0, "count_fake": 0, "sum_signed_real": 0.0, "count_real": 0}
            )
            entry["sum_abs"] += float(value)
            entry["sum_signed"] += float(signed)
            entry["count"] += 1
            if int(label) == 1:
                entry["sum_signed_fake"] += float(signed)
                entry["count_fake"] += 1
            else:
                entry["sum_signed_real"] += float(signed)
                entry["count_real"] += 1
        step_rows.append(
            {
                "batch": batch_index,
                "bce": float(bce.detach()),
                "penalty": float(penalty.detach()),
                "bce_fraction": int(bce_mask.sum()) / max(batch_size, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        progress.set_postfix(
            bce=f"{total_bce / total_examples:.4f}",
            pen=f"{total_pen / total_examples:.4f}",
        )
    elapsed = max(time.perf_counter() - started, 1e-9)
    if step_log_path is not None:
        step_log_path.parent.mkdir(parents=True, exist_ok=True)
        with step_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["batch", "bce", "penalty", "bce_fraction", "lr"])
            writer.writeheader()
            writer.writerows(step_rows)
    train_residual_summary = {
        name: {
            "mean_abs_residual": entry["sum_abs"] / max(entry["count"], 1),
            "mean_residual": entry["sum_signed"] / max(entry["count"], 1),
            "mean_residual_fake": entry["sum_signed_fake"] / max(entry["count_fake"], 1),
            "mean_residual_real": entry["sum_signed_real"] / max(entry["count_real"], 1),
            "count": entry["count"],
        }
        for name, entry in sorted(residual_stats.items())
    }
    return {
        "loss": total_loss / max(total_examples, 1),
        "bce": total_bce / max(total_examples, 1),
        "penalty": total_pen / max(total_examples, 1),
        "bce_fraction": bce_examples / max(total_examples, 1),
        "examples_per_second": total_examples / elapsed,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "residual_stats": train_residual_summary,
    }


@torch.inference_mode()
def validation_residual_stats(
    model: AdapterModel, loader: Any, device: torch.device, max_batches: int | None = None
) -> dict[str, Any]:
    """Clean-condition residual magnitude per dataset (+label split for CF)."""
    model.eval()
    stats: dict[str, dict[str, float]] = {}
    amp_enabled = device.type == "cuda"
    for batch_index, batch in enumerate(tqdm(loader, desc="residual-stats", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"]["clean"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _final, _base, residual = model.forward_with_residual(images)
        residual_cpu = residual.flatten().float().cpu()
        labels_cpu = batch["label"].cpu()
        for name, value, label in zip(batch["dataset"], residual_cpu, labels_cpu):
            entry = stats.setdefault(name, {"sum_abs": 0.0, "sum_signed": 0.0, "count": 0, "sum_fake": 0.0, "count_fake": 0, "sum_real": 0.0, "count_real": 0})
            entry["sum_abs"] += float(value.abs())
            entry["sum_signed"] += float(value)
            entry["count"] += 1
            if int(label) == 1:
                entry["sum_fake"] += float(value)
                entry["count_fake"] += 1
            else:
                entry["sum_real"] += float(value)
                entry["count_real"] += 1
    return {
        name: {
            "mean_abs_residual": entry["sum_abs"] / max(entry["count"], 1),
            "mean_residual": entry["sum_signed"] / max(entry["count"], 1),
            "mean_residual_fake": entry["sum_fake"] / max(entry["count_fake"], 1),
            "mean_residual_real": entry["sum_real"] / max(entry["count_real"], 1),
            "count": entry["count"],
        }
        for name, entry in sorted(stats.items())
    }


def write_predictions_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "dataset", "source_class", "label", "condition", "probability"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_gate_comparison(
    baseline_metrics: str,
    candidate_metrics: Path,
    output_json: Path,
    max_source_family_drop: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_robustness_candidates.py"),
        "--baseline",
        baseline_metrics,
        "--candidate",
        str(candidate_metrics),
        "--output",
        str(output_json),
        "--max-source-family-drop",
        f"{max_source_family_drop}",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"Gate comparison failed for {candidate_metrics}")
    with output_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_epoch0_parity(epoch0_metrics: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    """Require item-by-item equality between the untouched wrapper and the base."""
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    baseline_sources = per_source_robust_scores(baseline["conditions"], list(ALL_SOURCES))
    candidate_sources = per_source_robust_scores(epoch0_metrics["conditions"], list(ALL_SOURCES))
    pairs = {
        "overall_robust_score": (
            float(epoch0_metrics["robustness"]["robust_score"]),
            float(baseline["robustness"]["robust_score"]),
        ),
        "clean_auc": (
            float(epoch0_metrics["conditions"]["clean"]["overall"]["roc_auc"]),
            float(baseline["conditions"]["clean"]["overall"]["roc_auc"]),
        ),
    }
    for source in ALL_SOURCES:
        pairs[f"{source}_robust_score"] = (candidate_sources[source], baseline_sources[source])
    rows = []
    worst = 0.0
    for name, (candidate, reference) in pairs.items():
        difference = abs(candidate - reference)
        worst = max(worst, difference)
        rows.append({"metric": name, "epoch0": candidate, "baseline": reference, "abs_diff": difference})
    result = {"tolerance": PARITY_TOLERANCE, "max_abs_diff": worst, "rows": rows, "ok": worst <= PARITY_TOLERANCE}
    if not result["ok"]:
        raise RuntimeError(f"Epoch-0 parity FAILED vs baseline {baseline_path}: {rows}")
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    data_config = config["data"]
    training_config = config["training"]
    adapter_train_config = config["adapter_training"]
    adapter_config = config["adapter"]
    warmstart_config = config["warmstart"]
    judgement_config = config["judgement"]
    epochs = args.epochs or int(training_config["epochs"])
    max_train_batches = args.max_train_batches or training_config.get("max_batches_per_epoch")
    max_val_batches = args.max_val_batches or training_config.get("max_val_batches")
    output_dir = project_path(args.output_dir or config["output"]["directory"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {output_dir}. "
            "Choose a new --output-dir. Use --allow-existing-output only for a "
            "deliberate continuation; never use it on the frozen release directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    warmstart_path = project_path(warmstart_config["checkpoint"])
    baseline_metrics = str(project_path(judgement_config["baseline_metrics"]))
    num_workers = int(data_config["num_workers"] if args.num_workers is None else args.num_workers)
    runtime = {
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "num_workers": num_workers,
        "base_checkpoint": str(warmstart_path),
        "baseline_metrics": baseline_metrics,
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

    # Base: read-only load of the warmstart checkpoint, then wrap.
    warmstart = torch.load(warmstart_path, map_location="cpu", weights_only=False)
    warmstart_model_name = warmstart.get("config", {}).get("model", {}).get("name")
    if warmstart_model_name and warmstart_model_name != config["model"].get("name"):
        raise ValueError(
            f"Base checkpoint model {warmstart_model_name!r} != config model {config['model'].get('name')!r}"
        )
    base = create_model(config["model"], pretrained_override=False)
    base.load_state_dict(warmstart["model_state"])
    del warmstart
    model = AdapterModel(
        base,
        feature_dim=int(adapter_config.get("feature_dim", 768)),
        hidden_dim=int(adapter_config.get("hidden_dim", 256)),
        residual_gain=float(adapter_config.get("residual_gain", 1.0)),
    )
    counts = adapter_parameter_counts(model)
    if counts["trainable"] != counts["adapter_branch"]:
        raise RuntimeError("Trainable parameters must equal the adapter branch parameters")
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    zero_diff = assert_zero_residual_identity(model, device)
    print(
        f"params: base_frozen={counts['base_frozen']:,} adapter_branch={counts['adapter_branch']:,} "
        f"trainable={counts['trainable']:,} total={counts['total']:,}"
    )
    print(f"zero_init_identity max_abs_diff={zero_diff:.3e}")
    print(f"device={device} train={len(train_dataset)} val={len(validation_dataset)}")

    checkpoint_config = {k: v for k, v in config.items() if not k.startswith("_")}
    criterion = torch.nn.BCEWithLogitsLoss()

    # ---- Epoch 00: untouched wrapper must reproduce the baseline exactly ----
    if args.start_epoch <= 0:
        print(f"epoch=0 conditions={','.join(conditions)}")
        epoch0_rows: list[dict] = []
        epoch0_conditions = evaluate_condition_suite(
            model,
            validation_loader,
            criterion,
            device,
            conditions,
            max_batches=max_val_batches,
            prediction_rows=epoch0_rows,
        )
        epoch0_metrics = {
            "epoch": 0,
            "runtime": runtime,
            "conditions": epoch0_conditions,
            "robustness": robustness_summary(epoch0_conditions),
            "parameter_counts": counts,
            "note": "untrained zero-initialised adapter; must equal the baseline",
        }
        write_json(output_dir / "metrics_epoch_00.json", epoch0_metrics)
        write_predictions_csv(output_dir / "predictions_epoch_00.csv", epoch0_rows)
        save_checkpoint(
            output_dir / "epoch_00.pt",
            {
                "epoch": 0,
                "model_state": model.state_dict(),
                "config": checkpoint_config,
                "parameter_counts": counts,
                "runtime": runtime,
            },
        )
        parity = check_epoch0_parity(epoch0_metrics, Path(baseline_metrics))
        write_json(output_dir / "epoch00_parity.json", parity)
        for row in parity["rows"]:
            print(
                f"parity {row['metric']}: epoch0={row['epoch0']:.6f} baseline={row['baseline']:.6f} "
                f"diff={row['abs_diff']:.2e}"
            )
        print("epoch=0 parity OK; running gate comparison for the record")
        run_gate_comparison(
            baseline_metrics,
            output_dir / "metrics_epoch_00.json",
            output_dir / "gate_epoch_00.json",
            float(judgement_config["max_source_family_drop"]),
        )
    else:
        print("WARNING: epoch-00 baseline-parity evaluation skipped (--start-epoch > 0)")

    # ---- Training epochs ----
    cf_source = str(adapter_train_config.get("cf_source", CF_SOURCE_DEFAULT))
    bce_sources = list(adapter_train_config.get("bce_sources", [cf_source]))
    old_sources = list(adapter_train_config["old_sources"])
    penalty_weight = float(adapter_train_config["penalty_weight"])
    cf_target = float(adapter_train_config["cf_target_robust_score"])
    abort_max_drop = float(adapter_train_config["abort_max_source_drop"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(adapter_train_config["learning_rate"]),
        weight_decay=float(adapter_train_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    gradient_clip = float(adapter_train_config.get("gradient_clip", 1.0))
    with Path(baseline_metrics).open("r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    baseline_source_scores = per_source_robust_scores(baseline_payload["conditions"], old_sources)
    best_score = -math.inf
    stop_reason = "epochs_exhausted"
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch_adapter(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            penalty_weight,
            bce_sources,
            old_sources,
            gradient_clip,
            max_batches=max_train_batches,
            step_log_path=output_dir / f"train_steps_epoch_{epoch:02d}.csv",
        )
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        print(f"epoch={epoch} conditions={','.join(conditions)}")
        prediction_rows: list[dict] = []
        condition_metrics = evaluate_condition_suite(
            model,
            validation_loader,
            criterion,
            device,
            conditions,
            max_batches=max_val_batches,
            prediction_rows=prediction_rows,
        )
        write_predictions_csv(output_dir / f"predictions_epoch_{epoch:02d}.csv", prediction_rows)
        residual_stats_val = validation_residual_stats(model, validation_loader, device, max_val_batches)
        robustness = robustness_summary(condition_metrics)
        clean_auc = float(condition_metrics["clean"]["overall"]["roc_auc"])
        score = float(robustness["robust_score"])
        epoch_result = {
            "epoch": epoch,
            "runtime": runtime,
            "train": train_metrics,
            "conditions": condition_metrics,
            "robustness": robustness,
            "residual_stats_validation_clean": residual_stats_val,
            "peak_gpu_memory_bytes": peak_memory_bytes,
        }
        write_json(output_dir / f"metrics_epoch_{epoch:02d}.json", epoch_result)
        history_row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "bce": train_metrics["bce"],
            "penalty": train_metrics["penalty"],
            "examples_per_second": train_metrics["examples_per_second"],
            "clean_auc": clean_auc,
            "mean_degraded_auc": robustness["mean_degraded_auc"],
            "worst_degraded_auc": robustness["worst_degraded_auc"],
            "robust_score": score,
        }
        append_history(output_dir / "history.csv", history_row)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": checkpoint_config,
            "metrics": epoch_result,
            "parameter_counts": counts,
            "runtime": runtime,
        }
        save_checkpoint(output_dir / "last.pt", checkpoint)
        if config["output"].get("save_epoch_checkpoints", False):
            epoch_checkpoint = {k: v for k, v in checkpoint.items() if k != "optimizer_state"}
            save_checkpoint(output_dir / f"epoch_{epoch:02d}.pt", epoch_checkpoint)
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_checkpoint = {k: v for k, v in checkpoint.items() if k != "optimizer_state"}
            save_checkpoint(output_dir / "best.pt", best_checkpoint)
        gate_result = run_gate_comparison(
            baseline_metrics,
            output_dir / f"metrics_epoch_{epoch:02d}.json",
            output_dir / f"gate_epoch_{epoch:02d}.json",
            float(judgement_config["max_source_family_drop"]),
        )
        comparison = gate_result["comparisons"][0]
        cf_score = float(comparison["source_robust_scores"][CF_SOURCE_DEFAULT])
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} bce={train_metrics['bce']:.4f} "
            f"pen={train_metrics['penalty']:.4f} clean_auc={clean_auc:.4f} "
            f"robust={score:.6f} cf={cf_score:.6f} gates_pass={comparison['accepted']} best={best_score:.6f}"
        )
        if comparison["accepted"]:
            stop_reason = f"all_gates_passed_epoch_{epoch}"
            break
        if cf_score >= cf_target:
            stop_reason = f"cf_target_reached_epoch_{epoch}"
            break
        current_source_scores = per_source_robust_scores(condition_metrics, old_sources)
        aborts = {
            source: baseline_source_scores[source] - current_source_scores[source]
            for source in old_sources
            if baseline_source_scores[source] - current_source_scores[source] > abort_max_drop
        }
        if aborts:
            stop_reason = f"old_domain_abort_epoch_{epoch}: {aborts}"
            write_json(output_dir / "old_domain_abort.json", {"epoch": epoch, "aborts": aborts})
            print(f"ABORT: old-domain drop beyond {abort_max_drop}: {aborts}")
            break
    print(f"done stop_reason={stop_reason} best_robust={best_score:.6f}")


if __name__ == "__main__":
    main()
