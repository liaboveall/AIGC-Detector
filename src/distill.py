"""Knowledge distillation helpers for the replay-distill experiment.

This is a NEW module: no shared src/ file is modified. The teacher is a frozen
copy of the hot-start checkpoint (eval mode, requires_grad=False, no_grad
forward). The student freezes stem + stages.0/1 for the whole run and trains
stages.2/3 and head with separate learning rates.

Loss: L = BCE(student_logits, labels) on all samples
        + kd_weight * KD on old-domain samples only (GenImage / SID_Set).
KD (single-logit binary distillation, temperature-scaled soft targets):
        KD = BCEWithLogits(zs / T, sigmoid(zt / T).detach()) * T**2
(the logits form is numerically stable; sigmoid(scaled) + BCE would risk
inf when student logits saturate)
"""
from __future__ import annotations

import csv
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .adapter import build_checkpoint_model

DEFAULT_FROZEN_PREFIXES = ("stem.", "stages.0.", "stages.1.")
HEAD_PREFIX = "head."
TRAINABLE_STAGE_PREFIXES = ("stages.2.", "stages.3.")


def load_teacher(
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Build and freeze the model described by a teacher checkpoint.

    The teacher architecture is intentionally read from its own checkpoint so
    a larger student can distil from the accepted Tiny detector. Legacy bare
    checkpoints and adapter-enabled checkpoints are both supported.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = build_checkpoint_model(checkpoint["config"], checkpoint["model_state"])
    teacher.to(device)
    if device.type == "cuda":
        teacher.to(memory_format=torch.channels_last)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def freeze_prefixes(model: torch.nn.Module, prefixes: Iterable[str]) -> int:
    """Freeze every parameter whose name starts with one of the prefixes."""
    prefix_tuple = tuple(prefixes)
    frozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix_tuple):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def build_param_groups(
    model: torch.nn.Module,
    head_learning_rate: float,
    stages_learning_rate: float,
) -> list[dict[str, Any]]:
    """Two explicit groups: head (final LayerNorm + fc) and stages.2 + stages.3.

    Raises if any trainable parameter falls outside these groups, so an
    accidental unfrozen stem/stages.0/1 parameter can never slip through.
    """
    head_params: list[torch.nn.Parameter] = []
    stage_params: list[torch.nn.Parameter] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(HEAD_PREFIX):
            head_params.append(parameter)
        elif name.startswith(TRAINABLE_STAGE_PREFIXES):
            stage_params.append(parameter)
        else:
            unexpected.append(name)
    if unexpected:
        raise ValueError(f"Trainable parameters outside head/stages.2/3: {unexpected}")
    if not head_params or not stage_params:
        raise ValueError("Empty parameter group: check freezing configuration")
    return [
        {"params": head_params, "lr": head_learning_rate, "name": "head"},
        {"params": stage_params, "lr": stages_learning_rate, "name": "stages_2_3"},
    ]


def group_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    counts = {"frozen": 0, "head": 0, "stages_2_3": 0, "trainable": 0, "total": 0}
    for name, parameter in model.named_parameters():
        numel = parameter.numel()
        counts["total"] += numel
        if not parameter.requires_grad:
            counts["frozen"] += numel
        elif name.startswith(HEAD_PREFIX):
            counts["head"] += numel
            counts["trainable"] += numel
        elif name.startswith(TRAINABLE_STAGE_PREFIXES):
            counts["stages_2_3"] += numel
            counts["trainable"] += numel
        else:
            raise ValueError(f"Trainable parameter outside known groups: {name}")
    return counts


def kd_loss_binary(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled binary distillation on the masked (old-domain) samples.

    Uses binary_cross_entropy_with_logits on the temperature-scaled student
    logits against detached teacher soft targets: numerically stable even when
    student logits saturate (sigmoid + BCE would risk inf).
    """
    if not bool(mask.any()):
        return student_logits.new_zeros(())
    student_scaled = student_logits[mask].float() / temperature
    teacher_soft = torch.sigmoid(teacher_logits[mask].float() / temperature).detach()
    loss = F.binary_cross_entropy_with_logits(student_scaled, teacher_soft)
    return loss * (temperature**2)


def train_one_epoch_distill(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    accumulation_steps: int,
    gradient_clip: float,
    temperature: float,
    kd_weight: float,
    kd_sources: Iterable[str],
    max_batches: int | None = None,
    step_log_path: str | Path | None = None,
    check_teacher_grad: bool = False,
) -> dict[str, Any]:
    """One distillation epoch. Returns mean loss components and throughput."""
    student.train()
    teacher.eval()
    kd_source_set = set(kd_sources)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_bce = 0.0
    total_kd = 0.0
    total_loss = 0.0
    total_examples = 0
    kd_examples = 0
    teacher_grad_all_none: bool | None = None
    step_rows: list[dict[str, float]] = []
    started = time.perf_counter()
    processed_batches = 0
    amp_enabled = device.type == "cuda"
    progress = tqdm(loader, desc="distill", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        mask = torch.tensor(
            [source in kd_source_set for source in batch["dataset"]],
            dtype=torch.bool,
            device=device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            student_logits = student(images).flatten()
            bce = criterion(student_logits, labels)
            with torch.no_grad():
                teacher_logits = teacher(images).flatten()
        kd = kd_loss_binary(student_logits, teacher_logits, mask, temperature)
        loss = bce + kd_weight * kd
        scaler.scale(loss / accumulation_steps).backward()
        processed_batches += 1
        is_last = max_batches is not None and processed_batches >= max_batches
        should_step = processed_batches % accumulation_steps == 0 or is_last
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_ran = scaler.get_scale() >= scale_before_step
            if scheduler is not None and optimizer_ran:
                scheduler.step()
            if check_teacher_grad and teacher_grad_all_none is None:
                teacher_grad_all_none = all(p.grad is None for p in teacher.parameters())
        batch_size = labels.numel()
        batch_kd = int(mask.sum())
        total_examples += batch_size
        kd_examples += batch_kd
        total_bce += float(bce.detach()) * batch_size
        total_kd += float(kd.detach()) * batch_size
        total_loss += float((bce + kd_weight * kd).detach()) * batch_size
        step_rows.append(
            {
                "batch": batch_index,
                "bce": float(bce.detach()),
                "kd": float(kd.detach()),
                "kd_fraction": batch_kd / max(batch_size, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        progress.set_postfix(
            bce=f"{total_bce / total_examples:.4f}",
            kd=f"{total_kd / total_examples:.4f}",
            kdf=f"{kd_examples / total_examples:.2f}",
        )
    elapsed = max(time.perf_counter() - started, 1e-9)
    if step_log_path is not None:
        step_log_path = Path(step_log_path)
        step_log_path.parent.mkdir(parents=True, exist_ok=True)
        with step_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["batch", "bce", "kd", "kd_fraction", "lr"])
            writer.writeheader()
            writer.writerows(step_rows)
    result = {
        "loss": total_loss / max(total_examples, 1),
        "bce": total_bce / max(total_examples, 1),
        "kd": total_kd / max(total_examples, 1),
        "kd_fraction": kd_examples / max(total_examples, 1),
        "examples_per_second": total_examples / elapsed,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    if teacher_grad_all_none is not None:
        result["teacher_grad_all_none"] = teacher_grad_all_none
    return result


def per_source_robust_scores(
    condition_metrics: dict[str, dict],
    sources: Iterable[str],
) -> dict[str, float]:
    """Per-source robust score = 0.8 * mean + 0.2 * worst over degraded conditions.

    Mirrors scripts/compare_robustness_candidates.py exactly, for early stopping.
    """
    degraded = [name for name in condition_metrics if name != "clean"]
    scores: dict[str, float] = {}
    for source in sources:
        aucs = [
            float(condition_metrics[name]["by_dataset"][source]["roc_auc"]) for name in degraded
        ]
        scores[source] = 0.8 * (sum(aucs) / len(aucs)) + 0.2 * min(aucs)
    non_finite = {source: score for source, score in scores.items() if not math.isfinite(score)}
    if non_finite:
        raise ValueError(f"Non-finite per-source robust scores (early-stop guard): {non_finite}")
    return scores
