from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from tqdm import tqdm

from .metrics import binary_metrics, grouped_metrics, source_contrast_metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    accumulation_steps: int,
    gradient_clip: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    started = time.perf_counter()
    processed_batches = 0
    amp_enabled = device.type == "cuda"
    progress = tqdm(loader, desc="train", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images).flatten()
            raw_loss = criterion(logits, labels)
            loss = raw_loss / accumulation_steps
        scaler.scale(loss).backward()
        processed_batches += 1
        is_last = max_batches is not None and processed_batches >= max_batches
        should_step = processed_batches % accumulation_steps == 0 or is_last
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_ran = scaler.get_scale() >= scale_before_step
            if scheduler is not None and optimizer_ran:
                scheduler.step()
        batch_size = labels.numel()
        total_examples += batch_size
        total_loss += float(raw_loss.detach()) * batch_size
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")
    if processed_batches % accumulation_steps != 0 and not (
        max_batches is not None and processed_batches >= max_batches
    ):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_ran = scaler.get_scale() >= scale_before_step
        if scheduler is not None and optimizer_ran:
            scheduler.step()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "loss": total_loss / max(total_examples, 1),
        "examples_per_second": total_examples / elapsed,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    probabilities: list[float] = []
    labels: list[float] = []
    source_classes: list[str] = []
    datasets: list[str] = []
    total_loss = 0.0
    total_examples = 0
    amp_enabled = device.type == "cuda"
    for batch_index, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images).flatten()
            loss = criterion(logits, targets)
        batch_probabilities = torch.sigmoid(logits).float().cpu().tolist()
        batch_labels = targets.float().cpu().tolist()
        probabilities.extend(batch_probabilities)
        labels.extend(batch_labels)
        source_classes.extend(batch["source_class"])
        datasets.extend(batch["dataset"])
        total_examples += len(batch_labels)
        total_loss += float(loss) * len(batch_labels)
    result = {
        "loss": total_loss / max(total_examples, 1),
        "overall": binary_metrics(labels, probabilities),
        "by_source_class": grouped_metrics(labels, probabilities, source_classes),
        "source_contrasts": source_contrast_metrics(labels, probabilities, source_classes),
        "by_dataset": grouped_metrics(labels, probabilities, datasets),
    }
    return result


@torch.inference_mode()
def evaluate_condition_suite(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
    conditions: list[str],
    max_batches: int | None = None,
    prediction_rows: list[dict] | None = None,
) -> dict[str, dict]:
    """Evaluate every condition from one shared validation loader."""
    model.eval()
    accumulators = {
        condition: {
            "probabilities": [],
            "labels": [],
            "source_classes": [],
            "datasets": [],
            "loss": 0.0,
            "count": 0,
        }
        for condition in conditions
    }
    amp_enabled = device.type == "cuda"
    for batch_index, batch in enumerate(tqdm(loader, desc="robust-eval", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        targets = batch["label"].to(device, non_blocking=True)
        batch_labels = targets.float().cpu().tolist()
        batch_paths = list(batch["path"])
        batch_source_classes = list(batch["source_class"])
        batch_datasets = list(batch["dataset"])
        for condition in conditions:
            images = batch["images"][condition].to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images).flatten()
                loss = criterion(logits, targets)
            batch_probabilities = torch.sigmoid(logits).float().cpu().tolist()
            accumulator = accumulators[condition]
            accumulator["probabilities"].extend(batch_probabilities)
            accumulator["labels"].extend(batch_labels)
            accumulator["source_classes"].extend(batch_source_classes)
            accumulator["datasets"].extend(batch_datasets)
            accumulator["count"] += len(batch_labels)
            accumulator["loss"] += float(loss) * len(batch_labels)
            if prediction_rows is not None:
                prediction_rows.extend(
                    {
                        "path": path,
                        "dataset": dataset,
                        "source_class": source_class,
                        "label": int(label),
                        "condition": condition,
                        "probability": probability,
                    }
                    for path, dataset, source_class, label, probability in zip(
                        batch_paths,
                        batch_datasets,
                        batch_source_classes,
                        batch_labels,
                        batch_probabilities,
                        strict=True,
                    )
                )

    results = {}
    for condition, accumulator in accumulators.items():
        labels = accumulator["labels"]
        probabilities = accumulator["probabilities"]
        results[condition] = {
            "loss": accumulator["loss"] / max(accumulator["count"], 1),
            "overall": binary_metrics(labels, probabilities),
            "by_source_class": grouped_metrics(labels, probabilities, accumulator["source_classes"]),
            "source_contrasts": source_contrast_metrics(
                labels, probabilities, accumulator["source_classes"]
            ),
            "by_dataset": grouped_metrics(labels, probabilities, accumulator["datasets"]),
        }
    return results
