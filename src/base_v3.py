"""Base v3 quota sampling and cross-architecture distillation helpers."""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .data import ManifestImageDataset, seed_worker


HISTORICAL_DATASETS = ("CommunityForensics-Small", "GenImage", "SID_Set")
OLD_KD_DATASETS = frozenset(("GenImage", "SID_Set"))


class BaseV3QuotaBatchSampler(Sampler[list[int]]):
    """Yield exact 2x4 source/label batches without replacement per epoch.

    Each batch has real/fake pairs from CF, GenImage and SID, plus a modern
    SuSy-real / (SuSy or MS-COCOAI)-fake pair. Modern fake generators rotate
    round-robin, while all individual pools are independently shuffled.
    """

    def __init__(self, frame: pd.DataFrame, seed: int = 2026) -> None:
        self.frame = frame.reset_index(drop=True)
        self.seed = int(seed)
        self.epoch = 0
        self.pools: dict[str, np.ndarray] = {}
        for dataset in HISTORICAL_DATASETS:
            for label in (0, 1):
                key = f"{dataset}:{label}"
                mask = (self.frame["dataset"] == dataset) & (
                    self.frame["binary_label"].astype(int) == label
                )
                self.pools[key] = self.frame.index[mask].to_numpy(dtype=np.int64)
        modern_real = (self.frame["dataset"] == "SuSy") & (
            self.frame["binary_label"].astype(int) == 0
        )
        self.pools["modern:0"] = self.frame.index[modern_real].to_numpy(dtype=np.int64)

        modern_fake = self.frame[
            self.frame["dataset"].isin(("SuSy", "MS-COCOAI"))
            & (self.frame["binary_label"].astype(int) == 1)
        ].copy()
        if modern_fake.empty:
            raise ValueError("Base v3 manifest has no modern fake examples")
        generator = modern_fake.get("generator", pd.Series("", index=modern_fake.index)).astype(str)
        source_class = modern_fake["source_class"].astype(str)
        modern_fake["_quota_generator"] = (
            modern_fake["dataset"].astype(str)
            + ":"
            + generator.where(generator.str.len() > 0, source_class)
        )
        self.modern_fake_pools = {
            str(name): group.index.to_numpy(dtype=np.int64)
            for name, group in modern_fake.groupby("_quota_generator", sort=True)
        }
        required = [*self.pools.values(), *self.modern_fake_pools.values()]
        if any(len(pool) == 0 for pool in required):
            raise ValueError("Base v3 quota sampler contains an empty source/label pool")
        modern_fake_count = sum(len(pool) for pool in self.modern_fake_pools.values())
        self.num_batches = min(
            *(len(pool) for pool in self.pools.values()),
            modern_fake_count,
        )
        generator_sizes = set(map(len, self.modern_fake_pools.values()))
        if len(generator_sizes) != 1:
            raise ValueError(
                "Modern fake generator pools must be balanced for strict round-robin; "
                f"sizes={sorted(generator_sizes)}"
            )
        if modern_fake_count != self.num_batches:
            raise ValueError(
                "Modern fake pool must contain exactly one item per quota batch; "
                f"fake={modern_fake_count}, batches={self.num_batches}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def _shuffled(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = values.copy()
        rng.shuffle(result)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 10_007)
        pools = {name: self._shuffled(pool, rng) for name, pool in self.pools.items()}
        generator_names = sorted(self.modern_fake_pools)
        rng.shuffle(generator_names)
        generator_pools = {
            name: self._shuffled(self.modern_fake_pools[name], rng) for name in generator_names
        }
        positions = {name: 0 for name in generator_names}
        modern_fake: list[int] = []
        while len(modern_fake) < self.num_batches:
            for name in generator_names:
                position = positions[name]
                if position < len(generator_pools[name]):
                    modern_fake.append(int(generator_pools[name][position]))
                    positions[name] += 1
                    if len(modern_fake) == self.num_batches:
                        break
        for offset in range(self.num_batches):
            batch = []
            for dataset in HISTORICAL_DATASETS:
                batch.append(int(pools[f"{dataset}:0"][offset]))
                batch.append(int(pools[f"{dataset}:1"][offset]))
            batch.append(int(pools["modern:0"][offset]))
            batch.append(modern_fake[offset])
            rng.shuffle(batch)
            yield batch


def make_quota_loader(
    dataset: ManifestImageDataset,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
) -> tuple[DataLoader[dict[str, Any]], BaseV3QuotaBatchSampler]:
    sampler = BaseV3QuotaBatchSampler(dataset.frame, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, sampler


def quota_signature(batch: dict[str, Any]) -> Counter[tuple[str, int]]:
    signature: Counter[tuple[str, int]] = Counter()
    labels = [int(value) for value in batch["label"].tolist()]
    for dataset, label in zip(batch["dataset"], labels, strict=True):
        bucket = dataset if dataset in HISTORICAL_DATASETS else "modern"
        signature[(bucket, label)] += 1
    return signature


def validate_quota_batch(batch: dict[str, Any]) -> None:
    expected = Counter(
        {(dataset, label): 1 for dataset in (*HISTORICAL_DATASETS, "modern") for label in (0, 1)}
    )
    actual = quota_signature(batch)
    if actual != expected:
        raise RuntimeError(f"Invalid Base v3 quota batch: actual={actual}, expected={expected}")


def set_phase_trainability(model: torch.nn.Module, active_prefixes: Iterable[str]) -> None:
    active = tuple(active_prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(active)
    unexpected = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith(active)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected Base v3 trainable parameters: {unexpected}")


def build_phase_param_groups(
    model: torch.nn.Module,
    projection: torch.nn.Module,
    learning_rates: dict[str, float],
    projection_learning_rate: float,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    assigned: set[int] = set()
    for prefix, learning_rate in learning_rates.items():
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(prefix)
        ]
        if parameters:
            groups.append({"params": parameters, "lr": float(learning_rate), "name": prefix.rstrip(".")})
            assigned.update(map(id, parameters))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    missing = [parameter for parameter in trainable if id(parameter) not in assigned]
    if missing:
        raise ValueError(f"{len(missing)} trainable model parameters have no learning-rate group")
    groups.append(
        {
            "params": list(projection.parameters()),
            "lr": float(projection_learning_rate),
            "name": "feature_projection",
        }
    )
    return groups


def model_logits_and_features(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detector logits and pooled pre-logit features without a second backbone pass."""
    # Accepted Tiny vNext is the pooled-feature AdapterModel.
    if hasattr(model, "pooled_features") and hasattr(model, "adapter") and hasattr(model, "base"):
        flat = model.pooled_features(images)
        head = model.base.head
        base_logit = head.fc(head.drop(head.pre_logits(flat)))
        residual = model.adapter(flat)
        logits = base_logit + float(model.residual_gain) * residual
        return logits.flatten(), flat
    feature_map = model.forward_features(images)
    head = model.head
    flat = head.flatten(head.norm(head.global_pool(feature_map)))
    logits = head.fc(head.drop(head.pre_logits(flat)))
    return logits.flatten(), flat


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    optimizer_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(round(optimizer_steps * warmup_ratio))

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch_base_v3(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    projection: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    accumulation_steps: int,
    gradient_clip: float,
    logit_weight: float,
    logit_delta: float,
    feature_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    projection.train()
    optimizer.zero_grad(set_to_none=True)
    totals = Counter()
    examples = 0
    old_examples = 0
    processed = 0
    started = time.perf_counter()
    progress = tqdm(loader, desc="base-v3", leave=False)
    amp_enabled = device.type == "cuda"
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        validate_quota_batch(batch)
        images = batch["image"].to(device, non_blocking=True)
        if amp_enabled:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)
        old_mask = torch.tensor(
            [dataset in OLD_KD_DATASETS for dataset in batch["dataset"]],
            dtype=torch.bool,
            device=device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            student_logits, student_features = model_logits_and_features(student, images)
            bce = F.binary_cross_entropy_with_logits(student_logits, labels)
            with torch.no_grad():
                teacher_logits, teacher_features = model_logits_and_features(teacher, images)
            projected = projection(student_features)
        logit_kd = F.huber_loss(
            student_logits[old_mask].float(),
            teacher_logits[old_mask].float().detach(),
            delta=logit_delta,
        )
        if feature_weight > 0:
            feature_kd = 1.0 - F.cosine_similarity(
                projected[old_mask].float(),
                teacher_features[old_mask].float().detach(),
                dim=1,
            ).mean()
        else:
            feature_kd = bce.new_zeros(())
        loss = bce + logit_weight * logit_kd + feature_weight * feature_kd
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite Base v3 loss at batch {batch_index}: "
                f"bce={bce}, logit={logit_kd}, feature={feature_kd}"
            )
        scaler.scale(loss / accumulation_steps).backward()
        processed += 1
        is_last = max_batches is not None and processed >= max_batches
        should_step = processed % accumulation_steps == 0 or is_last
        if should_step:
            trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        batch_size = labels.numel()
        examples += batch_size
        old_examples += int(old_mask.sum())
        totals["loss"] += float(loss.detach()) * batch_size
        totals["bce"] += float(bce.detach()) * batch_size
        totals["logit_kd"] += float(logit_kd.detach()) * batch_size
        totals["feature_kd"] += float(feature_kd.detach()) * batch_size
        progress.set_postfix(
            loss=f"{totals['loss'] / examples:.4f}",
            bce=f"{totals['bce'] / examples:.4f}",
        )
    if processed % accumulation_steps != 0 and not (
        max_batches is not None and processed >= max_batches
    ):
        trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= scale_before:
            scheduler.step()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "loss": totals["loss"] / max(examples, 1),
        "bce": totals["bce"] / max(examples, 1),
        "logit_kd": totals["logit_kd"] / max(examples, 1),
        "feature_kd": totals["feature_kd"] / max(examples, 1),
        "old_fraction": old_examples / max(examples, 1),
        "examples_per_second": examples / elapsed,
    }
