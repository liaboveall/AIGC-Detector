"""Loss helpers for a frozen-Base residual repair adapter.

The adapter has two disjoint responsibilities:

* on historical samples, correct the frozen Base logit toward the accepted
  Tiny teacher while retaining supervised label pressure;
* on modern samples, stay near zero so the strong frozen Base prediction is
  preserved exactly.

Only the adapter branch is trainable.  This makes preservation structural
rather than relying on a full-model regularizer.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def repair_loss_components(
    final_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_repair_logits: torch.Tensor,
    repair_mask: torch.Tensor,
    protect_mask: torch.Tensor,
    *,
    bce_weight: float,
    distill_weight: float,
    protect_weight: float,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return the source-routed repair objective and its components.

    ``teacher_repair_logits`` contains only the rows selected by
    ``repair_mask``. Smooth-L1 logit matching preserves teacher margins more
    directly than soft-label cross entropy, while remaining stable for large
    teacher logits.
    """
    final_logits = final_logits.flatten()
    residual_logits = residual_logits.flatten()
    labels = labels.flatten()
    repair_mask = repair_mask.bool().flatten()
    protect_mask = protect_mask.bool().flatten()
    teacher_repair_logits = teacher_repair_logits.flatten()

    expected = final_logits.shape
    if residual_logits.shape != expected or labels.shape != expected:
        raise ValueError("final, residual, and label tensors must have identical shapes")
    if repair_mask.shape != expected or protect_mask.shape != expected:
        raise ValueError("routing masks must match the logit shape")
    if bool((repair_mask & protect_mask).any()):
        raise ValueError("repair and protection masks overlap")
    if bool((~(repair_mask | protect_mask)).any()):
        raise ValueError("every row must be routed to repair or protection")
    repair_count = int(repair_mask.sum().item())
    if teacher_repair_logits.numel() != repair_count:
        raise ValueError(
            f"teacher logits contain {teacher_repair_logits.numel()} rows; expected {repair_count}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    for name, value in {
        "bce_weight": bce_weight,
        "distill_weight": distill_weight,
        "protect_weight": protect_weight,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    zero = final_logits.new_zeros(())
    if repair_count:
        repair_final = final_logits[repair_mask].float()
        teacher = teacher_repair_logits.detach().float()
        bce = F.binary_cross_entropy_with_logits(repair_final, labels[repair_mask].float())
        distill = F.smooth_l1_loss(repair_final / temperature, teacher / temperature)
        distill = distill * (temperature**2)
        teacher_mae = (repair_final.detach() - teacher).abs().mean()
    else:
        bce = zero
        distill = zero
        teacher_mae = zero

    if bool(protect_mask.any()):
        protect = residual_logits[protect_mask].float().pow(2).mean()
    else:
        protect = zero

    loss = bce_weight * bce + distill_weight * distill + protect_weight * protect
    return {
        "loss": loss,
        "bce": bce,
        "distill": distill,
        "protect": protect,
        "teacher_mae": teacher_mae,
    }


def routing_masks(
    datasets: list[str],
    repair_sources: set[str],
    protect_sources: set[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build validated, exhaustive source-routing masks."""
    overlap = repair_sources & protect_sources
    if overlap:
        raise ValueError(f"repair and protection source lists overlap: {sorted(overlap)}")
    known = repair_sources | protect_sources
    unknown = sorted({name for name in datasets if name not in known})
    if unknown:
        raise ValueError(f"training batch contains unrouted sources: {unknown}")
    repair = torch.tensor([name in repair_sources for name in datasets], device=device)
    protect = torch.tensor([name in protect_sources for name in datasets], device=device)
    return repair.bool(), protect.bool()


def detached_metrics(components: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert scalar loss components into loggable floats."""
    return {name: float(value.detach()) for name, value in components.items()}
