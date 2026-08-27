from __future__ import annotations

from typing import Any

import timm
import torch


def create_model(model_config: dict[str, Any], pretrained_override: bool | None = None) -> torch.nn.Module:
    pretrained = bool(model_config.get("pretrained", True))
    if pretrained_override is not None:
        pretrained = pretrained_override
    return timm.create_model(
        model_config.get("name", "convnext_tiny"),
        pretrained=pretrained,
        num_classes=1,
        drop_rate=float(model_config.get("dropout", 0.0)),
        drop_path_rate=float(model_config.get("drop_path", 0.1)),
    )


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
