"""Residual adapter wrapper for the frozen base detector (task #10).

Design contract (backwards compatible):
- ``AdapterModel`` wraps a frozen ConvNeXt-Tiny base with a small residual
  MLP branch. The branch consumes the base's pooled+normalised 768-d feature
  (head: global_pool -> norm -> flatten) and adds its logit to the base logit
  with a fixed gain of 1.0.
- The final MLP layer of the branch is zero-initialised (weights AND bias),
  so an untrained wrapper is numerically identical to the bare base.
- Checkpoint schema stays ``{"config": ..., "model_state": ...}``. A
  checkpoint whose config carries ``adapter.enabled: true`` loads as a
  wrapped model; old checkpoints without that key behave exactly as before.
"""
from __future__ import annotations

from typing import Any

import torch

from .model import create_model

DEFAULT_FEATURE_DIM = 768
DEFAULT_HIDDEN_DIM = 256
DEFAULT_RESIDUAL_GAIN = 1.0
# NormMlpClassifierHead components we replicate in forward_with_residual.
HEAD_COMPONENTS = ("global_pool", "norm", "flatten", "pre_logits", "drop", "fc")


class ResidualAdapter(torch.nn.Module):
    """MLP 768 -> hidden -> 1 with a zero-initialised last layer."""

    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AdapterModel(torch.nn.Module):
    """Frozen base + zero-initialised residual branch.

    ``forward`` returns ``base_logit + gain * residual_logit`` so the wrapper
    is a drop-in replacement for the bare base in evaluate.py / predict.py.
    The base is permanently in eval mode (no dropout / drop-path noise on the
    frozen path); only the branch participates in training dynamics.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        residual_gain: float = DEFAULT_RESIDUAL_GAIN,
    ) -> None:
        super().__init__()
        head = getattr(base, "head", None)
        if head is None:
            raise ValueError("Base model has no .head attribute")
        for component in HEAD_COMPONENTS:
            if not hasattr(head, component):
                raise ValueError(f"Base head is missing component {component!r}")
        self.base = base
        self.adapter = ResidualAdapter(feature_dim, hidden_dim)
        self.residual_gain = float(residual_gain)
        # Freeze the base completely and pin it to eval mode.
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

    def train(self, mode: bool = True) -> "AdapterModel":
        super().train(mode)
        self.base.eval()  # frozen path stays deterministic
        return self

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        """head global_pool -> norm -> flatten output (B, feature_dim)."""
        features = self.base.forward_features(images)
        head = self.base.head
        pooled = head.global_pool(features)
        normed = head.norm(pooled)
        return head.flatten(normed)

    def forward_with_residual(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.pooled_features(images)
        head = self.base.head
        base_logit = head.fc(head.drop(head.pre_logits(flat)))
        residual = self.adapter(flat)
        final_logit = base_logit + self.residual_gain * residual
        return final_logit, base_logit, residual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        final_logit, _, _ = self.forward_with_residual(images)
        return final_logit


def adapter_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("adapter", {}).get("enabled", False))


def build_checkpoint_model(
    config: dict[str, Any], model_state: dict[str, torch.Tensor] | None = None
) -> torch.nn.Module:
    """Construct the model described by a checkpoint config.

    Old checkpoints (no ``adapter`` key) follow the original
    ``create_model`` path unchanged.
    """
    base = create_model(config["model"], pretrained_override=False)
    if adapter_enabled(config):
        adapter_config = config["adapter"]
        model: torch.nn.Module = AdapterModel(
            base,
            feature_dim=int(adapter_config.get("feature_dim", DEFAULT_FEATURE_DIM)),
            hidden_dim=int(adapter_config.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
            residual_gain=float(adapter_config.get("residual_gain", DEFAULT_RESIDUAL_GAIN)),
        )
    else:
        model = base
    if model_state is not None:
        model.load_state_dict(model_state)
    return model


def adapter_parameter_counts(model: AdapterModel) -> dict[str, int]:
    branch = sum(p.numel() for p in model.adapter.parameters())
    base = sum(p.numel() for p in model.base.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "base_frozen": base,
        "adapter_branch": branch,
        "trainable": trainable,
        "total": base + branch,
    }


@torch.no_grad()
def assert_zero_residual_identity(
    model: AdapterModel,
    device: torch.device | str = "cpu",
    batch_size: int = 4,
    image_size: int = 224,
    tolerance: float = 1e-6,
) -> float:
    """Verify an untrained branch leaves the base output untouched.

    Raises AssertionError when wrapped and bare logits differ beyond the
    tolerance. Returns the observed maximum absolute difference.
    """
    model.eval()
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(2026)
    images = torch.rand(
        batch_size, 3, image_size, image_size, generator=generator, device=device
    )
    wrapped = model(images)
    bare = model.base(images)
    max_diff = float((wrapped - bare).abs().max().item())
    if max_diff > tolerance:
        raise AssertionError(
            f"Zero-init contract violated: wrapped vs bare base logit diff {max_diff} > {tolerance}"
        )
    return max_diff
