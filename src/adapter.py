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
- ``adapter.kind: multiscale_stats`` selects a forensic repair branch that
  consumes mean and standard-deviation statistics from all four ConvNeXt
  stages.  Omitting ``kind`` keeps the original pooled-feature MLP exactly.
"""
from __future__ import annotations

from typing import Any

import torch

from .model import create_model

DEFAULT_FEATURE_DIM = 768
DEFAULT_HIDDEN_DIM = 256
DEFAULT_RESIDUAL_GAIN = 1.0
DEFAULT_ADAPTER_KIND = "pooled_mlp"
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


class MultiScaleStatsAdapter(torch.nn.Module):
    """Texture-aware branch over per-stage channel mean and standard deviation.

    Pooling both first- and second-order statistics retains low-level codec and
    texture evidence that can be discarded by the final semantic pooling
    layer.  The final projection is zero-initialised, preserving the exact
    frozen-Base prediction before fitting.
    """

    def __init__(
        self,
        stage_dims: tuple[int, ...],
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not stage_dims or any(dimension <= 0 for dimension in stage_dims):
            raise ValueError("stage_dims must contain positive channel counts")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        self.stage_dims = tuple(int(dimension) for dimension in stage_dims)
        feature_dim = 2 * sum(self.stage_dims)
        self.norm = torch.nn.LayerNorm(feature_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def pooled_statistics(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != len(self.stage_dims):
            raise ValueError(
                f"received {len(feature_maps)} feature maps; expected {len(self.stage_dims)}"
            )
        statistics = []
        for index, (features, expected_channels) in enumerate(
            zip(feature_maps, self.stage_dims, strict=True)
        ):
            if features.ndim != 4 or features.shape[1] != expected_channels:
                raise ValueError(
                    f"stage {index} shape {tuple(features.shape)} does not match "
                    f"NCHW with {expected_channels} channels"
                )
            # Accumulate moments in fp32 even under autocast. Codec and noise
            # signatures can be small, and half-precision variance would erase
            # part of the signal this branch is intended to recover.
            moments = features.float()
            variance, mean = torch.var_mean(moments, dim=(-2, -1), correction=0)
            statistics.extend((mean, torch.sqrt(variance.clamp_min(1e-6))))
        return self.norm(torch.cat(statistics, dim=1))

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        return self.net(self.pooled_statistics(feature_maps))


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


class MultiScaleAdapterModel(torch.nn.Module):
    """Frozen ConvNeXt + a zero-init multi-scale forensic residual branch."""

    def __init__(
        self,
        base: torch.nn.Module,
        stage_dims: tuple[int, ...],
        hidden_dim: int = 512,
        residual_gain: float = DEFAULT_RESIDUAL_GAIN,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for component in ("stem", "stages", "norm_pre", "forward_head", "head"):
            if not hasattr(base, component):
                raise ValueError(f"Base model is missing ConvNeXt component {component!r}")
        if len(base.stages) != len(stage_dims):
            raise ValueError(
                f"Base exposes {len(base.stages)} stages but stage_dims has {len(stage_dims)}"
            )
        self.base = base
        self.adapter = MultiScaleStatsAdapter(stage_dims, hidden_dim, dropout)
        self.residual_gain = float(residual_gain)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

    def train(self, mode: bool = True) -> "MultiScaleAdapterModel":
        super().train(mode)
        self.base.eval()
        return self

    def forward_feature_maps(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.base.stem(images)
        stage_maps = []
        for stage in self.base.stages:
            features = stage(features)
            stage_maps.append(features)
        return self.base.norm_pre(features), stage_maps

    def forward_with_residual(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_features, stage_maps = self.forward_feature_maps(images)
        base_logit = self.base.forward_head(final_features)
        residual = self.adapter(stage_maps)
        final_logit = base_logit + self.residual_gain * residual
        return final_logit, base_logit, residual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        final_logit, _, _ = self.forward_with_residual(images)
        return final_logit


def adapter_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("adapter", {}).get("enabled", False))


def build_adapter_model(
    base: torch.nn.Module,
    adapter_config: dict[str, Any],
) -> AdapterModel | MultiScaleAdapterModel:
    """Construct the configured frozen-base adapter wrapper."""
    kind = str(adapter_config.get("kind", DEFAULT_ADAPTER_KIND))
    if kind == DEFAULT_ADAPTER_KIND:
        return AdapterModel(
            base,
            feature_dim=int(adapter_config.get("feature_dim", DEFAULT_FEATURE_DIM)),
            hidden_dim=int(adapter_config.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
            residual_gain=float(adapter_config.get("residual_gain", DEFAULT_RESIDUAL_GAIN)),
        )
    if kind == "multiscale_stats":
        stage_dims = tuple(int(value) for value in adapter_config["stage_dims"])
        return MultiScaleAdapterModel(
            base,
            stage_dims=stage_dims,
            hidden_dim=int(adapter_config.get("hidden_dim", 512)),
            residual_gain=float(adapter_config.get("residual_gain", DEFAULT_RESIDUAL_GAIN)),
            dropout=float(adapter_config.get("dropout", 0.0)),
        )
    raise ValueError(f"unsupported adapter kind: {kind!r}")


def build_checkpoint_model(
    config: dict[str, Any], model_state: dict[str, Any] | None = None
) -> torch.nn.Module:
    """Construct the model described by a checkpoint config.

    Old checkpoints (no ``adapter`` key) follow the original
    ``create_model`` path unchanged. Checkpoints with ``ensemble.enabled``
    build a frozen fixed-weight logit blend of two sub-checkpoints instead
    (see ``src/ensemble.py``); this is checked first because an ensemble
    checkpoint's top-level config has no single ``model`` entry of its own.
    """
    if config.get("ensemble", {}).get("enabled", False):
        from .ensemble import build_ensemble_model

        return build_ensemble_model(config["ensemble"], model_state)
    base = create_model(config["model"], pretrained_override=False)
    if adapter_enabled(config):
        model: torch.nn.Module = build_adapter_model(base, config["adapter"])
    else:
        model = base
    if model_state is not None:
        model.load_state_dict(model_state)
    return model


def adapter_parameter_counts(
    model: AdapterModel | MultiScaleAdapterModel,
) -> dict[str, int]:
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
    model: AdapterModel | MultiScaleAdapterModel,
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
