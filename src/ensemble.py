"""Fixed-weight logit-space ensemble of two independently trained detectors.

Motivation (see ``docs/ENSEMBLE_VNEXT.md``): three independent single-model
attempts to replace the accepted Tiny vNext detector with a trained
ConvNeXt-Base checkpoint (branches ``base-v1``/``base-v2``/``base-v3``) each
failed the preregistered historical-robustness gates versus Tiny vNext. A
zero-training, fixed-weight logit blend of the accepted Tiny vNext checkpoint
and the (individually rejected) Base v1 checkpoint was found, as an offline
diagnostic in ``base-v2`` (``outputs/base_v2/blend_upper_bound/``), to pass
every historical *and* modern gate simultaneously with wide margins. This
module promotes that diagnostic blend to a first-class, checkpoint-describable
model so it can be produced, evaluated, and served through the existing
``predict.py`` / ``evaluate.py`` entry points exactly like any other
checkpoint in this repository.

Design contract:

- ``EnsembleModel`` wraps two already-trained, already-frozen detector
  modules (``model_a`` and ``model_b``) of any architecture describable by
  ``src.adapter.build_checkpoint_model``. It performs two independent forward
  passes and returns ``(1 - alpha) * logit_a + alpha * logit_b``.
- Both sub-models are frozen (``requires_grad = False``) and pinned to eval
  mode: this wrapper composes two checkpoints that were already trained and
  selected independently. It is not designed to be trained further, mirroring
  how ``AdapterModel`` freezes its wrapped base.
- Checkpoint schema (self-contained, one ``.pt`` file, mirrors the
  ``adapter.enabled`` pattern in ``src/adapter.py``)::

      {
          "config": {
              "ensemble": {
                  "enabled": true,
                  "alpha": 0.50,           # weight on model_b in [0, 1]
                  "model_a": {"config": {...sub-checkpoint config...}},
                  "model_b": {"config": {...sub-checkpoint config...}},
              },
              "data": {...},   # copied from model_a so predict.py / evaluate.py
                                # keep working through the shared entry points
              "seed": ..., "device": ...,
          },
          "model_state": {"model_a": {...state_dict...}, "model_b": {...state_dict...}},
      }

  Use ``scripts/package_ensemble_checkpoint.py`` to build this file from two
  existing checkpoints.
"""
from __future__ import annotations

from typing import Any

import torch


DEFAULT_ALPHA = 0.5


class EnsembleModel(torch.nn.Module):
    """Frozen, inference-only fixed-weight logit blend of two detectors.

    ``forward`` returns ``(1 - alpha) * logit_a + alpha * logit_b``, matching
    ``scripts/blend_predictions.py``'s offline logit-space blend exactly (that
    script round-trips through probabilities; blending the pre-sigmoid logits
    directly is algebraically identical and avoids the clipping it needs).
    """

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {alpha}")
        self.model_a = model_a
        self.model_b = model_b
        self.alpha = float(alpha)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "EnsembleModel":
        # Inference-only wrapper: both branches are already-trained, already-
        # selected checkpoints, so they stay frozen/eval regardless of any
        # caller's training-mode toggle (mirrors AdapterModel.train()).
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Promote member logits before arithmetic. Under CUDA autocast the
        # backbones emit float16 values; blending those values in float16 makes
        # the packaged path differ from the direct sweep, which intentionally
        # blends in float32 after both forward passes.
        logits_a = self.model_a(images).flatten().float()
        logits_b = self.model_b(images).flatten().float()
        blended = (1.0 - self.alpha) * logits_a + self.alpha * logits_b
        return blended.unsqueeze(-1)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.6g}, frozen=True"


def ensemble_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("ensemble", {}).get("enabled", False))


def build_ensemble_model(
    ensemble_config: dict[str, Any],
    model_state: dict[str, Any] | None = None,
) -> EnsembleModel:
    """Construct the two sub-models described by an ensemble checkpoint config.

    Sub-models are built through ``build_checkpoint_model`` recursively, so
    either branch may itself be an adapter-enabled model (Tiny vNext) or a
    bare backbone (Base v1) -- any checkpoint this repository can already
    build is a valid ensemble member.
    """
    # Local import: src.adapter.build_checkpoint_model dispatches here for
    # ensemble-enabled configs, so a module-level import would be circular.
    from .adapter import build_checkpoint_model

    for key in ("model_a", "model_b"):
        if key not in ensemble_config:
            raise ValueError(f"ensemble config is missing {key!r}")
    if model_state is not None:
        missing_states = [key for key in ("model_a", "model_b") if key not in model_state]
        if missing_states:
            raise ValueError(f"ensemble model_state is missing {missing_states}")
    state_a = model_state["model_a"] if model_state is not None else None
    state_b = model_state["model_b"] if model_state is not None else None
    model_a = build_checkpoint_model(ensemble_config["model_a"]["config"], state_a)
    model_b = build_checkpoint_model(ensemble_config["model_b"]["config"], state_b)
    alpha = float(ensemble_config.get("alpha", DEFAULT_ALPHA))
    return EnsembleModel(model_a, model_b, alpha=alpha)


def ensemble_parameter_counts(model: EnsembleModel) -> dict[str, int]:
    model_a_total = sum(parameter.numel() for parameter in model.model_a.parameters())
    model_b_total = sum(parameter.numel() for parameter in model.model_b.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "model_a": model_a_total,
        "model_b": model_b_total,
        "trainable": trainable,
        "total": model_a_total + model_b_total,
    }
