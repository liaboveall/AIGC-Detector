from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import save_checkpoint  # noqa: E402

DEFAULT_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weight-space interpolation between two same-architecture checkpoints: "
        "theta = theta_base + alpha * (theta_modern - theta_base)"
    )
    parser.add_argument(
        "--base",
        default="outputs/multisource_blur_finetune/best.pt",
        help="Base checkpoint (read-only input)",
    )
    parser.add_argument(
        "--modern",
        default="outputs/multisource_modern_finetune_v1/epoch_02.pt",
        help="Modern fine-tune checkpoint (read-only input)",
    )
    parser.add_argument("--output-dir", default="outputs/interpolation_diag")
    parser.add_argument(
        "--alphas",
        default=",".join(f"{a:g}" for a in DEFAULT_ALPHAS),
        help="Comma-separated interpolation coefficients",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_state_dicts(base_state: dict, modern_state: dict) -> None:
    base_keys = set(base_state)
    modern_keys = set(modern_state)
    if base_keys != modern_keys:
        missing = sorted(modern_keys - base_keys)
        extra = sorted(base_keys - modern_keys)
        raise ValueError(f"state_dict key mismatch: missing_from_base={missing}, extra_in_base={extra}")
    for key in sorted(base_keys):
        if base_state[key].shape != modern_state[key].shape:
            raise ValueError(
                f"Shape mismatch for {key!r}: base={tuple(base_state[key].shape)} "
                f"modern={tuple(modern_state[key].shape)}"
            )


def interpolate_state(base_state: dict, modern_state: dict, alpha: float) -> dict:
    interpolated: dict = {}
    for key, base_tensor in base_state.items():
        modern_tensor = modern_state[key]
        if base_tensor.is_floating_point():
            base_f = base_tensor.to(torch.float32)
            modern_f = modern_tensor.to(torch.float32)
            merged = base_f + alpha * (modern_f - base_f)
            interpolated[key] = merged.to(base_tensor.dtype)
        else:
            # Non-float buffers (e.g. num_batches_tracked) are copied verbatim.
            interpolated[key] = base_tensor.clone()
    return interpolated


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    if not alphas:
        raise ValueError("--alphas did not contain any values")
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {alpha}")
    # Collision guard: distinct alphas must produce distinct output filenames.
    tags: dict[str, float] = {}
    for alpha in alphas:
        tag = alpha_tag(alpha)
        if tag in tags:
            raise ValueError(
                f"alpha tag collision: {tags[tag]:g} and {alpha:g} both map to '{tag}.pt'"
            )
        tags[tag] = alpha

    base_path = resolve(args.base)
    modern_path = resolve(args.modern)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_payload = load_checkpoint(base_path)
    modern_payload = load_checkpoint(modern_path)
    for name, payload, path in (("base", base_payload, base_path), ("modern", modern_payload, modern_path)):
        if "model_state" not in payload:
            raise ValueError(f"{name} checkpoint is missing 'model_state': {path}")
        if "config" not in payload:
            raise ValueError(f"{name} checkpoint is missing 'config': {path}")

    base_model = base_payload["config"].get("model", {}).get("name")
    modern_model = modern_payload["config"].get("model", {}).get("name")
    if base_model != modern_model:
        raise ValueError(f"Model name mismatch: base={base_model!r} modern={modern_model!r}")

    base_state = base_payload["model_state"]
    modern_state = modern_payload["model_state"]
    validate_state_dicts(base_state, modern_state)
    print(
        f"validated: {len(base_state)} tensors, model={base_model}, "
        f"base={base_path}, modern={modern_path}"
    )

    base_name = base_payload.get("config", {}).get("model", {}).get("name")
    for alpha in alphas:
        merged_state = interpolate_state(base_state, modern_state, alpha)
        payload = copy.deepcopy(base_payload)
        payload.pop("optimizer_state", None)
        payload.pop("scheduler_state", None)
        payload["model_state"] = merged_state
        payload["interpolation"] = {
            "alpha": alpha,
            "base_checkpoint": str(base_path),
            "modern_checkpoint": str(modern_path),
            "formula": "theta = theta_base + alpha * (theta_modern - theta_base)",
            "base_epoch": base_payload.get("epoch"),
            "modern_epoch": modern_payload.get("epoch"),
        }
        output_path = output_dir / f"{alpha_tag(alpha)}.pt"
        save_checkpoint(output_path, payload)
        print(f"wrote {output_path} alpha={alpha:g} model={base_name}")


if __name__ == "__main__":
    main()
