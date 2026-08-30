"""Paired latency and parameter-count benchmark for two detector checkpoints."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapter import build_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--bursts", type=int, default=20)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    parameters = sum(tensor.numel() for tensor in state.values())
    model = build_checkpoint_model(payload["config"], state).to(device).eval()
    return model, parameters


def timed_burst(
    model: torch.nn.Module, images: torch.Tensor, bursts: int, device: torch.device
) -> float:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
        for _ in range(bursts):
            model(images)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / bursts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this release latency benchmark")
    device = torch.device("cuda")
    baseline_path = (PROJECT_ROOT / args.baseline).resolve()
    candidate_path = (PROJECT_ROOT / args.candidate).resolve()
    baseline, baseline_parameters = load_model(baseline_path, device)
    candidate, candidate_parameters = load_model(candidate_path, device)
    generator = torch.Generator(device=device).manual_seed(2026)
    images = torch.rand(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        generator=generator,
        device=device,
    )

    for _ in range(args.warmup_rounds):
        timed_burst(baseline, images, 1, device)
        timed_burst(candidate, images, 1, device)

    timings: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for round_index in range(args.rounds):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if round_index % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        for name, model in order:
            timings[name].append(timed_burst(model, images, args.bursts, device))

    baseline_ms = statistics.median(timings["baseline"])
    candidate_ms = statistics.median(timings["candidate"])
    result = {
        "device": torch.cuda.get_device_name(device),
        "baseline_checkpoint": str(baseline_path),
        "candidate_checkpoint": str(candidate_path),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup_rounds": args.warmup_rounds,
        "rounds": args.rounds,
        "bursts_per_round": args.bursts,
        "protocol": "interleaved alternating-order paired timing; median round milliseconds per forward",
        "latency_ms": {"baseline": baseline_ms, "candidate": candidate_ms},
        "relative_overhead": candidate_ms / baseline_ms - 1.0,
        "parameter_count": {
            "baseline": baseline_parameters,
            "candidate": candidate_parameters,
        },
        "gates": {
            "batch32_overhead_at_most_10pct": (
                args.batch_size != 32 or candidate_ms / baseline_ms - 1.0 <= 0.10
            ),
            "candidate_below_30m_parameters": candidate_parameters < 30_000_000,
        },
    }
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
