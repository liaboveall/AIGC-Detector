from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import project_path
from src.model import create_model
from src.transforms import build_eval_transform
from src.utils import get_device, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class DirectoryDataset(Dataset[dict[str, Any]]):
    def __init__(self, input_dir: Path, image_size: int) -> None:
        self.input_dir = input_dir
        self.paths = sorted(
            path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = build_eval_transform(image_size, "clean")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        valid = True
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception:
            tensor = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            valid = False
        return {
            "image": tensor,
            "image_path": path.relative_to(self.input_dir).as_posix(),
            "valid": valid,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AI-image probabilities for a directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = get_device(config.get("device", "auto"))
    model = create_model(config["model"], pretrained_override=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    input_dir = project_path(args.input_dir)
    dataset = DirectoryDataset(input_dir, int(config["data"]["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    predictions = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(model(images).flatten()).float().cpu().tolist()
        for image_path, probability, valid in zip(batch["image_path"], probabilities, batch["valid"]):
            value = float(probability) if bool(valid) else 0.5
            predictions.append({"image_path": image_path, "pred": min(max(value, 0.0), 1.0)})
    write_json(project_path(args.output), predictions)
    print(f"wrote {len(predictions)} predictions to {project_path(args.output)}")


if __name__ == "__main__":
    main()
