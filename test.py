from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import ManifestImageDataset, RobustnessImageDataset
from src.model import create_model
from src.transforms import RandomDegradation, build_eval_transform


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Dataset"


class BaselineTests(unittest.TestCase):
    def test_manifest_policy_and_counts(self) -> None:
        main = pd.read_csv(DATASET_ROOT / "manifests" / "training_main.csv")
        smoke = pd.read_csv(DATASET_ROOT / "manifests" / "training_smoke.csv")
        self.assertEqual(set(main["dataset"]), {"SID_Set"})
        self.assertTrue(main["allowed_for_training"].all())
        self.assertEqual(len(main), 202820)
        self.assertEqual(
            smoke["source_class"].value_counts().to_dict(),
            {"real": 4000, "full_synthetic": 2000, "tampered": 2000},
        )

    def test_dataset_sample(self) -> None:
        dataset = ManifestImageDataset(
            DATASET_ROOT,
            DATASET_ROOT / "manifests" / "training_smoke.csv",
            build_eval_transform(224),
            training=True,
            max_samples=3,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))
        self.assertIn(float(sample["label"]), (0.0, 1.0))

    def test_model_forward(self) -> None:
        model = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_robustness_dataset(self) -> None:
        transforms = {
            condition: build_eval_transform(64, condition)
            for condition in ("clean", "jpeg_50", "noise_0.02")
        }
        dataset = RobustnessImageDataset(
            DATASET_ROOT,
            DATASET_ROOT / "manifests" / "validation_smoke.csv",
            transforms,
            max_samples=3,
        )
        sample = dataset[0]
        self.assertEqual(set(sample["images"]), set(transforms))
        self.assertTrue(all(tuple(image.shape) == (3, 64, 64) for image in sample["images"].values()))

    def test_weighted_strong_blur_augmentation(self) -> None:
        transform = RandomDegradation(
            probability=1.0,
            kind_weights={"jpeg": 0.0, "blur": 1.0, "scale": 0.0, "noise": 0.0},
            blur_weights=[0.0, 0.0, 1.0],
        )
        image = Image.fromarray(
            ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8),
            mode="L",
        ).convert("RGB")
        degraded = transform(image)
        self.assertFalse(np.array_equal(np.asarray(image), np.asarray(degraded)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
