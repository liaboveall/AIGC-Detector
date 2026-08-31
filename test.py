from __future__ import annotations

import unittest
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.adapter import (
    AdapterModel,
    MultiScaleAdapterModel,
    adapter_parameter_counts,
    build_checkpoint_model,
)
from src.base_v3 import (
    BaseV3QuotaBatchSampler,
    build_phase_param_groups,
    model_logits_and_features,
    quota_signature,
    set_phase_trainability,
)
from src.data import ManifestImageDataset, RobustnessImageDataset
from src.ensemble import EnsembleModel, build_ensemble_model, ensemble_parameter_counts
from src.model import create_model
from src.repair import repair_loss_components, routing_masks
from src.transforms import RandomDegradation, RandomLabelIndependentReencode, build_eval_transform


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Dataset"


class BaselineTests(unittest.TestCase):
    @staticmethod
    def _fixture_manifest(root: Path) -> Path:
        rows = []
        for index, label in enumerate((0, 1, 0)):
            path = Path("images") / f"sample_{index}.png"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((48, 48, 3), 40 + 80 * label + index, dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(root / path)
            rows.append(
                {
                    "path": path.as_posix(),
                    "dataset": "test-fixture",
                    "source_class": "real" if label == 0 else "fake",
                    "binary_label": label,
                    "allowed_for_training": True,
                }
            )
        manifest = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return manifest

    def test_manifest_policy_and_counts(self) -> None:
        main_path = DATASET_ROOT / "manifests" / "training_main.csv"
        smoke_path = DATASET_ROOT / "manifests" / "training_smoke.csv"
        if not main_path.is_file() or not smoke_path.is_file():
            self.skipTest("local dataset manifests are intentionally not committed")
        main = pd.read_csv(main_path)
        smoke = pd.read_csv(smoke_path)
        self.assertEqual(set(main["dataset"]), {"SID_Set"})
        self.assertTrue(main["allowed_for_training"].all())
        self.assertEqual(len(main), 202820)
        self.assertEqual(
            smoke["source_class"].value_counts().to_dict(),
            {"real": 4000, "full_synthetic": 2000, "tampered": 2000},
        )

    def test_dataset_sample(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture_manifest(root)
            dataset = ManifestImageDataset(
                root,
                manifest,
                build_eval_transform(224),
                training=True,
                max_samples=3,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))
            self.assertIn(float(sample["label"]), (0.0, 1.0))

    def test_model_forward(self) -> None:
        base = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        model = AdapterModel(base, feature_dim=768, hidden_dim=256)
        with torch.inference_mode():
            images = torch.zeros(2, 3, 64, 64)
            base_output = model.base(images)
            output = model(images)
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertTrue(torch.allclose(output, base_output, atol=1e-6, rtol=0.0))
        counts = adapter_parameter_counts(model)
        self.assertEqual(counts["adapter_branch"], 197_121)
        self.assertEqual(counts["trainable"], 197_121)
        self.assertEqual(counts["total"], 28_018_018)

    def test_ensemble_model_forward(self) -> None:
        model_a = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        model_b = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        images = torch.rand(3, 3, 64, 64)
        with torch.inference_mode():
            logits_a = model_a(images).flatten()
            logits_b = model_b(images).flatten()

            zero_alpha = EnsembleModel(model_a, model_b, alpha=0.0)
            output_zero = zero_alpha(images)
            self.assertEqual(tuple(output_zero.shape), (3, 1))
            self.assertTrue(torch.allclose(output_zero.flatten(), logits_a, atol=1e-6))

            one_alpha = EnsembleModel(model_a, model_b, alpha=1.0)
            self.assertTrue(torch.allclose(one_alpha(images).flatten(), logits_b, atol=1e-6))

            mixed = EnsembleModel(model_a, model_b, alpha=0.3)
            expected = 0.7 * logits_a + 0.3 * logits_b
            self.assertTrue(torch.allclose(mixed(images).flatten(), expected, atol=1e-5))

        counts = ensemble_parameter_counts(mixed)
        self.assertEqual(counts["trainable"], 0)
        self.assertEqual(counts["total"], counts["model_a"] + counts["model_b"])
        for parameter in mixed.parameters():
            self.assertFalse(parameter.requires_grad)
        mixed.train(True)
        self.assertFalse(mixed.training)
        self.assertFalse(mixed.model_a.training)
        self.assertFalse(mixed.model_b.training)

        with self.assertRaises(ValueError):
            EnsembleModel(model_a, model_b, alpha=1.5)

    def test_ensemble_blend_promotes_member_logits_to_float32(self) -> None:
        class HalfLogit(torch.nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.register_buffer("value", torch.tensor([[value]], dtype=torch.float16))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.value.expand(images.shape[0], 1)

        model = EnsembleModel(HalfLogit(0.1), HalfLogit(0.2), alpha=0.5)
        output = model(torch.zeros(2, 3, 8, 8))
        expected = 0.5 * model.model_a.value.float() + 0.5 * model.model_b.value.float()
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, expected.expand_as(output)))

    def test_build_checkpoint_model_ensemble(self) -> None:
        base_a = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        base_b = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        config_a = {"model": {"name": "convnext_tiny", "pretrained": False}, "data": {"image_size": 224}}
        config_b = {"model": {"name": "convnext_tiny", "pretrained": False}, "data": {"image_size": 224}}
        checkpoint = {
            "config": {
                "seed": 2026,
                "device": "auto",
                "data": {"image_size": 224},
                "ensemble": {
                    "enabled": True,
                    "alpha": 0.25,
                    "model_a": {"config": config_a},
                    "model_b": {"config": config_b},
                },
            },
            "model_state": {"model_a": base_a.state_dict(), "model_b": base_b.state_dict()},
        }
        model = build_checkpoint_model(checkpoint["config"], checkpoint["model_state"])
        self.assertIsInstance(model, EnsembleModel)
        self.assertAlmostEqual(model.alpha, 0.25)
        images = torch.rand(2, 3, 64, 64)
        with torch.inference_mode():
            expected = 0.75 * base_a(images).flatten() + 0.25 * base_b(images).flatten()
            self.assertTrue(torch.allclose(model(images).flatten(), expected, atol=1e-5))

        with TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "ensemble.pt"
            torch.save(checkpoint, checkpoint_path)
            reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            reloaded = build_ensemble_model(
                reloaded_payload["config"]["ensemble"], reloaded_payload["model_state"]
            )
            with torch.inference_mode():
                self.assertTrue(torch.allclose(reloaded(images).flatten(), expected, atol=1e-5))

        with self.assertRaisesRegex(ValueError, "model_state is missing"):
            build_ensemble_model(checkpoint["config"]["ensemble"], {"model_a": base_a.state_dict()})
    def test_base_v3_quota_sampler_contract(self) -> None:
        rows = []
        for dataset in ("CommunityForensics-Small", "GenImage", "SID_Set"):
            for label in (0, 1):
                for index in range(4):
                    rows.append(
                        {
                            "dataset": dataset,
                            "binary_label": label,
                            "source_class": "real" if label == 0 else "fake",
                            "generator": "",
                        }
                    )
        for index in range(4):
            rows.append(
                {
                    "dataset": "SuSy",
                    "binary_label": 0,
                    "source_class": "real",
                    "generator": "",
                }
            )
        for generator in ("modern_a", "modern_b"):
            for _ in range(2):
                rows.append(
                    {
                        "dataset": "MS-COCOAI",
                        "binary_label": 1,
                        "source_class": generator,
                        "generator": generator,
                    }
                )
        frame = pd.DataFrame(rows)
        sampler = BaseV3QuotaBatchSampler(frame, seed=2026)
        self.assertEqual(len(sampler), 4)
        batches = list(sampler)
        self.assertEqual(len({index for batch in batches for index in batch}), 32)
        for indices in batches:
            batch_frame = frame.iloc[indices]
            batch = {
                "dataset": batch_frame["dataset"].tolist(),
                "label": torch.tensor(batch_frame["binary_label"].tolist()),
            }
            signature = quota_signature(batch)
            self.assertEqual(len(indices), 8)
            self.assertTrue(all(value == 1 for value in signature.values()))
            self.assertEqual(len(signature), 8)

    def test_base_v3_phase_groups_cover_trainable_parameters(self) -> None:
        model = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        projection = torch.nn.Linear(768, 32, bias=False)
        set_phase_trainability(model, ("stages.3.", "head."))
        groups = build_phase_param_groups(
            model,
            projection,
            {"stages.3.": 1e-5, "head.": 1e-4},
            projection_learning_rate=1e-4,
        )
        self.assertEqual({group["name"] for group in groups}, {"stages.3", "head", "feature_projection"})
        assigned = sum(sum(parameter.numel() for parameter in group["params"]) for group in groups)
        expected = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        expected += sum(parameter.numel() for parameter in projection.parameters())
        self.assertEqual(assigned, expected)

    def test_base_v3_logits_and_features_match_bare_model(self) -> None:
        model = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0}).eval()
        images = torch.zeros(2, 3, 64, 64)
        with torch.inference_mode():
            expected = model(images).flatten()
            logits, features = model_logits_and_features(model, images)
        self.assertTrue(torch.equal(logits, expected))
        self.assertEqual(tuple(features.shape), (2, 768))

    def test_adapter_residual_gain(self) -> None:
        base = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        model = AdapterModel(base, feature_dim=768, hidden_dim=256, residual_gain=0.6)
        with torch.no_grad():
            model.adapter.net[-1].bias.fill_(1.0)
        with torch.inference_mode():
            final, base_logits, residual = model.forward_with_residual(torch.zeros(2, 3, 64, 64))
        self.assertTrue(torch.allclose(residual, torch.ones_like(residual)))
        self.assertTrue(torch.allclose(final - base_logits, torch.full_like(final, 0.6), atol=1e-6))

    def test_multiscale_adapter_zero_identity_and_gain(self) -> None:
        base = create_model({"name": "convnext_tiny", "pretrained": False, "drop_path": 0.0})
        model = MultiScaleAdapterModel(
            base,
            stage_dims=(96, 192, 384, 768),
            hidden_dim=64,
            residual_gain=0.6,
        )
        images = torch.zeros(2, 3, 64, 64)
        with torch.inference_mode():
            bare = model.base(images)
            wrapped = model(images)
        self.assertTrue(torch.allclose(wrapped, bare, atol=1e-6, rtol=0.0))
        with torch.no_grad():
            model.adapter.net[-1].bias.fill_(1.0)
        with torch.inference_mode():
            final, base_logits, residual = model.forward_with_residual(images)
        self.assertTrue(torch.allclose(residual, torch.ones_like(residual)))
        self.assertTrue(torch.allclose(final - base_logits, torch.full_like(final, 0.6), atol=1e-6))
        counts = adapter_parameter_counts(model)
        self.assertEqual(counts["trainable"], counts["adapter_branch"])
        self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))

    def test_multiscale_checkpoint_round_trip(self) -> None:
        config = {
            "model": {
                "name": "convnext_tiny",
                "pretrained": False,
                "dropout": 0.0,
                "drop_path": 0.0,
            },
            "adapter": {
                "enabled": True,
                "kind": "multiscale_stats",
                "stage_dims": [96, 192, 384, 768],
                "hidden_dim": 32,
                "residual_gain": 1.25,
                "dropout": 0.0,
            },
        }
        base = create_model(config["model"], pretrained_override=False)
        original = MultiScaleAdapterModel(
            base,
            stage_dims=(96, 192, 384, 768),
            hidden_dim=32,
            residual_gain=1.25,
        ).eval()
        with torch.no_grad():
            original.adapter.net[-1].bias.fill_(0.25)
        restored = build_checkpoint_model(config, original.state_dict()).eval()
        images = torch.zeros(1, 3, 64, 64)
        with torch.inference_mode():
            expected = original(images)
            actual = restored(images)
        self.assertIsInstance(restored, MultiScaleAdapterModel)
        self.assertTrue(torch.equal(actual, expected))

    def test_robustness_dataset(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture_manifest(root)
            transforms = {
                condition: build_eval_transform(64, condition)
                for condition in ("clean", "jpeg_50", "noise_0.02")
            }
            dataset = RobustnessImageDataset(
                root,
                manifest,
                transforms,
                max_samples=3,
            )
            sample = dataset[0]
            self.assertEqual(set(sample["images"]), set(transforms))
            self.assertTrue(
                all(tuple(image.shape) == (3, 64, 64) for image in sample["images"].values())
            )

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

    def test_joint_degradation_exposure_distribution(self) -> None:
        transform = RandomDegradation(
            probability=0.80,
            kind_weights={
                "jpeg": 1.0,
                "blur": 4.0,
                "scale": 1.0,
                "noise": 2.0,
                "color": 0.5,
                "crop": 0.5,
            },
            blur_weights=[1.0, 2.0, 3.0],
        )
        random.seed(2026)
        count = 60_000
        samples = Counter(
            (item.kind, item.value)
            for item in (transform.sample_degradation() for _ in range(count))
        )
        self.assertAlmostEqual(samples[("clean", None)] / count, 0.20, delta=0.01)
        self.assertAlmostEqual(samples[("blur", 2.0)] / count, 0.1778, delta=0.01)
        self.assertAlmostEqual(samples[("noise", 0.10)] / count, 0.0593, delta=0.008)
        self.assertAlmostEqual(samples[("color", -0.20)] / count, 0.0222, delta=0.005)
        self.assertAlmostEqual(samples[("color", 0.20)] / count, 0.0222, delta=0.005)
        self.assertAlmostEqual(samples[("crop", 0.80)] / count, 0.0444, delta=0.007)

    def test_extended_degradations_are_opt_in(self) -> None:
        transform = RandomDegradation(probability=1.0)
        random.seed(2026)
        kinds = {transform.sample_degradation().kind for _ in range(2_000)}
        self.assertEqual(kinds, {"jpeg", "blur", "scale", "noise"})

    def test_label_independent_reencode(self) -> None:
        transform = RandomLabelIndependentReencode(
            probability=1.0,
            qualities=[70],
            codecs=["jpeg"],
        )
        image = Image.fromarray(
            np.random.default_rng(2026).integers(0, 256, size=(32, 32, 3), dtype=np.uint8),
            mode="RGB",
        )
        reencoded = transform(image)
        self.assertEqual(reencoded.mode, "RGB")
        self.assertFalse(np.array_equal(np.asarray(image), np.asarray(reencoded)))

    def test_repair_loss_routes_gradients(self) -> None:
        final = torch.tensor([0.2, -0.4, 0.7, -0.1], requires_grad=True)
        residual = torch.tensor([0.3, -0.2, 0.5, -0.6], requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        repair = torch.tensor([True, True, False, False])
        protect = ~repair
        teacher = torch.tensor([0.8, -1.0])
        components = repair_loss_components(
            final,
            residual,
            labels,
            teacher,
            repair,
            protect,
            bce_weight=0.5,
            distill_weight=1.0,
            protect_weight=3.0,
        )
        components["loss"].backward()
        self.assertGreater(float(final.grad[:2].abs().sum()), 0.0)
        self.assertEqual(float(final.grad[2:].abs().sum()), 0.0)
        self.assertEqual(float(residual.grad[:2].abs().sum()), 0.0)
        self.assertGreater(float(residual.grad[2:].abs().sum()), 0.0)

    def test_repair_routing_rejects_unknown_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "unrouted"):
            routing_masks(
                ["GenImage", "unknown"],
                {"GenImage", "SID_Set"},
                {"CommunityForensics-Small"},
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
