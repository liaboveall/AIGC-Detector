from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REQUIRED_COLUMNS = {
    "path",
    "dataset",
    "source_class",
    "binary_label",
    "allowed_for_training",
}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
    frame["binary_label"] = frame["binary_label"].astype(int)
    invalid_labels = set(frame["binary_label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Invalid binary labels in {manifest_path}: {invalid_labels}")
    return frame


def stratified_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(frame) <= limit:
        return frame.reset_index(drop=True)
    groups = list(frame.groupby("source_class", sort=True))
    per_group = limit // len(groups)
    samples: list[pd.DataFrame] = []
    selected_indices: set[int] = set()
    for offset, (_, group) in enumerate(groups):
        count = min(per_group, len(group))
        sample = group.sample(n=count, random_state=seed + offset)
        samples.append(sample)
        selected_indices.update(sample.index.tolist())
    remaining = limit - sum(len(sample) for sample in samples)
    if remaining > 0:
        pool = frame.loc[~frame.index.isin(selected_indices)]
        samples.append(pool.sample(n=min(remaining, len(pool)), random_state=seed + 1000))
    result = pd.concat(samples, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        training: bool = False,
        max_samples: int | None = None,
        seed: int = 2026,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.frame = stratified_sample(read_manifest(self.manifest_path), max_samples, seed)
        self.transform = transform
        self.training = training
        if training:
            forbidden = ~self.frame["allowed_for_training"].map(_truthy)
            if forbidden.any():
                examples = self.frame.loc[forbidden, "path"].head(3).tolist()
                raise ValueError(f"Training manifest contains forbidden rows: {examples}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.dataset_root / Path(str(row["path"]))
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return {
            "image": tensor,
            "label": torch.tensor(float(row["binary_label"]), dtype=torch.float32),
            "path": str(row["path"]),
            "source_class": str(row["source_class"]),
            "dataset": str(row["dataset"]),
        }


class RobustnessImageDataset(Dataset[dict[str, Any]]):
    """Decode each validation image once and create all degradation views."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        transforms: dict[str, Callable[[Image.Image], torch.Tensor]],
        max_samples: int | None = None,
        seed: int = 2026,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.frame = stratified_sample(read_manifest(self.manifest_path), max_samples, seed)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.dataset_root / Path(str(row["path"]))
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                views = {
                    condition: transform(image.copy())
                    for condition, transform in self.transforms.items()
                }
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return {
            "images": views,
            "label": torch.tensor(float(row["binary_label"]), dtype=torch.float32),
            "path": str(row["path"]),
            "source_class": str(row["source_class"]),
            "dataset": str(row["dataset"]),
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    training: bool,
    balanced_sampling: bool,
    seed: int,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = training
    if training and balanced_sampling:
        counts = dataset.frame["binary_label"].value_counts().to_dict()
        weights = dataset.frame["binary_label"].map(lambda label: 1.0 / counts[int(label)])
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights.to_numpy(copy=True), dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    keep_workers = training if persistent_workers is None else persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=keep_workers and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=training,
    )
