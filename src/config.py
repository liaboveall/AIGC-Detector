from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_paths(config: dict[str, Any], manifest_name: str) -> tuple[Path, Path]:
    dataset_root = project_path(config["data"]["dataset_root"])
    manifest = Path(manifest_name)
    if not manifest.is_absolute():
        manifest = dataset_root / "manifests" / manifest
    return dataset_root, manifest
