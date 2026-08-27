from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MANIFEST_ROOT = DATASET_ROOT / "manifests"
SEED = 2026


def sample_classes(frame: pd.DataFrame, targets: dict[str, int], seed: int) -> pd.DataFrame:
    parts = []
    for offset, (source_class, count) in enumerate(targets.items()):
        group = frame.loc[frame["source_class"] == source_class]
        if len(group) < count:
            raise ValueError(f"Need {count} {source_class} rows, only found {len(group)}")
        parts.append(group.sample(n=count, random_state=seed + offset))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def counts(frame: pd.DataFrame) -> dict:
    return {
        "total": int(len(frame)),
        "by_dataset": {str(k): int(v) for k, v in frame["dataset"].value_counts().items()},
        "by_source_class": {str(k): int(v) for k, v in frame["source_class"].value_counts().items()},
        "by_binary_label": {str(k): int(v) for k, v in frame["binary_label"].value_counts().items()},
    }


def main() -> None:
    training_pool = pd.read_csv(MANIFEST_ROOT / "training_pool.csv", keep_default_na=False)
    validation_pool = pd.read_csv(MANIFEST_ROOT / "validation_pool.csv", keep_default_na=False)

    allowed = training_pool["allowed_for_training"].astype(str).str.lower().eq("true")
    main_train = training_pool.loc[allowed & training_pool["dataset"].eq("SID_Set")].copy()
    main_train = main_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    smoke_train = sample_classes(
        main_train,
        {"real": 4000, "full_synthetic": 2000, "tampered": 2000},
        SEED,
    )
    smoke_validation = sample_classes(
        validation_pool,
        {"real": 800, "full_synthetic": 400, "tampered": 400},
        SEED + 100,
    )

    outputs = {
        "training_main.csv": main_train,
        "training_smoke.csv": smoke_train,
        "validation_smoke.csv": smoke_validation,
    }
    for filename, frame in outputs.items():
        frame.to_csv(MANIFEST_ROOT / filename, index=False)

    report = {
        "seed": SEED,
        "policy": "Main training excludes CIFAKE; CIFAKE is reserved for ablation.",
        "manifests": {name: counts(frame) for name, frame in outputs.items()},
    }
    report_path = DATASET_ROOT / "audit" / "training_manifest_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
