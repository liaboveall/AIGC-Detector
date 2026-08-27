# Dataset layout

This directory was organized on 2026-08-26. Original downloads are preserved in
`_archives`; extracted files are not renamed or deleted.

## Directories

- `_archives/SID_Set`: seven original SID_Set ZIP archives.
- `_archives/WildFake_demo`: original COCO/DALL-E ZIP files and source CSV files.
- `SID_Set/{train,validation,test}`: official split layout expected by SIDA.
- `CIFAKE/{train,test}/{REAL,FAKE}`: the original CIFAKE split layout.
- `WildFake_demo`: only COCO val2017 and DALL-E Advanced, isolated from training.
- `manifests`: official, de-duplicated, and combined CSV manifests.
- `audit`: archive inventory, extraction counts, duplicate details, and rebuild script.

## Manifest policy

- `sid_official_*.csv` preserves every official SID_Set image entry.
- `sid_clean_*.csv` keeps the split priority `test > validation > train` and retains
  one canonical copy for each byte-identical SHA-256 group.
- `training_pool.csv` contains SID clean train plus CIFAKE train.
- `validation_pool.csv` contains SID clean validation.
- `evaluation_pool.csv` contains SID clean test.
- `wildfake_demo.csv` is marked `demo_only` and every row has
  `allowed_for_training=false`.

SID labels are preserved as `real=0`, `full_synthetic=1`, `tampered=2`. The derived
binary label is `real=0` and both generated/tampered classes are `1`. Tampered masks
are referenced through `mask_path` and are not treated as classifier input images.

## Verified counts

- SID_Set official inputs: train 210,000; validation 30,000; test 60,000.
- SID_Set clean inputs: train 202,820; validation 29,272; test 59,853.
- CIFAKE: train 100,000; test 20,000.
- WildFake demo: COCO val2017 4,998; DALL-E Advanced 8,843.
- Combined training pool: 302,820.

All nine ZIP archives passed a complete `7z t` integrity check before extraction.
See `audit/extraction_counts.json` and `audit/duplicate_groups.csv` for details.

## Rebuild

Run the following from the repository root after the extracted directory layout is
present:

```powershell
python Dataset/audit/build_dataset_manifests.py
```

The script rebuilds manifests and audit tables without changing image files.

## Repository tracking policy

Only this README and `audit/build_dataset_manifests.py` are committed. Every manifest
CSV under `manifests/` exceeds 1 MB and is regenerated deterministically (fixed seeds)
from the extracted datasets; see the main README's *Reproduction* section for the
full command chain.
