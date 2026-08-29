# Dataset layout

This directory was organized on 2026-08-26. The original SID/WildFake archives were
removed after verified extraction to reclaim disk space; extracted image trees are
preserved.

## Directories

- `SID_Set/{train,validation,test}`: official split layout expected by SIDA.
- `CIFAKE/{train,test}/{REAL,FAKE}`: the original CIFAKE split layout.
- `GenImage_subset`: exported GenImage training subset and GLIDE holdout.
- `CommunityForensics_Small/images`: full usable CommunityForensics-Small export;
  temporary Parquet shards are removed after verified extraction.
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
- `communityforensics_train_balanced.csv` contains a format-balanced 293,792-image
  modern-generator fine-tuning set (146,896 real / 146,896 fake).
- `communityforensics_selection_6000.csv` and
  `communityforensics_confirmation_6000.csv` use disjoint generator identities;
  the combined 16,000-image confirmation set was opened once for a model-soup
  candidate, then permanently marked consumed after that candidate was rejected. It
  was not reused for the formal Adapter v2 checkpoint.
- `training_modern_generators.csv` is retained as a full-retraining option, but is not
  the default fine-tuning input because the historical sources retain a strong
  label/codec association.

SID labels are preserved as `real=0`, `full_synthetic=1`, `tampered=2`. The derived
binary label is `real=0` and both generated/tampered classes are `1`. Tampered masks
are referenced through `mask_path` and are not treated as classifier input images.

## Verified counts

- SID_Set official inputs: train 210,000; validation 30,000; test 60,000.
- SID_Set clean inputs: train 202,820; validation 29,272; test 59,853.
- CIFAKE: train 100,000; test 20,000.
- WildFake demo: COCO val2017 4,998; DALL-E Advanced 8,843.
- Combined training pool: 302,820.
- CommunityForensics-Small usable export: 553,531 images (277,969 real / 275,562
  fake), 4,780 fake generator identities, 0 missing files, and 0 duplicate SHA-256
  values.
- CommunityForensics generator split: 3,822 train / 479 selection / 479 confirmation.

All nine ZIP archives passed a complete `7z t` integrity check before extraction.
See `audit/extraction_counts.json` and `audit/duplicate_groups.csv` for details.

## Rebuild

Run the following from the repository root after the extracted directory layout is
present:

```powershell
python Dataset/audit/build_dataset_manifests.py
```

The script rebuilds manifests and audit tables without changing image files.

Modern-generator manifests are rebuilt deterministically with:

```powershell
python scripts/build_modern_generator_manifests.py
```

The pinned CommunityForensics source revision, CC-BY-NC-SA-4.0 license reference,
exclusion accounting, and leakage checks are recorded in
`audit/communityforensics_small_download_report.json` and
`audit/modern_generator_manifest_report.json`.

The final Adapter v2 replay manifest contains 560,000 rows: 280,000
CommunityForensics-Small, 140,000 GenImage, and 140,000 SID_Set, with each source
balanced 1:1 by binary label. Aggregate non-sensitive facts are tracked publicly in
`../reports/final_adapter_v2/dataset_summary.json`; large manifests and image bodies
remain local.

## Repository tracking policy

Only this README and `audit/build_dataset_manifests.py` are committed. Every manifest
CSV under `manifests/` exceeds 1 MB and is regenerated deterministically (fixed seeds)
from the extracted datasets; see the main README's *Reproduction* section for the
full command chain.
