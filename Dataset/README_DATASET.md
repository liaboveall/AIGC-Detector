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
- `SuSy`: pinned train/validation archives plus audited extraction for Tiny vNext.
- `MS_COCOAI`: pinned Defactify parquet shards plus audited image export for Tiny vNext.
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
- `susy_vnext_{train,dev}.csv` and `cocoai_vnext_{train,dev}.csv` are exact-hash and
  content-group audited modern-source manifests.
- `tiny_vnext_train_balanced_280000.csv` is the source/label-balanced training manifest
  embedded in both Ensemble vNext member configs: CommunityForensics 40%, GenImage
  20%, SID_Set 20%, and modern SuSy/MS-COCOAI 20%.
- `tiny_vnext_modern_dev.csv` contains 12,896 source-disjoint development images across
  eight fake-generator strata. It is development/model-selection data, not a sealed
  confirmation set.

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
- SuSy prepared train/validation: 14,451 / 5,555 images from the pinned train/val
  archives; exact train/validation SHA overlap is zero.
- MS-COCOAI filtered train/validation: 34,969 / 7,341 fake images after duplicate and
  prompt-group exclusions.
- Tiny vNext training manifest: 280,000 rows; every historical source is label-balanced,
  and the 56,000-row modern allocation contains 28,000 real and 28,000 fake rows.
- Tiny vNext modern development manifest: 12,896 rows (1,234 shared real references and
  11,662 fake images across eight generator strata).

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

Tiny vNext modern data uses separately pinned download, preparation, and combination
steps. The SuSy sealed test archive is not downloaded by default and must not be opened
for development:

```powershell
python scripts/download_susy_vnext.py --splits train val
python scripts/prepare_susy_vnext.py --splits train val
python scripts/download_cocoai_vnext.py
python scripts/prepare_cocoai_vnext.py
python scripts/build_tiny_vnext_manifests.py --total-train 280000 --seed 2026
```

Pinned sources:

- [SuSy](https://huggingface.co/datasets/aminasifar1/SuSy-Dataset):
  `aminasifar1/SuSy-Dataset` at
  `df5f324e4438cddaaf0de87f231c356b47aa555d`.
- [MS-COCOAI/Defactify](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset):
  `Rajarshi-Roy-research/Defactify_Image_Dataset` at
  `787334f7857fa54f29027a7f09c30e895ad486ef`.
- [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
  at `6c539a534c07917307c381f5af4053c6091b5278`.
- [GenImage Arrow export](https://huggingface.co/datasets/nebula/GenImage-arrow)
  at `3f4b9f921a673be09a93b335ed728cea0c6ecf33`; review the
  [GenImage license](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
  before download or use.

Download receipts and preparation summaries live inside each local dataset directory.
MS-COCOAI's dataset card did not declare a license at the pinned revision; it was
included by explicit project decision and must be reviewed before redistribution or
commercial use.

The pinned CommunityForensics source revision, CC-BY-NC-SA-4.0 license reference,
exclusion accounting, and leakage checks are recorded in
`audit/communityforensics_small_download_report.json` and
`audit/modern_generator_manifest_report.json`.

The historical Adapter v2 replay manifest contains 560,000 rows: 280,000
CommunityForensics-Small, 140,000 GenImage, and 140,000 SID_Set, with each source
balanced 1:1 by binary label. Aggregate non-sensitive facts are tracked publicly in
`../reports/final_adapter_v2/dataset_summary.json`; large manifests and image bodies
remain local.

## Repository tracking policy

Only documentation and deterministic acquisition/preparation/build scripts are
committed. Large manifest CSVs and image bodies remain local; see the main README's
*Verification and reproduction* section for the command chain.
