# Rebuilding the datasets

Image bodies and manifest CSVs are not stored in Git. This guide recreates `Dataset/`
from upstream sources in the order the build scripts require. None of it is needed to
run the released checkpoint: `predict.py` works without any dataset.

Run every command from the repository root inside the pinned `requirements.txt`
environment.

## What you need for what

| Goal | Data required | Download | Free disk |
|---|---|---:|---:|
| Run the released model | none | 0 | 0 |
| Re-check the modern development results | SuSy validation + MS-COCOAI | ~8 GB | ~16 GB |
| Re-check the historical anchor, or retrain | all seven sources | ~560 GB | ≥ 550 GB (1 TB is comfortable) |

The modern development set (`tiny_vnext_modern_dev.csv`) depends only on SuSy and
MS-COCOAI, so it can be rebuilt on its own; see
[Modern development set only](#modern-development-set-only). Every other manifest
depends on the full chain below.

## Sources

| Dataset | Source | Revision | Size | Access |
|---|---|---|---|---|
| SID_Set | SIDA authors' Google Drive: [train/validation folder](https://drive.google.com/drive/folders/1sFZxSrDibjpvzTrHeNVS1fTf-qIue744), [test.zip](https://drive.google.com/file/d/1_ivsEV5e14efuv93tJgXjWOondYnEC2G/view) | not pinned | 130.3 GB + test.zip | browser; large files may need a Google sign-in |
| CIFAKE | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | not pinned | ~0.1 GB | Kaggle account |
| WildFake | [ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake): `Images/Real/coco.zip`, `Images/Diffusion_based/DALLE.zip` | not pinned | 2.2 + 23.8 GiB | may need a ModelScope sign-in |
| GenImage | [`nebula/GenImage-arrow`](https://huggingface.co/datasets/nebula/GenImage-arrow) | `3f4b9f921a673be09a93b335ed728cea0c6ecf33` | 610 GiB repository; the script fetches ~85 GiB | accept the [GenImage license](https://github.com/GenImage-Dataset/GenImage/blob/main/License) |
| CommunityForensics-Small | [`OwensLab/CommunityForensics-Small`](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) | `6c539a534c07917307c381f5af4053c6091b5278` | 241.9 GiB, 186 shards | accept CC BY-NC-SA 4.0 |
| SuSy | [`aminasifar1/SuSy-Dataset`](https://huggingface.co/datasets/aminasifar1/SuSy-Dataset) | `df5f324e4438cddaaf0de87f231c356b47aa555d` | train 14.1 GiB, validation 4.4 GiB | none |
| MS-COCOAI (Defactify) | [`Rajarshi-Roy-research/Defactify_Image_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) | `787334f7857fa54f29027a7f09c30e895ad486ef` | 3.6 GiB (train + validation) | none |

- **SID_Set archives:** `train_real.zip` (20.28 GB), `train_full_synthetic_part1.zip`
  (37.18 GB), `train_full_synthetic_part2.zip` (37.3 GB), `train_tampered.zip`
  (19.2 GB), `train_masks.zip` (665.5 MB), `validation.zip` (15.64 GB), and
  `test.zip`. The links come from the [SIDA repository](https://github.com/hzlsaber/SIDA).
  The Hugging Face copy `saberzl/SID_Set` is Parquet and has no test split, so it
  cannot feed `audit/build_dataset_manifests.py`.
- **WildFake:** only `coco2017/val2017` (4,998 images) and `DALLE/Advanced`
  (8,843 images) are used. WildFake is never used for training, model selection, or
  threshold tuning; the build uses it for a count check and to exclude overlapping
  CommunityForensics images.
- **Sealed splits:** never download SuSy `data/test.zip` or the Defactify `test-*`
  shards.
- **Licenses:** SID_Set CC BY 4.0; CommunityForensics-Small CC BY-NC-SA 4.0; GenImage
  non-commercial; WildFake is listed as Apache-2.0 on ModelScope; SuSy carries
  per-source terms (see `SELECTED_SOURCES` in `scripts/prepare_susy_vnext.py`);
  MS-COCOAI declares no license at the pinned revision. See also
  [`../MODEL_USAGE_NOTICE.md`](../MODEL_USAGE_NOTICE.md).
- The three sources outside Hugging Face are not pinned by hash. The count checks in
  each step and the final [verification](#verify-the-rebuild) detect upstream changes.

## Before you start

- Put the working copy on a volume with enough free space. The scripts always write
  into `<repo>/Dataset/`. The download scripts check free space on the volume that
  holds `Dataset/`; if you redirect subfolders to another disk with junctions or
  symlinks, those checks measure the wrong disk.
- Keep Hugging Face caches off the system drive, for example
  `$env:HF_HOME = 'D:\hf_home'` pointing at a large volume.
- Install [7-Zip](https://www.7-zip.org/) for `7z l`, `7z t`, and extraction.
- Accepting dataset licenses (`--accept-license`) and signing in to Google, Kaggle, or
  ModelScope are manual steps.

## Order matters

Later steps de-duplicate against manifests written by earlier steps. A missing
manifest is treated as empty rather than as an error, so running steps out of order
silently changes every downstream sample. Follow the numbered order.

## Full rebuild

### 1. Place the manually downloaded sources

Arrange the SID_Set, CIFAKE, and WildFake files like this:

```text
Dataset/
├── _archives/SID_Set/          # the seven SID_Set ZIPs; needed until step 2 finishes
├── SID_Set/
│   ├── train/{real,full_synthetic,tampered,masks}/
│   ├── validation/{real,full_synthetic,tampered,masks}/
│   └── test/{real,full_synthetic,tampered,masks}/
├── CIFAKE/{train,test}/{REAL,FAKE}/
└── WildFake_demo/Images/
    ├── Real/coco/coco2017/val2017/
    └── Diffusion_based/DALLE/Advanced/
```

`audit/build_dataset_manifests.py` enumerates the SID_Set ZIPs and expects each image
at the extracted path below:

| Archive | Top-level folder inside the ZIP | Extract to |
|---|---|---|
| `train_real.zip` | `real/` | `SID_Set/train/real/` |
| `train_full_synthetic_part1.zip` | `full_synthetic_part1/` | `SID_Set/train/full_synthetic/` |
| `train_full_synthetic_part2.zip` | `full_synthetic_part2/` | `SID_Set/train/full_synthetic/` |
| `train_tampered.zip` | `tampered/` | `SID_Set/train/tampered/` |
| `train_masks.zip` | masks | `SID_Set/train/masks/` |
| `validation.zip` | `real/`, `full_synthetic/`, `tampered/`, masks | `SID_Set/validation/` |
| `test.zip` | `real/`, `full_synthetic/`, `tampered/`, masks | `SID_Set/test/` |

- The two `full_synthetic_part*` folders merge into one `full_synthetic/` directory;
  the part prefix is dropped and the relative paths below it are kept.
- Every tampered image needs a mask named `<stem>_mask.png` in its split's `masks/`
  directory.
- The inside-ZIP layout is inferred from the build script. List each archive with
  `7z l` before extracting; the build script stops on any missing image or mask.
- For WildFake, extract only the two directories shown above from `coco.zip` and
  `DALLE.zip`, then discard the rest.

### 2. Base manifests: SID_Set, CIFAKE, WildFake

```powershell
python Dataset/audit/build_dataset_manifests.py
```

Expected: SID_Set official 210,000 / 30,000 / 60,000; SID_Set clean
202,820 / 29,272 / 59,853; CIFAKE 120,000; WildFake 13,841 (4,998 + 8,843).

After this succeeds you may delete `_archives/SID_Set/`, `SID_Set/test/`, and
`CIFAKE/`; nothing downstream reads them. Keep `WildFake_demo/` until step 7. This step
cannot be rerun without the ZIPs, so back up `manifests/` and `audit/` first.

### 3. SID_Set training manifests

```powershell
python scripts/build_training_manifests.py
```

Writes `training_main.csv` (202,820 SID_Set rows) and the smoke manifests.

### 4. GenImage subset

```powershell
python scripts/download_genimage_subset.py --accept-license
python scripts/verify_genimage_subset.py
```

- Exports 290,000 images: 140,000 ImageNet real images; 20,000 each from Midjourney,
  SD1.4, SD1.5, ADM, BigGAN, VQDM, and Wukong; and a 5,000 + 5,000 GLIDE holdout.
- The download is resumable.
- Arrow shards accumulate in `_hf_cache_shards/` (roughly 85 GiB, estimated) and are
  not deleted automatically. Remove `_hf_cache/` and `_hf_cache_shards/` once
  verification passes.

### 5. Multi-source manifests

```powershell
python scripts/build_multisource_manifests.py
```

Hashes every SID_Set train and validation image (slow; `--workers` sets the thread
count). Writes `training_multisource.csv`, `validation_multisource_full.csv`, and the
16,000-row `validation_multisource.csv`.

### 6. Selection and confirmation splits

```powershell
python scripts/build_robust_validation_splits.py
```

Writes `validation_selection_6000.csv` and `validation_confirmation_10000.csv`.

### 7. CommunityForensics-Small

```powershell
python scripts/download_communityforensics_full.py --accept-license
```

- Requires `wildfake_demo.csv` with its images and the step 5 manifests; it uses them
  to exclude overlapping images.
- Streams the 186 Parquet shards one at a time, exports the images, and deletes each
  shard afterwards (`--keep-parquet` keeps them).
- Refuses to start a shard unless 40 GiB plus the shard size is free.
- Expected: 553,531 usable images (277,969 real / 275,562 fake), with 101 exact
  duplicates and 24 WildFake overlaps excluded. Compare
  [`../reports/final_adapter_v2/dataset_summary.json`](../reports/final_adapter_v2/dataset_summary.json).
- `WildFake_demo/` images can be deleted after this step.

### 8. Modern-generator manifests and the historical anchor

```powershell
python scripts/build_modern_generator_manifests.py
```

Writes `communityforensics_train_balanced.csv` (293,792 rows), the generator-disjoint
CommunityForensics selection and confirmation splits (6,000 each),
`validation_modern_combined_selection_12000.csv` (the historical anchor), and
`validation_modern_combined_confirmation_16000.csv`.

### 9. Replay manifest

```powershell
python scripts/build_replay_manifest.py
```

Writes `replay_balanced_560000.csv`.

### 10. SuSy

```powershell
python scripts/download_susy_vnext.py
hf download aminasifar1/SuSy-Dataset susy_dataset.json --repo-type dataset --revision df5f324e4438cddaaf0de87f231c356b47aa555d --local-dir Dataset/SuSy
python scripts/prepare_susy_vnext.py
```

- `download_susy_vnext.py` does not fetch `susy_dataset.json`, which
  `prepare_susy_vnext.py` needs; the `hf download` line fills the gap.
- Expected: 14,451 train / 5,555 validation images.
- Preparation stops if any SuSy image duplicates an image in an existing manifest.

### 11. MS-COCOAI

```powershell
python scripts/download_cocoai_vnext.py
python scripts/prepare_cocoai_vnext.py
```

Expected: 34,969 train / 7,341 validation images, all synthetic. The MS-COCOAI real
rows drop out here as exact duplicates of COCO images already present in the
CommunityForensics manifests.

### 12. Tiny vNext manifests

```powershell
python scripts/build_tiny_vnext_manifests.py --total-train 280000 --seed 2026
```

- Writes `tiny_vnext_train_balanced_280000.csv` (280,000 rows) and
  `tiny_vnext_modern_dev.csv` (12,896 rows: 1,234 real, 11,662 synthetic).
- The script writes both files before checking train/development overlap. If it
  reports an overlap, delete both outputs before retrying.

## Verify the rebuild

The released checkpoint produces fixed metrics on the two evaluation manifests, so
those metrics fingerprint the exact image sets:

```powershell
python evaluate.py --checkpoint weights/aigc-detector-ensemble-vnext.pt --manifest validation_modern_combined_selection_12000.csv --suite full --output outputs/rebuild_check/historical.json
python evaluate.py --checkpoint weights/aigc-detector-ensemble-vnext.pt --manifest tiny_vnext_modern_dev.csv --suite full --output outputs/rebuild_check/modern.json
```

| Manifest | Rows | Expected `robustness.robust_score` | Reference |
|---|---:|---:|---|
| `validation_modern_combined_selection_12000.csv` | 12,000 | 0.978314 | [`packaged_historical_evaluation.json`](../reports/ensemble_vnext/packaged_historical_evaluation.json) |
| `tiny_vnext_modern_dev.csv` | 12,896 | 0.901913 | [`modern_full_evaluation.json`](../reports/ensemble_vnext/modern_full_evaluation.json) |

- Agreement within about 1e-4 means the rebuilt set matches the original.
- CUDA inference runs in FP16, so the last digits can differ slightly across GPUs.
- A larger gap means the image set differs; re-check the step order and the expected
  counts above.

## Modern development set only

`tiny_vnext_modern_dev.csv` contains every image in the SuSy validation archive
(5,555 images, including all 1,234 real images) plus the 7,341 synthetic MS-COCOAI
validation images that survive preparation. No sampling is involved, so about 8 GB of
downloads is enough:

```powershell
python scripts/download_susy_vnext.py --splits val
hf download aminasifar1/SuSy-Dataset susy_dataset.json --repo-type dataset --revision df5f324e4438cddaaf0de87f231c356b47aa555d --local-dir Dataset/SuSy
python scripts/prepare_susy_vnext.py --splits val
python scripts/download_cocoai_vnext.py
python scripts/prepare_cocoai_vnext.py
```

`download_cocoai_vnext.py` fetches both MS-COCOAI splits on purpose: preparation
removes development rows that share a prompt or exact image with the training split.

No script automates the remaining two steps yet:

1. Without the CommunityForensics manifests, `prepare_cocoai_vnext.py` keeps the
   MS-COCOAI real rows. Drop every row with `binary_label == 0` from
   `cocoai_vnext_dev.csv`; the full chain removes all of them.
2. `build_tiny_vnext_manifests.py` also needs the replay manifest, so build the
   development manifest by hand:
   - concatenate `susy_vnext_dev.csv` and the filtered `cocoai_vnext_dev.csv`;
   - set `dataset` to `TinyVNext-Modern`, `role` to `tiny_vnext_development`, and
     `allowed_for_training` to `False`;
   - save the result as `manifests/tiny_vnext_modern_dev.csv`.

Then confirm 12,896 rows with 1,234 real images and run the modern verification
command above.

## Disk and time budget

| Source | Download | Kept after cleanup |
|---|---:|---:|
| SID_Set | 130.3 GB + test.zip | ~130 GB (train + validation) |
| CIFAKE | ~0.1 GB | 0 (deleted after step 2) |
| WildFake | 26 GiB | 0 (deleted after step 7) |
| GenImage subset | ~85 GiB (estimated) | ~50 GiB (estimated) |
| CommunityForensics-Small | 241.9 GiB | ~230 GiB |
| SuSy | 18.5 GiB | ~18.5 GiB extracted, plus the archives unless deleted |
| MS-COCOAI | 3.6 GiB | ~3.6 GiB extracted, plus the Parquet unless deleted |

- Usage peaks at step 2, when the SID_Set ZIPs and their extracted images coexist,
  and at step 7, which requires 40 GiB of headroom on top of everything kept.
- Plan for at least 550 GB free; 1 TB is comfortable.
- At 100 Mbit/s the downloads alone take about 13 hours. Google Drive may throttle
  large files, and hashing and extraction add several hours.

## After the rebuild

Back up `manifests/` and `audit/` outside the working tree before deleting any image
bodies. They are small (hundreds of MB) and record every selection decision with
SHA-256 hashes. With them, a later rebuild only needs to re-fetch the images and check
them against the recorded hashes instead of re-running the whole chain.

## Known gaps

- SID_Set, CIFAKE, and WildFake have no download or extraction script.
- `download_susy_vnext.py` does not fetch `susy_dataset.json`; step 10 works around it.
- No script builds the modern development set on its own.
