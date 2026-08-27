# Robust Detection of AI-Generated Images Under Real-World Transformations

**TikTok TechJam 2026 — Track 5**

AI-generated images are increasingly indistinguishable from real photographs, and in
real deployments they rarely arrive clean: they are re-compressed, blurred, downscaled,
noised, color-shifted, and cropped by social-media pipelines. This project builds a
binary detector that must keep working after those real-world transformations.

Our submission is a **ConvNeXt-Tiny binary classifier** (27,820,897 parameters — well
under the official 2-billion parameter limit) trained on a multi-source mixture of
real and synthetic images, with a blur-focused fine-tuning stage, and evaluated under
the official 16-condition robustness protocol on a strictly held-out WildFake split.

## Results at a glance

Evaluation set: official WildFake demo subset (4,998 COCO val2017 real + 8,843
DALL·E 3 Advanced synthetic = 13,841 images per condition), **never used for training
or threshold selection**. Metric: ROC AUC.

| Family | Conditions | AUC range |
|---|---|---|
| Clean | clean | **0.9636** |
| JPEG compression | q90 / q70 / q50 / q30 | 0.9818 / 0.9887 / 0.9897 / 0.9808 |
| Gaussian blur | σ 0.5 / 1.0 / 2.0 | 0.9604 / 0.9469 / 0.8151 |
| Downscale | ×0.5 / ×0.25 | 0.9407 / 0.9522 |
| Additive noise | σ 0.02 / 0.05 / 0.10 | 0.8598 / 0.8719 / 0.8576 |
| Brightness shift | −0.20 / +0.20 | 0.9284 / 0.9240 |
| Center crop | 0.80 | 0.9082 |

- Mean degraded AUC: **0.9271** · Worst degraded AUC: **0.8151** (`blur_2.0`)
- Robust score (`0.8 × mean + 0.2 × worst`): **0.9047** (previous multisource model: 0.8972)
- Decision threshold **0.209**, calibrated only on the internal validation split
  (five degradation conditions), then frozen.

Full breakdowns: [`docs/ROBUSTNESS_SUMMARY.md`](docs/ROBUSTNESS_SUMMARY.md) ·
error analysis: [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md)

## How we got here

1. **SID-only baseline** — strong in-domain (mean degraded AUC 0.9732 on SID
   validation) but failed cross-source: WildFake clean AUC was only ~0.646.
2. **Multi-source training** (SID_Set + GenImage subset, multiple generators, exact-hash
   deduplication) restored cross-source generalization.
3. **Blur fine-tuning** from the frozen multisource checkpoint, raising `blur_2.0`
   training exposure from ~5.8% to 20%, repaired the weakest condition
   (WildFake `blur_2.0` AUC 0.7834 → 0.8151) and produced the final submission model.

See [`reports/main_baseline_analysis.md`](reports/main_baseline_analysis.md) and
[`reports/wildfake_analysis_blur_finetune/report.md`](reports/wildfake_analysis_blur_finetune/report.md).

## Environment setup

Versions are **not pinned**; the project runs on recent stable releases.

```powershell
conda create -n jam python=3.11 -y
conda activate jam
pip install -r requirements.txt
```

Key libraries: PyTorch, torchvision, timm (ConvNeXt-Tiny weights), pandas,
scikit-learn, Pillow, PyYAML, tqdm, tensorboard, datasets, pyarrow.

## Reproduction

### 1. Data preparation

Place the datasets under `Dataset/` (see `Dataset/README_DATASET.md`), then build
training manifests:

```powershell
python scripts/build_training_manifests.py          # SID-only manifests
python scripts/build_multisource_manifests.py       # SID + GenImage manifests (deduplicated)
```

GenImage shards can be fetched with `scripts/download_genimage_subset.py` and verified
with `scripts/verify_genimage_subset.py`.

### 2. Verify the pipeline (smoke tests)

```powershell
python test.py
python train.py --config configs/baseline_smoke.yaml --epochs 1 --max-train-batches 5 --max-val-batches 5 --num-workers 0 --no-pretrained --output-dir outputs/pipeline_test
python train.py --config configs/multisource_smoke.yaml
```

### 3. Train

```powershell
# Stage 1: multi-source training (downloads ImageNet weights on first run)
python train.py --config configs/multisource.yaml

# Stage 2: blur fine-tuning from the frozen multisource checkpoint
python train.py --config configs/multisource_blur_finetune.yaml --init-checkpoint outputs/multisource/best.pt
```

Checkpoint selection uses `0.8 × mean degraded AUC + 0.2 × worst degraded AUC` on the
internal validation split only — never on WildFake.

### 4. Evaluate (16-condition robustness suite)

```powershell
python evaluate.py --checkpoint outputs/multisource_blur_finetune/best.pt --suite full
```

Per-image score export for threshold calibration and held-out error analysis:

```powershell
python evaluate.py --checkpoint outputs/multisource/best.pt --manifest validation_multisource.csv --conditions clean,jpeg_30,blur_2.0,scale_0.25,noise_0.10 --output outputs/multisource/calibration_validation_5_conditions.json --predictions-output outputs/multisource/calibration_validation_5_conditions_predictions.csv
python evaluate.py --checkpoint outputs/multisource/best.pt --manifest wildfake_demo.csv --conditions clean,blur_2.0 --output outputs/multisource/wildfake_error_conditions.json --predictions-output outputs/multisource/wildfake_error_conditions_predictions.csv
python scripts/analyze_wildfake.py --calibration-predictions outputs/multisource/calibration_validation_5_conditions_predictions.csv --target-predictions outputs/multisource/wildfake_error_conditions_predictions.csv --full-evaluation outputs/multisource/wildfake_demo_full.json --output-dir reports/wildfake_analysis
```

The threshold is selected only on `validation_multisource.csv`; WildFake remains held
out and is never used for training or threshold selection.

### 5. Submission-format inference

```powershell
python predict.py --checkpoint outputs/multisource_blur_finetune/best.pt --input-dir path/to/images --output predictions.json
```

Output is a JSON array of `{"image_path", "pred"}` objects; the input directory is
scanned recursively and unreadable images receive the neutral fallback score `0.5`.

## Model weights

The final checkpoint (`multisource_blur_finetune/best.pt`) will be distributed via a
GitHub Release:

> `https://github.com/<OWNER>/<REPO>/releases`  *(placeholder — to be replaced)*

Each Release asset ships with its **SHA256 checksum**; verify with:

```powershell
Get-FileHash best.pt -Algorithm SHA256
```

and compare against the checksum published in the Release notes.

## Limitations & future work

- **Noise robustness regressed slightly.** The blur fine-tune traded a little noise
  performance (`noise_0.10` AUC 0.8780 → 0.8576) for blur gains. Next step: a joint
  blur+noise augmentation schedule or a second fine-tune stage restoring noise exposure.
- **`blur_2.0` is still the weakest condition** (AUC 0.8151). Strong blur destroys the
  high-frequency forensic cues the model relies on; a multi-scale or frequency-domain
  branch could help.
- **Content-driven systematic errors.** A small set of COCO real images is repeatedly
  flagged fake with scores > 0.99, and a few DALL·E 3 Advanced images are missed with
  scores < 0.001 across *all* conditions (see
  [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md)). These are semantic failure modes
  that augmentation cannot fix; semantic/CLIP-style features are the planned remedy.
- **Cross-generator generalization is unproven beyond DALL·E 3.** Training covers
  several generators via GenImage, but evaluation was only possible on DALL·E 3
  Advanced; behavior on unseen generator families needs further testing.
- **Fixed single threshold.** 0.209 optimizes mean balanced accuracy across the five
  calibration conditions; deployments with different FP/FN cost profiles should
  re-calibrate on their own held-out data.

## Team contributions

| Member | Contribution |
|---|---|
| _[Name 1]_ | _[e.g., multi-source training pipeline, blur fine-tuning]_ |
| _[Name 2]_ | _[e.g., robustness evaluation harness, error analysis]_ |
| _[Name 3]_ | _[e.g., data preparation, documentation]_ |

*(Placeholders — replace before submission.)*

## Compliance statements

- **Parameter budget:** the model is ConvNeXt-Tiny with **27,820,897 parameters**,
  far below the official 2-billion parameter cap.
- **Data compliance:** training data comes from publicly released research datasets
  (SID_Set; GenImage subset from public generators; COCO val2017 real images via the
  official WildFake demo bundle). CIFAKE was kept for ablation only and excluded from
  the final training mixture. The WildFake evaluation subset was used exclusively for
  demonstration evaluation — never for training, checkpoint selection, or threshold
  tuning.
- **Evaluation integrity:** threshold 0.209 was frozen on the internal validation
  split before any WildFake scoring of the final model.

## Assumptions

1. Images are standard RGB formats decodable by Pillow (JPEG/PNG/WebP); corrupt files
   fall back to the neutral score 0.5.
2. The scoring metric is ROC AUC per condition, with the robust score
   `0.8 × mean degraded AUC + 0.2 × worst degraded AUC`.
3. The official degradation transforms (JPEG quality, Gaussian blur σ, downscale
   factor, Gaussian noise σ, brightness shift, center-crop ratio) are applied by the
   organizer-side evaluation harness; our `evaluate.py` reproduces them locally for
   development.
4. One score per image; no ensembling or test-time augmentation at inference.
5. Hardware assumption: a single CUDA GPU (training ran on one GPU); CPU inference is
   supported but slower.

## Repository layout

```
configs/        YAML training configs (baseline, multisource, blur fine-tune, smoke)
scripts/        manifest builders, GenImage downloader/verifier, error analysis
src/            model, dataset, transforms, training engine, metrics
train.py        training entry point
evaluate.py     16-condition robustness evaluation entry point
test.py         unit/pipeline tests
predict.py      submission-format inference
docs/           reviewer-facing deliverables (robustness summary, error analysis, …)
reports/        machine-generated analysis artifacts
Dataset/        datasets (not committed; see Dataset/README_DATASET.md)
```
