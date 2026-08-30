# Robust AIGC Image Detector — TikTok TechJam Track 5

A compact, degradation-aware detector for distinguishing authentic images from
AI-generated images after JPEG recompression, Gaussian blur, resizing, additive noise,
color shifts, and cropping.

**Release status:** `v1.0.0` is frozen. The formal checkpoint is a ConvNeXt-Tiny base
plus a small residual adapter: **28,018,018 parameters**, threshold **0.209**, SHA-256
`C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`.

The separately protected Tiny vNext result on `tiny-vnext` is documented
in [`docs/TINY_VNEXT_RESULTS.md`](docs/TINY_VNEXT_RESULTS.md). It does not
overwrite the `v1.0.0` rollback release.

## Results at a glance

### Internal development selection

Fixed 12,000-image, three-source selection split; 16 deterministic conditions. This is
the model-selection result, not an official hidden-test score.

| Metric | Frozen base | Adapter v2 | Delta |
|---|---:|---:|---:|
| Robust score | 0.930488 | **0.942425** | **+0.011938** |
| Clean AUC | 0.965637 | **0.973125** | +0.007488 |
| Mean degraded AUC | 0.939995 | **0.950238** | +0.010243 |
| Worst degraded AUC | 0.892458 | **0.911173** | +0.018715 |
| CommunityForensics robust score | 0.903910 | **0.928369** | +0.024459 |
| GenImage robust score | 0.920356 | 0.919772 | -0.000584 |
| SID_Set robust score | 0.958171 | 0.957819 | -0.000352 |

All 16 global condition AUCs improved. The candidate passed all **31/31 pre-registered
acceptance gates**, including source-specific degradation-family guards.

### One-time WildFake observation

After freezing the checkpoint and threshold, we observed it once on the organizer's
WildFake demo subset: 4,998 COCO real images and 8,843 DALL-E 3 Advanced images.
WildFake was never used for training, checkpoint selection, or threshold selection.

| Metric | Frozen base | Adapter v2 | Delta |
|---|---:|---:|---:|
| Robust score | 0.904694 | **0.908171** | +0.003477 |
| Clean AUC | 0.963591 | **0.964738** | +0.001148 |
| Mean degraded AUC | 0.927088 | **0.929697** | +0.002610 |
| Worst AUC (`blur_2.0`) | 0.815118 | **0.822067** | +0.006949 |
| Mean balanced accuracy at 0.209 | 0.8341 | **0.8400** | +0.0059 |

All 16 WildFake condition AUC and balanced-accuracy values were non-decreasing versus
the frozen base. WildFake remains a narrow, demonstration-only observation; no
leaderboard, hidden-test, or universal-generator claim is made.

Full aggregate evidence is tracked in
[`reports/final_adapter_v2/`](reports/final_adapter_v2/README.md).

## Model and training design

The final model wraps the accepted ConvNeXt-Tiny detector with a zero-initialized
`768 -> 256 -> 1` residual MLP:

```text
image -> frozen ConvNeXt-Tiny -> base logit
                          \-> pooled 768-d feature -> residual adapter -> + final logit
```

- Frozen base: 27,820,897 parameters.
- Trainable adapter: 197,121 parameters (0.70% of the total).
- Total: 28,018,018 parameters, far below the Track 5 two-billion-parameter cap.
- Training data: 560,000 balanced samples — CommunityForensics-Small 50%, GenImage
  25%, SID_Set 25%; each source is label-balanced.
- Objective: BCE on CommunityForensics plus a residual-squared preservation penalty on
  GenImage and SID_Set, preventing the modern-domain adaptation from overwriting older
  forensic knowledge.
- Augmentation: all six official degradation families plus 15% label-independent
  JPEG/WebP re-encoding to reduce compression-history shortcuts.
- Selection: robustness-aware score `0.8 * mean degraded AUC + 0.2 * worst degraded AUC`.

The experiment chain deliberately rejected modern-only fine-tuning, replay-only,
Replay+KD, and model-soup candidates when they violated cross-source guards. The formal
model is the best candidate that passed the complete pre-registered acceptance protocol,
not merely the checkpoint with the largest single headline number.

## Quick start

### 1. Install

```powershell
git clone https://github.com/liaboveall/AIGC-Detector.git
cd AIGC-Detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The frozen release was verified with Python 3.14.7, PyTorch 2.13.0+cu132, and the exact
top-level package versions in `requirements.txt`. Select the appropriate official
PyTorch CPU/CUDA wheel for the target machine if the default wheel is unsuitable.

### 2. Download and verify the frozen checkpoint

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest `
  https://github.com/liaboveall/AIGC-Detector/releases/download/v1.0.0/aigc-detector-adapter-v2.pt `
  -OutFile weights/aigc-detector-adapter-v2.pt
Get-FileHash weights/aigc-detector-adapter-v2.pt -Algorithm SHA256
python scripts/verify_release.py
```

The expected checksum is recorded in
[`weights/SHA256SUMS.txt`](weights/SHA256SUMS.txt). `verify_release.py` checks the hash,
checkpoint schema, parameter count, adapter flag, unreadable-image fallback, and exact
directory-to-JSON output contract.

### 3. Run submission-format inference

```powershell
python predict.py --input-dir path/to/images --output predictions.json
```

`predict.py` scans recursively and writes one continuous AI probability per supported
image:

```json
[
  {"image_path": "subfolder/example.jpg", "pred": 0.9731}
]
```

- `pred=0` means more likely authentic; `pred=1` means more likely AI-generated.
- Unreadable supported image files receive the neutral score `0.5` instead of crashing.
- Override the default asset with `--checkpoint PATH` and hardware with `--device cpu`
  or `--device cuda:0`.
- The frozen 0.209 threshold is documented for binary demos, but the required output is
  the continuous score.

## Verification

Self-contained unit and pipeline checks:

```powershell
python test.py
python -m compileall -q src scripts train.py train_adapter.py train_replay_distill.py evaluate.py predict.py
```

Full release check after downloading the checkpoint:

```powershell
python scripts/verify_release.py
```

Local 16-condition evaluation requires the private image trees and regenerated
manifests described in [`Dataset/README_DATASET.md`](Dataset/README_DATASET.md):

```powershell
python evaluate.py `
  --checkpoint weights/aigc-detector-adapter-v2.pt `
  --manifest validation_modern_combined_selection_12000.csv `
  --suite full `
  --output outputs/release_selection_full.json
```

## Training reproduction

Datasets and large manifests are intentionally excluded from Git. After recreating the
layout and manifests documented in [`Dataset/README_DATASET.md`](Dataset/README_DATASET.md),
the final adapter experiment is configured by [`configs/adapter_v2.yaml`](configs/adapter_v2.yaml):

```powershell
python train_adapter.py `
  --config configs/adapter_v2.yaml `
  --output-dir outputs/reproduction_adapter_v2
```

The config expects the protected historical base checkpoint at
`outputs/multisource_blur_finetune/best.pt`. Reproduction runs must use a new output
directory and must not overwrite the frozen release asset.

## Data governance

- CommunityForensics-Small: 553,531 usable images, 4,780 fake-generator identities;
  pinned source revision `6c539a534c07917307c381f5af4053c6091b5278`.
- Generator split: 3,822 train / 479 selection / 479 confirmation, mutually disjoint.
- Exact SHA-256 checks separate training from selection and confirmation; WildFake
  exact/perceptual overlap checks and NSFW exclusion are applied during export.
- Dataset image bodies and large manifests are never committed.
- CommunityForensics-Small is recorded as CC-BY-NC-SA-4.0. SID_Set, GenImage, COCO,
  CIFAKE, and WildFake remain subject to their respective upstream terms. Users must
  review those terms for their intended use; this repository does not redistribute the
  datasets.

Aggregate acquisition and split facts are preserved in
[`reports/final_adapter_v2/dataset_summary.json`](reports/final_adapter_v2/dataset_summary.json).

## Evaluation integrity and limitations

- The 12,000-image selection split is a development set and influenced model choice.
- The 16,000-image confirmation split was opened exactly once for an earlier model-soup
  candidate. That candidate was rejected, the split was permanently consumed, and it
  was not reused for Adapter v2.
- WildFake was observed once only after Adapter v2 and threshold 0.209 were frozen. It
  contains one fake-generator family and is demonstration evidence, not a final score.
- Heavy blur remains the primary failure mode (`blur_2.0` WildFake AUC 0.8221, balanced
  accuracy 0.7322). Strong noise and crop also remain weaker than clean/JPEG ranking.
- Under JPEG q30, ranking remains high (AUC 0.9811) but the fixed-threshold fake recall
  falls to 53.69%, showing score-distribution shift rather than ranking collapse.
- No claim of state of the art, universal generator coverage, or competition-winning
  performance is made without an official hidden evaluation.

See [`docs/ROBUSTNESS_SUMMARY.md`](docs/ROBUSTNESS_SUMMARY.md) and
[`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) for reviewer-facing detail.

## Submission materials

- [`docs/DEVPOST_DESCRIPTION.md`](docs/DEVPOST_DESCRIPTION.md) — copy-ready project text.
- [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md) — final-model three-minute shot list.
- [`docs/ROBUSTNESS_SUMMARY.md`](docs/ROBUSTNESS_SUMMARY.md) — 16-condition results.
- [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) — threshold and failure analysis.
- [`docs/DELIVERY_CHECKLIST.md`](docs/DELIVERY_CHECKLIST.md) — automated gates and the
  remaining Devpost-only human actions.

## Repository layout

```text
configs/                 Training and adaptation configurations
Dataset/                 Layout and deterministic manifest-rebuild documentation
docs/                    Devpost, demo, robustness, error, and delivery documents
reports/final_adapter_v2 Tracked aggregate evidence for the frozen model
scripts/                 Acquisition, audit, evaluation, and release-verification tools
src/                     Models, adapter, data, transforms, metrics, and training helpers
weights/                 Release download instructions and SHA-256 manifest
predict.py               Official directory-to-JSON inference entry point
evaluate.py              Deterministic robustness evaluation entry point
test.py                  Self-contained unit/pipeline checks
```
