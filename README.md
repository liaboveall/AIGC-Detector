# Robust AIGC Image Detector — TikTok TechJam Track 5

A degradation-aware detector for continuous AI-image scoring under JPEG,
Gaussian blur, resizing, additive noise, color shifts, and cropping.

## Current fusion candidate

`ensemble-vnext` is the frozen repository candidate. It blends two independently
trained logits with fixed equal weights:

```text
score = sigmoid(0.50 * Tiny-vNext-logit + 0.50 * Base-v1-logit)
```

- Asset: `weights/aigc-detector-ensemble-vnext.pt`
- SHA-256: `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`
- Parameters: 115,585,507 total, 0 trainable at inference
- Input: RGB images transformed to 224 × 224
- Output: one continuous probability in `[0, 1]`; higher means more likely AI-generated
- Publication state: complete and verified on branch `ensemble-vnext`, not yet merged,
  tagged, pushed, or uploaded as a new public release

The published `v1.0.0` Adapter v2 asset remains the rollback release. The fusion
branch does not change `main` or the `v1.0.0` tag.

## Results

These are internal development results, not an official hidden-test score.

### Historical 12,000-image selection anchor

The unchanged 16-condition Tiny-relative gates selected `alpha=0.50`. Alpha `0.60`
was rejected because its SID_Set scale-family drop was `0.005079`, beyond the
frozen `0.005000` limit.

| Metric | Tiny vNext | Ensemble vNext | Delta |
|---|---:|---:|---:|
| Robust score | 0.944886 | **0.978314** | **+0.033429** |
| Clean AUC | 0.972336 | **0.991032** | +0.018696 |
| Mean degraded AUC | 0.952022 | **0.982465** | +0.030443 |
| Worst degraded AUC | 0.916339 | **0.961712** | +0.045373 |
| CommunityForensics robust score | 0.935609 | **0.985517** | +0.049909 |
| GenImage robust score | 0.918252 | **0.960966** | +0.042714 |
| SID_Set robust score | 0.958064 | **0.967631** | +0.009568 |

### Source-disjoint modern development set

The frozen alpha was then run live on 12,896 images under all 16 conditions.

| Metric | Tiny vNext | Ensemble vNext | Delta |
|---|---:|---:|---:|
| Global robust score | 0.740077 | **0.901913** | **+0.161836** |
| Clean AUC | 0.850770 | **0.958501** | +0.107731 |
| Generator-macro robust score | 0.718331 | **0.894738** | **+0.176407** |
| Worst-generator robust score | 0.548970 | **0.796370** | +0.247400 |
| Worst generator-condition AUC | 0.322516 | **0.648097** | +0.325581 |

All 16 global condition AUCs improved. The 1,000-replicate content-group bootstrap
gave a macro-gain 95% interval of `[0.171015, 0.182232]`. Complete machine-readable evidence is tracked in
[`reports/ensemble_vnext/`](reports/ensemble_vnext/README.md).

### Evidence boundary

The historical confirmation split and WildFake were already consumed during earlier
project stages. They were not reopened for fusion selection, calibration, threshold
tuning, or a new sealed-performance claim. There is no newly validated binary
threshold for the ensemble; use the continuous score unless a new target-domain
calibration set is available.

## Quick start

```powershell
git clone https://github.com/liaboveall/AIGC-Detector.git
cd AIGC-Detector
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the committed LFS asset and both inference paths:

```powershell
Get-FileHash weights/aigc-detector-ensemble-vnext.pt -Algorithm SHA256
python scripts/verify_ensemble_release.py --device cuda
python scripts/verify_ensemble_release.py --device cpu
```

Run recursive directory inference:

```powershell
python predict.py --input-dir path/to/images --output predictions.json
```

The output contract is:

```json
[
  {"image_path": "subfolder/example.jpg", "pred": 0.9731}
]
```

Unreadable supported files receive the neutral score `0.5`. Override the checkpoint
with `--checkpoint PATH` or the device with `--device cpu` / `--device cuda:0`.

## Verification and reproduction

```powershell
python test.py
python -m compileall -q src scripts train.py train_adapter.py evaluate.py predict.py
python -m pip check
git diff --check
```

Repackage the fusion from the two hash-pinned source checkpoints:

```powershell
python scripts/package_ensemble_checkpoint.py `
  --checkpoint-a outputs/tiny_vnext/final_candidate/tiny_vnext_seed2026_gain1p60.pt `
  --checkpoint-b outputs/base_v1/primary/best.pt `
  --alpha 0.50 `
  --output weights/aigc-detector-ensemble-vnext.pt
```

The source checkpoint hashes and protocol are frozen in
[`docs/ENSEMBLE_VNEXT.md`](docs/ENSEMBLE_VNEXT.md). Dataset reconstruction requires
the private image trees and manifests described in
[`Dataset/README_DATASET.md`](Dataset/README_DATASET.md).

## Data governance and limitations

- Dataset image bodies and private manifests are not redistributed.
- CommunityForensics-Small is recorded as CC-BY-NC-SA-4.0. SID_Set, GenImage, COCO,
  CIFAKE, and WildFake remain subject to their upstream terms.
- Strong noise is the weakest global modern condition; the weakest generator-condition
  pair is Midjourney v1/v2 under `noise_0.05`.
- On an RTX 4080 Laptop GPU, paired median latency is 13.87 ms/image at batch 1 and
  92.88 ms/batch at batch 32 (2.77× and 3.37× Tiny vNext latency, respectively).
- Scores can reflect content, codec, camera, and semantic shortcuts. They are not proof
  of authorship, fraud, or misconduct.
- Use human review and target-domain validation for consequential decisions.
- No state-of-the-art, universal-generator, leaderboard, or competition-winning claim
  is made without a new official hidden evaluation.

## Project map

- [`MODEL_CARD.md`](MODEL_CARD.md) — current architecture, uses, risks, and evidence
- [`docs/ROBUSTNESS_SUMMARY.md`](docs/ROBUSTNESS_SUMMARY.md) — validation breakdown
- [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) — current failure analysis
- [`docs/DELIVERY_CHECKLIST.md`](docs/DELIVERY_CHECKLIST.md) — verified and external gates
- [`reports/ensemble_vnext/`](reports/ensemble_vnext/README.md) — tracked fusion evidence
- [`docs/TINY_VNEXT_RESULTS.md`](docs/TINY_VNEXT_RESULTS.md) — accepted Tiny member lineage
- [`reports/base_v1_primary.md`](reports/base_v1_primary.md) — selected Base member evidence
- [`reports/base_v3_outcome.md`](reports/base_v3_outcome.md) — rejected Base restart evidence
- [`reports/final_adapter_v2/`](reports/final_adapter_v2/README.md) — historical v1.0.0 evidence
- `predict.py` — directory-to-JSON inference entry point
- `evaluate.py` — deterministic robustness evaluation entry point
