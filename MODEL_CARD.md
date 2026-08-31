# Model Card — AIGC Detector Ensemble vNext

## Model details

- Status: frozen candidate on branch `ensemble-vnext`
- Asset: `aigc-detector-ensemble-vnext.pt`
- SHA-256: `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`
- Architecture: fixed 0.50/0.50 logit blend of:
  - ConvNeXt-Tiny plus a `768 -> 256 -> 1` residual adapter, gain `1.60`
  - ConvNeXt-Base `convnext_base.fb_in22k_ft_in1k`
- Parameters: 115,585,507 total; 0 trainable during inference
- Input: RGB images transformed to 224 × 224
- Output: one sigmoid probability in `[0, 1]`, where higher means more likely
  AI-generated
- Binary threshold: none newly validated for this ensemble

Both source model states, source hashes, and the fixed alpha are embedded in one
self-contained checkpoint. The loader freezes both members and keeps them in evaluation
mode.

## Intended use

Research, education, and hackathon demonstration of robust AI-generated-image
detection under JPEG, blur, resize, noise, color, and crop transformations. The
preferred interface is `predict.py`, which emits continuous scores for a directory.

## Out-of-scope use

- Treating the score as proof of authorship, fraud, or misconduct.
- High-stakes automated moderation without human review and target-domain validation.
- Claiming universal generator coverage or authenticity guarantees.
- Selecting, calibrating, or tuning on the already observed confirmation/WildFake data.
- Applying the historical Adapter v2 threshold `0.209` to the ensemble without a new,
  independent calibration set.
- Commercial use without reviewing all upstream dataset and checkpoint terms.

## Source models and data

| Member | Weight | Parameters | Source SHA-256 |
|---|---:|---:|---|
| Tiny vNext adapter | 0.50 | 28,018,018 | `1AF51D…EDE44` |
| Base v1 | 0.50 | 87,567,489 | `F49D42…64DD5` |

The embedded configurations record the balanced 280,000-row
`tiny_vnext_train_balanced_280000.csv` manifest for both members, with the six
degradation families and label-independent recompression augmentation. Dataset image
bodies and private manifests are not included in the repository.

## Evaluation summary

Historical 12,000-image development selection anchor, 16 conditions:

- robust score: `0.944886 -> 0.978314`
- clean AUC: `0.972336 -> 0.991032`
- worst degraded AUC: `0.916339 -> 0.961712`
- CommunityForensics / GenImage / SID_Set robust deltas:
  `+0.049909 / +0.042714 / +0.009568`
- alpha `0.50` passed all unchanged Tiny-relative gates; `0.60` was rejected

Source-disjoint modern development set, 12,896 images × 16 conditions:

- global robust score: `0.740077 -> 0.901913`
- clean AUC: `0.850770 -> 0.958501`
- generator-macro robust score: `0.718331 -> 0.894738`
- worst-generator robust score: `0.548970 -> 0.796370`
- worst generator-condition AUC: `0.322516 -> 0.648097`
- 1,000-replicate grouped-bootstrap macro-gain 95% CI: `[0.171015, 0.182232]`
- all 16 global condition AUCs improved

The historical confirmation split and WildFake were not reopened. Therefore this model
has strong development evidence but no new sealed, official-hidden-test, or external
generator claim. See [`reports/ensemble_vnext/`](reports/ensemble_vnext/README.md).

## Limitations and risks

- Strong additive noise is the weakest global condition.
- Midjourney v1/v2 under `noise_0.05` is the weakest evaluated generator-condition
  pair; Midjourney v6 has the lowest generator-level robust score.
- A two-network ensemble increases latency, memory use, and artifact size.
- Paired RTX 4080 Laptop measurements: batch 1 `5.01 -> 13.87 ms`; batch 32
  `27.57 -> 92.88 ms`.
- Content, camera, codec, semantic, and acquisition-pipeline shortcuts may affect
  predictions.
- Unseen generators, editing pipelines, regions, and content categories can behave
  differently.
- AUC improvements do not guarantee calibrated probabilities or fixed-threshold
  accuracy.

Outputs are uncertain model scores, not factual authenticity certificates.

## Reproducibility

The LFS checkpoint is hash-pinned in `weights/SHA256SUMS.txt`. Run:

```powershell
git lfs pull
python test.py
python scripts/verify_ensemble_release.py --device cuda
python scripts/verify_ensemble_release.py --device cpu
```

The exact alpha-selection and evidence protocol is in
[`docs/ENSEMBLE_VNEXT.md`](docs/ENSEMBLE_VNEXT.md). Training reproduction requires
the upstream datasets and private manifests documented in
`Dataset/README_DATASET.md`.
