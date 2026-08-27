# Deliverable 5 — Error Analysis

Scope: final submission model (`outputs/multisource_blur_finetune/best.pt`) on the held-out
WildFake demo subset (13,841 images/condition), frozen threshold **0.209**.
All findings below are drawn from
[`../reports/wildfake_analysis_blur_finetune/report.md`](../reports/wildfake_analysis_blur_finetune/report.md);
full per-case exports live in
[`error_cases.csv`](../reports/wildfake_analysis_blur_finetune/error_cases.csv).

## 1. Weakest point: heavy blur (`blur_2.0`)

- Lowest ranking performance of all 16 conditions: **ROC AUC 0.8151**
  (clean: 0.9636; blur_1.0: 0.9469).
- At threshold 0.209 the confusion shifts dramatically: clean FP/FN = 363/1322, but under
  `blur_2.0` it becomes **1,899 FP / 1,504 FN** — the condition with the highest
  false-positive count and the most severe FP/FN imbalance. (`color_-0.20` is the only
  other condition where FPs outnumber FNs, 1,227/867; every other condition stays
  FN-dominated.)
- Cause: σ=2.0 Gaussian blur removes/distorts the high-frequency forensic cues the
  detector exploits. The blur fine-tune stage already improved this condition materially
  (WildFake `blur_2.0` AUC 0.7834 → 0.8151), and per-image inspection shows the fix works
  mainly by pushing *real*-image scores down (blur_2.0 real median 0.1788 → 0.0552) so the
  frozen threshold transfers without recalibration.

## 2. JPEG calibration drift: ranking survives, binary decisions pay

- All JPEG qualities keep **higher AUC than clean** (0.9808–0.9897), i.e. the model's
  ranking ability is fully intact — or better — under recompression.
- Yet fixed-threshold accuracy collapses from 0.8985 (q90) to **0.6971 (q30)**: strong
  compression compresses the fake-score distribution toward the threshold, so recall at
  0.209 drops to 0.5269 even though AUC stays at 0.9808.
- Interpretation: this is a **score-calibration shift, not a loss of discriminative
  signal**. The loss happens in the binary decision layer; emitting continuous confidence
  scores (as our submission does) avoids hard-coding this fragility.
- Example (jpeg_30 false positive): `WildFake_demo/Images/Real/coco/coco2017/val2017/img163785.jpg`
  scored 0.8960.

## 3. Content-driven hard cases (systematic, not degradation-driven)

A small set of images fails **across nearly every condition**, which rules out
transformation-specific causes:

**Recurring real-image false positives (COCO val2017), scores > 0.99 on clean:**

| Image | clean | blur_1.0 | jpeg_50 |
|---|---:|---:|---:|
| `Dataset/WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` | 0.9995 | 1.0000 | 0.6982 |
| `Dataset/WildFake_demo/Images/Real/coco/coco2017/val2017/img160993.jpg` | 0.9995 | — | — |
| `Dataset/WildFake_demo/Images/Real/coco/coco2017/val2017/img163818.jpg` | 0.9990 | — | — |

(`img159274.jpg` is the same kind of content-driven case, but its 0.9995 scores occur
under the `noise_0.02/0.05/0.10` conditions; on clean it does not enter the top
false-positive list.)

**Recurring synthetic-image false negatives (DALL·E 3 Advanced), scores < 0.001 on clean:**

| Image | clean | jpeg_30 | blur_2.0 |
|---|---:|---:|---:|
| `Dataset/WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` | 0.0005 | 0.0002 | 0.0006 |
| `Dataset/WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` | 0.0006 | 0.0002 | — |

Full paths above point into the (uncommitted) dataset tree under
`Dataset/WildFake_demo/`; every top case for all 16 conditions is exported in
[`error_cases.csv`](../reports/wildfake_analysis_blur_finetune/error_cases.csv).

These are **semantic failure modes** — the model mistakes certain photographic styles for
synthetic content and vice versa. Augmentation cannot fix them; richer (e.g. CLIP-style)
semantic features or generator-specific artifact branches are the planned remedy.

## 4. Trade-off accounting: blur up, noise down

Comparing the final model with the previous multisource model on the internal validation
(9,000 fakes per condition) and WildFake:

| Dimension | Old multisource | Blur fine-tune (final) |
|---|---:|---:|
| WildFake `blur_2.0` AUC | 0.7834 | **0.8151** |
| WildFake `noise_0.10` AUC | **0.8780** | 0.8576 |
| Robust score | 0.8972 | **0.9047** |
| `blur_2.0` real median score | 0.1788 | 0.0552 |
| `noise_0.10` FN @ 0.209 (internal val) | 753 | 1,004 |

The fine-tune concentrates its gains on the real-image side (lower real scores, fewer
blur FPs) and pays on the fake side under heavy JPEG/noise (more fake scores slip below
0.209). The net robust score improved, so the trade was accepted for the submission.

## 5. Threshold sanity

- 0.209 was chosen by maximizing mean balanced accuracy on the five internal calibration
  conditions (`clean, jpeg_30, blur_2.0, scale_0.25, noise_0.10`); no WildFake-based
  recalibration was performed afterwards.
- Calibration improved clean balanced accuracy 0.8703 → 0.8889 at the default-0.5
  comparison point, but under `blur_2.0` it is not a universal fix (0.7333 → 0.7250) —
  severe blur needs model-level improvement, not threshold tuning.
- Recommendation carried into the submission interface: always emit confidence scores;
  the calibrated threshold is for binary demo decisions only.
