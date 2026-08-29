# Robustness Summary — Frozen Adapter v2

**Release:** `v1.0.0`

**Checkpoint:** `aigc-detector-adapter-v2.pt`

**SHA-256:** `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`

**Architecture:** frozen ConvNeXt-Tiny + 768→256→1 residual adapter

**Parameters:** 28,018,018

**Frozen binary-demo threshold:** 0.209

## Evaluation protocol

The deterministic 16-condition suite contains:

- clean;
- JPEG quality 90, 70, 50, and 30;
- Gaussian blur sigma 0.5, 1.0, and 2.0;
- downscale to 0.5 and 0.25 followed by restoration to model input size;
- Gaussian noise sigma 0.02, 0.05, and 0.10;
- brightness shift -20% and +20%;
- center crop ratio 0.80 followed by restoration to model input size.

The project robustness score is:

```text
0.8 * mean AUC across the 15 degraded conditions
+ 0.2 * worst AUC among those degraded conditions
```

ROC AUC measures ranking independently of a threshold. Balanced accuracy is also
reported at the frozen threshold 0.209 to expose score-distribution shift.

## Internal development selection

The fixed selection split contains 12,000 images from CommunityForensics-Small,
GenImage, and SID_Set. CommunityForensics generators are disjoint from training and
the formerly sealed confirmation split. This split influenced model choice and is
therefore development evidence, not a final test.

| Metric | Frozen base | Adapter v2 | Delta |
|---|---:|---:|---:|
| Robust score | 0.930488 | **0.942425** | **+0.011938** |
| Clean AUC | 0.965637 | **0.973125** | +0.007488 |
| Mean degraded AUC | 0.939995 | **0.950238** | +0.010243 |
| Worst degraded AUC | 0.892458 | **0.911173** | +0.018715 |
| CommunityForensics robust score | 0.903910 | **0.928369** | +0.024459 |
| GenImage robust score | 0.920356 | 0.919772 | -0.000584 |
| SID_Set robust score | 0.958171 | 0.957819 | -0.000352 |

All 16 global condition AUCs improved. The formal 31-check gate passed 31/31:

- overall and CommunityForensics minimum scores passed;
- GenImage and SID robust-score drops stayed below 0.003;
- clean and global-family safeguards passed;
- all 18 source-by-family safeguards stayed below the pre-registered 0.005 bound;
- noise-family improvement and `blur_2.0` protection passed.

A separate, non-binding stress test tightened source-by-family drops from 0.005 to
0.002. It passed 30/31 checks; GenImage noise-family drop was 0.002583. The formal
bound was not changed after results were observed.

## WildFake one-time post-freeze observation

Composition: 4,998 COCO val2017 real images and 8,843 DALL-E 3 Advanced fake images,
13,841 images per condition. WildFake did not influence training, checkpoint selection,
or threshold selection.

| Condition | AUC | Balanced accuracy @ 0.209 |
|---|---:|---:|
| clean | **0.9647** | 0.8929 |
| jpeg_90 | **0.9825** | 0.9198 |
| jpeg_70 | **0.9891** | 0.9012 |
| jpeg_50 | **0.9899** | 0.8488 |
| jpeg_30 | **0.9811** | 0.7676 |
| blur_0.5 | 0.9618 | 0.8907 |
| blur_1.0 | 0.9488 | 0.8761 |
| blur_2.0 | **0.8221** | **0.7322** |
| scale_0.5 | 0.9428 | 0.8664 |
| scale_0.25 | 0.9548 | 0.8806 |
| noise_0.02 | 0.8665 | 0.8084 |
| noise_0.05 | 0.8759 | 0.7893 |
| noise_0.10 | **0.8604** | **0.7630** |
| color_-0.20 | 0.9323 | 0.8331 |
| color_+0.20 | 0.9265 | 0.8460 |
| crop_0.80 | 0.9112 | 0.8233 |

Summary versus the frozen base:

| Metric | Frozen base | Adapter v2 | Delta |
|---|---:|---:|---:|
| Clean AUC | 0.963591 | **0.964738** | +0.001148 |
| Mean degraded AUC | 0.927088 | **0.929697** | +0.002610 |
| Worst degraded AUC | 0.815118 | **0.822067** | +0.006949 |
| Robust score | 0.904694 | **0.908171** | +0.003477 |
| Mean balanced accuracy | 0.8341 | **0.8400** | +0.0059 |

All 16 AUC and balanced-accuracy values were non-decreasing. Across all conditions,
false negatives fell by 1,664 while false positives fell by 2, so the adapter's gain is
primarily improved fake-image recall without a systematic real-image false-positive
increase.

## What improved across the project

| Stage | WildFake clean AUC | WildFake robust score | Interpretation |
|---|---:|---:|---|
| SID-only | 0.6463 | — | severe cross-source collapse |
| Multi-source ConvNeXt-Tiny | 0.9609 | 0.8972 | generator diversity solved most of the gap |
| Blur-focused frozen base | 0.9636 | 0.9047 | weakest-condition repair |
| Frozen base + Adapter v2 | **0.9647** | **0.9082** | modern-domain gain with old-domain preservation |

The largest contribution came from source and generator coverage. The residual adapter
is a smaller but consistent final improvement with only 0.70% additional parameters.

## Robustness weaknesses

1. **Heavy blur remains the deployment bottleneck.** At `blur_2.0`, real-image false
   positive rate is 37.62% and fake recall is 84.06%. Blur destroys high-frequency
   forensic evidence and causes authentic content to move into the fake-score region.
2. **Strong noise remains weak.** At `noise_0.10`, fake recall is 66.55% and real-image
   false-positive rate is 13.95%.
3. **Strong JPEG reveals threshold shift.** `jpeg_30` AUC is 0.9811, but fake recall at
   threshold 0.209 is only 53.69%. Ranking survives; a single operating threshold does
   not transfer perfectly to the compressed score distribution.
4. **External generator coverage is narrow.** WildFake's fake side contains DALL-E 3
   Advanced only. It cannot establish universal generalization.

## Format-history and efficiency checks

The paired 3,000-image format audit found no detectable worsening versus the frozen
base. Candidate-minus-base absolute AUC changes were between -0.00004 and -0.00064,
well inside the bootstrap noise floor. Label-independent re-encoding sensitivity still
exists and is documented rather than described as solved.

On an RTX 4080 Laptop GPU, the adapter adds approximately 0.24 ms for batch 1, 1.0% for
batch 8, and 0.6% for batch 32. It adds 197,121 parameters to the 27.82M base.

## Evidence boundary

The 16,000-image confirmation set was opened exactly once for an earlier model-soup
candidate, which failed one source-family gate. The set was permanently consumed and
not reused for Adapter v2. Consequently, Adapter v2 is supported by the generator-
disjoint development selection result and the one-time post-freeze WildFake observation,
not by a fresh internal confirmation or official hidden test.

Machine-readable aggregate tables are in
[`../reports/final_adapter_v2/`](../reports/final_adapter_v2/README.md).
