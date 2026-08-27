# Deliverable 4 — Clean vs. Transformed Image Robustness Summary

**Model:** ConvNeXt-Tiny binary classifier, checkpoint `outputs/multisource_blur_finetune/best.pt`
**Evaluation set:** official WildFake demo subset — 4,998 COCO val2017 real + 8,843 DALL·E 3 Advanced
synthetic images = **13,841 images per condition** (held out from training and threshold selection).
**Metric:** ROC AUC. **Threshold:** 0.209 (frozen, calibrated on internal validation only).

## Clean baseline

| Condition | Images | ROC AUC | Avg. precision | Acc @ 0.209 | F1 @ 0.209 |
|---|---:|---:|---:|---:|---:|
| **clean** | 13,841 | **0.9636** | 0.9778 | 0.8783 | 0.8993 |

## Transformed conditions (grouped by transformation family)

### 1. JPEG compression

| Condition | ROC AUC | Δ vs clean | Avg. precision | Acc @ 0.209 |
|---|---:|---:|---:|---:|
| jpeg_90 | 0.9818 | +0.0182 | 0.9898 | 0.8985 |
| jpeg_70 | 0.9887 | +0.0251 | 0.9936 | 0.8696 |
| jpeg_50 | 0.9897 | +0.0261 | 0.9941 | 0.7985 |
| jpeg_30 | 0.9808 | +0.0172 | 0.9893 | 0.6971 |

### 2. Gaussian blur

| Condition | ROC AUC | Δ vs clean | Avg. precision | Acc @ 0.209 |
|---|---:|---:|---:|---:|
| blur_0.5 | 0.9604 | −0.0032 | 0.9763 | 0.8754 |
| blur_1.0 | 0.9469 | −0.0167 | 0.9685 | 0.8657 |
| blur_2.0 | **0.8151** | **−0.1485** | 0.8814 | 0.7541 |

### 3. Downscaling

| Condition | ROC AUC | Δ vs clean | Avg. precision | Acc @ 0.209 |
|---|---:|---:|---:|---:|
| scale_0.5 | 0.9407 | −0.0229 | 0.9654 | 0.8588 |
| scale_0.25 | 0.9522 | −0.0114 | 0.9736 | 0.8622 |

### 4. Additive Gaussian noise

| Condition | ROC AUC | Δ vs clean | Avg. precision | Acc @ 0.209 |
|---|---:|---:|---:|---:|
| noise_0.02 | 0.8598 | −0.1038 | 0.9285 | 0.7801 |
| noise_0.05 | 0.8719 | −0.0917 | 0.9256 | 0.7387 |
| noise_0.10 | 0.8576 | −0.1060 | 0.9000 | 0.7270 |

### 5. Brightness shift

| Condition | ROC AUC | Δ vs clean | Avg. precision | Acc @ 0.209 |
|---|---:|---:|---:|---:|
| color_−0.20 | 0.9284 | −0.0352 | 0.9586 | 0.8487 |
| color_+0.20 | 0.9240 | −0.0396 | 0.9538 | 0.8418 |

### 6. Center crop

| Condition | ROC AUC | Δ vs clean | Avg. precision | Acc @ 0.209 |
|---|---:|---:|---:|---:|
| crop_0.80 | 0.9082 | −0.0554 | 0.9487 | 0.7816 |

## Aggregate scores

| Metric | Value |
|---|---:|
| Clean AUC | 0.9636 |
| Mean degraded AUC (15 conditions) | **0.9271** |
| Worst degraded AUC (`blur_2.0`) | **0.8151** |
| Robust score (0.8 × mean + 0.2 × worst) | **0.9047** |
| Previous multisource model robust score | 0.8972 |

## Interpretation

- **JPEG compression does not hurt ranking — it slightly helps.** All four JPEG
  qualities score *above* clean AUC (0.98–0.99). The detector's forensic cues survive
  (and may even be sharpened by) recompression. However, heavy compression shifts the
  score distribution downward, so fixed-threshold accuracy still drops (0.6971 at q30)
  — a calibration effect, not a ranking failure.
- **Blur is the main vulnerability.** σ=2.0 blur costs ~15 AUC points because it
  destroys the high-frequency artifacts the model relies on. The blur fine-tune stage
  already lifted this from 0.7834 (previous model) to 0.8151; it remains the weakest
  condition and the top target for further work.
- **Noise is the second vulnerability** (~9–10 AUC points), roughly flat across the
  three noise levels, with a slight regression vs. the previous model (noise_0.10:
  0.8780 → 0.8576) accepted as the price of the blur improvement.
- **Downscaling, brightness, and crop are well handled** (≤ 5.5 AUC points below
  clean), and interestingly ×0.25 downscale scores *higher* than ×0.5, likely because
  the evaluation harness resizes back to a fixed input size, preserving content.

Raw numbers: [`../reports/wildfake_analysis_blur_finetune/robustness_table.csv`](../reports/wildfake_analysis_blur_finetune/robustness_table.csv).
