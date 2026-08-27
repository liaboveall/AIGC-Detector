# WildFake held-out robustness and error analysis

## Evaluation policy

WildFake remained held out: it was not used to train the model or choose the threshold. The single threshold `0.315` was selected on `calibration_predictions.csv` by maximising mean balanced accuracy across the five internal validation conditions. It was then frozen and applied to `wildfake_predictions.csv`.

## Headline robustness

| Condition | ROC AUC | Average precision | Accuracy @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| clean | 0.9609 | 0.9755 | 0.8441 | 0.7848 | 0.8654 |
| jpeg_90 | 0.9806 | 0.9888 | 0.8637 | 0.7964 | 0.8819 |
| jpeg_70 | 0.9880 | 0.9928 | 0.8116 | 0.7086 | 0.8278 |
| jpeg_50 | 0.9893 | 0.9934 | 0.7164 | 0.5574 | 0.7152 |
| jpeg_30 | 0.9806 | 0.9883 | 0.6047 | 0.3821 | 0.5526 |
| blur_0.5 | 0.9592 | 0.9750 | 0.8485 | 0.7917 | 0.8697 |
| blur_1.0 | 0.9420 | 0.9649 | 0.8372 | 0.7901 | 0.8612 |
| blur_2.0 | 0.7834 | 0.8648 | 0.7154 | 0.7533 | 0.7718 |
| scale_0.5 | 0.9353 | 0.9614 | 0.8336 | 0.7897 | 0.8584 |
| scale_0.25 | 0.9433 | 0.9681 | 0.8239 | 0.7568 | 0.8460 |
| noise_0.02 | 0.8638 | 0.9283 | 0.7398 | 0.6287 | 0.7553 |
| noise_0.05 | 0.8996 | 0.9399 | 0.7078 | 0.5647 | 0.7118 |
| noise_0.10 | 0.8780 | 0.9115 | 0.6958 | 0.5758 | 0.7075 |
| color_-0.20 | 0.9143 | 0.9492 | 0.8389 | 0.8521 | 0.8711 |
| color_0.20 | 0.9181 | 0.9499 | 0.7945 | 0.7356 | 0.8206 |
| crop_0.80 | 0.9091 | 0.9478 | 0.7283 | 0.5960 | 0.7370 |

## Threshold comparison on WildFake

| Condition | Threshold | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.500 | 0.8438 | 0.8438 | 0.9583 | 0.7188 | 0.9688 | 0.8214 | 1 | 9 |
| clean | 0.315 | 0.8438 | 0.8438 | 0.9231 | 0.7500 | 0.9375 | 0.8276 | 2 | 8 |
| blur_2.0 | 0.500 | 0.6406 | 0.6406 | 0.6452 | 0.6250 | 0.6562 | 0.6349 | 11 | 12 |
| blur_2.0 | 0.315 | 0.6562 | 0.6562 | 0.6389 | 0.7188 | 0.5938 | 0.6765 | 13 | 9 |

## Error analysis

- The weakest ranking condition is `blur_2.0` with ROC AUC 0.7834; strong blur removes or distorts the high-frequency cues used by the detector.
- At the frozen calibrated threshold, clean false positives/false negatives are 2/8; under `blur_2.0` they become 13/9.
- JPEG AUC remains high even when fixed-threshold recall falls, indicating score calibration shift rather than a complete loss of ranking ability.
- Representative high-confidence errors are listed below and fully exported in `error_cases.csv`.

### clean: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160249.jpg` — score 0.6938
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163446.jpg` — score 0.4927

### clean: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/57bec8c81d42da1440153e87f559d83e.jpg` — score 0.0079
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/11b2a25405796b97e3ad45186efbbd8f.jpg` — score 0.0127
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/160acb9076feae89d6c0ef45911b13f7.jpg` — score 0.0804
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/d05d6540053eb91682367a67f2cfd46e.jpg` — score 0.1058
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/c0e184c2affefc0e82da00870916db83.jpg` — score 0.1368

### blur_2.0: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160959.jpg` — score 0.9980
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163770.jpg` — score 0.9961
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159026.jpg` — score 0.9917
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160249.jpg` — score 0.9785
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161605.jpg` — score 0.9443

### blur_2.0: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/11b2a25405796b97e3ad45186efbbd8f.jpg` — score 0.0107
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/57bec8c81d42da1440153e87f559d83e.jpg` — score 0.0209
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/7e2ead9593aac3d45ae64e74cc0642c2.jpg` — score 0.0499
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/c0e184c2affefc0e82da00870916db83.jpg` — score 0.0709
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/d05d6540053eb91682367a67f2cfd46e.jpg` — score 0.1045

## Calibration details

- Objective: mean condition-balanced accuracy; ties use worst condition then proximity to 0.5
- Mean internal balanced accuracy: 0.9162
- Worst internal balanced accuracy: 0.8723

The WildFake demonstration split is a reference benchmark only and does not contribute to the final score.
