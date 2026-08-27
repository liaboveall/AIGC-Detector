# WildFake held-out robustness and error analysis

## Evaluation policy

WildFake remained held out: it was not used to train the model or choose the threshold. The single threshold `0.209` was selected on `calibration_validation_5_conditions_predictions.csv` by maximising mean balanced accuracy across the five internal validation conditions. It was then frozen and applied to `wildfake_error_conditions_predictions.csv`.

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
| clean | 0.500 | 0.8441 | 0.8669 | 0.9646 | 0.7848 | 0.9490 | 0.8654 | 255 | 1903 |
| clean | 0.209 | 0.8774 | 0.8853 | 0.9461 | 0.8569 | 0.9136 | 0.8993 | 432 | 1265 |
| blur_2.0 | 0.500 | 0.7154 | 0.7009 | 0.7913 | 0.7533 | 0.6485 | 0.7718 | 1757 | 2182 |
| blur_2.0 | 0.209 | 0.7244 | 0.6798 | 0.7556 | 0.8406 | 0.5190 | 0.7958 | 2404 | 1410 |

## Error analysis

- The weakest ranking condition is `blur_2.0` with ROC AUC 0.7834; this is consistent with strong blur removing or distorting high-frequency forensic cues.
- At the frozen calibrated threshold, clean false positives/false negatives are 432/1265; under `blur_2.0` they become 2404/1410.
- Calibration improves clean balanced accuracy from 0.8669 to 0.8853, but under `blur_2.0` it changes from 0.7009 to 0.6798. The lower threshold recovers more synthetic images but also increases false positives, so it is not a universal fix for severe blur.
- JPEG AUC remains high even when fixed-threshold recall falls, indicating score calibration shift rather than a complete loss of ranking ability.
- The submission interface should continue to emit confidence scores. The calibrated threshold is for binary demo decisions and is not baked into the score output.
- Representative high-confidence errors are listed below and fully exported in `error_cases.csv`.

### clean: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161581.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159317.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160993.jpg` — score 0.9985
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161863.jpg` — score 0.9980

### clean: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/c5097ad662b67c9af001724e3a102d3c.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/0763e21106d9bb5adb44fafdebba1337.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0009
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/f3969cbaa698d69874ad67eda89fc187.jpg` — score 0.0011
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/edc0ae92b84620fd57f2a72508c01e67.jpg` — score 0.0012

### blur_2.0: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163903.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163909.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163583.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162986.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162044.jpg` — score 0.9995

### blur_2.0: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a6fee19a25b72d87e762b8b5ed5986cf.jpg` — score 0.0011
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/ed4740ee61bd78c461b59c8d1ec73304.jpg` — score 0.0012
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0012
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0016
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/dde040947d7f2ea164c178f5b7dad202.jpg` — score 0.0021

## Calibration details

- Objective: mean condition-balanced accuracy; ties use worst condition then proximity to 0.5
- Mean internal balanced accuracy: 0.8873
- Worst internal balanced accuracy: 0.8483

The WildFake demonstration split is a reference benchmark only and does not contribute to the final score.
