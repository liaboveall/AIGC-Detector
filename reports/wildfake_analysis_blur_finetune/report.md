# WildFake held-out robustness and error analysis

## Evaluation policy

WildFake remained held out: it was not used to train the model or choose the threshold. The single threshold `0.209` was selected on `calibration_validation_5_conditions_predictions.csv` by maximising mean balanced accuracy across the five internal validation conditions. It was then frozen and applied to `wildfake_demo_full_predictions.csv`.

## Headline robustness

| Condition | ROC AUC | Average precision | Accuracy @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| clean | 0.9636 | 0.9778 | 0.8463 | 0.7837 | 0.8669 |
| jpeg_90 | 0.9818 | 0.9898 | 0.8617 | 0.7909 | 0.8796 |
| jpeg_70 | 0.9887 | 0.9936 | 0.8184 | 0.7179 | 0.8347 |
| jpeg_50 | 0.9897 | 0.9941 | 0.7325 | 0.5822 | 0.7355 |
| jpeg_30 | 0.9808 | 0.9893 | 0.6348 | 0.4287 | 0.6000 |
| blur_0.5 | 0.9604 | 0.9763 | 0.8444 | 0.7836 | 0.8655 |
| blur_1.0 | 0.9469 | 0.9685 | 0.8338 | 0.7794 | 0.8569 |
| blur_2.0 | 0.8151 | 0.8814 | 0.7397 | 0.7562 | 0.7878 |
| scale_0.5 | 0.9407 | 0.9654 | 0.8334 | 0.7832 | 0.8573 |
| scale_0.25 | 0.9522 | 0.9736 | 0.8308 | 0.7598 | 0.8516 |
| noise_0.02 | 0.8598 | 0.9285 | 0.7525 | 0.6502 | 0.7705 |
| noise_0.05 | 0.8719 | 0.9256 | 0.6884 | 0.5384 | 0.6883 |
| noise_0.10 | 0.8576 | 0.9000 | 0.6614 | 0.5229 | 0.6637 |
| color_-0.20 | 0.9284 | 0.9586 | 0.8506 | 0.8577 | 0.8800 |
| color_0.20 | 0.9240 | 0.9538 | 0.8015 | 0.7445 | 0.8274 |
| crop_0.80 | 0.9082 | 0.9487 | 0.7313 | 0.5991 | 0.7402 |

## Threshold comparison on WildFake

| Condition | Threshold | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.500 | 0.8463 | 0.8703 | 0.9699 | 0.7837 | 0.9570 | 0.8669 | 215 | 1913 |
| clean | 0.209 | 0.8783 | 0.8889 | 0.9540 | 0.8505 | 0.9274 | 0.8993 | 363 | 1322 |
| jpeg_90 | 0.500 | 0.8617 | 0.8890 | 0.9908 | 0.7909 | 0.9870 | 0.8796 | 65 | 1849 |
| jpeg_90 | 0.209 | 0.8985 | 0.9143 | 0.9815 | 0.8573 | 0.9714 | 0.9152 | 143 | 1262 |
| jpeg_70 | 0.500 | 0.8184 | 0.8570 | 0.9970 | 0.7179 | 0.9962 | 0.8347 | 19 | 2495 |
| jpeg_70 | 0.209 | 0.8696 | 0.8962 | 0.9944 | 0.8004 | 0.9920 | 0.8869 | 40 | 1765 |
| jpeg_50 | 0.500 | 0.7325 | 0.7903 | 0.9984 | 0.5822 | 0.9984 | 0.7355 | 8 | 3695 |
| jpeg_50 | 0.209 | 0.7985 | 0.8417 | 0.9975 | 0.6863 | 0.9970 | 0.8132 | 15 | 2774 |
| jpeg_30 | 0.500 | 0.6348 | 0.7141 | 0.9992 | 0.4287 | 0.9994 | 0.6000 | 3 | 5052 |
| jpeg_30 | 0.209 | 0.6971 | 0.7625 | 0.9981 | 0.5269 | 0.9982 | 0.6897 | 9 | 4184 |
| blur_0.5 | 0.500 | 0.8444 | 0.8679 | 0.9667 | 0.7836 | 0.9522 | 0.8655 | 239 | 1914 |
| blur_0.5 | 0.209 | 0.8754 | 0.8853 | 0.9501 | 0.8497 | 0.9210 | 0.8971 | 395 | 1329 |
| blur_1.0 | 0.500 | 0.8338 | 0.8547 | 0.9517 | 0.7794 | 0.9300 | 0.8569 | 350 | 1951 |
| blur_1.0 | 0.209 | 0.8657 | 0.8716 | 0.9335 | 0.8504 | 0.8928 | 0.8900 | 536 | 1323 |
| blur_2.0 | 0.500 | 0.7397 | 0.7333 | 0.8221 | 0.7562 | 0.7105 | 0.7878 | 1447 | 2156 |
| blur_2.0 | 0.209 | 0.7541 | 0.7250 | 0.7944 | 0.8299 | 0.6200 | 0.8118 | 1899 | 1504 |
| scale_0.5 | 0.500 | 0.8334 | 0.8527 | 0.9468 | 0.7832 | 0.9222 | 0.8573 | 389 | 1917 |
| scale_0.5 | 0.209 | 0.8588 | 0.8617 | 0.9218 | 0.8513 | 0.8721 | 0.8851 | 639 | 1315 |
| scale_0.25 | 0.500 | 0.8308 | 0.8581 | 0.9686 | 0.7598 | 0.9564 | 0.8516 | 218 | 2124 |
| scale_0.25 | 0.209 | 0.8622 | 0.8743 | 0.9471 | 0.8307 | 0.9180 | 0.8851 | 410 | 1497 |
| noise_0.02 | 0.500 | 0.7525 | 0.7919 | 0.9454 | 0.6502 | 0.9336 | 0.7705 | 332 | 3093 |
| noise_0.02 | 0.209 | 0.7801 | 0.8023 | 0.9156 | 0.7225 | 0.8822 | 0.8077 | 589 | 2454 |
| noise_0.05 | 0.500 | 0.6884 | 0.7461 | 0.9537 | 0.5384 | 0.9538 | 0.6883 | 231 | 4082 |
| noise_0.05 | 0.209 | 0.7387 | 0.7787 | 0.9355 | 0.6347 | 0.9226 | 0.7563 | 387 | 3230 |
| noise_0.10 | 0.500 | 0.6614 | 0.7147 | 0.9083 | 0.5229 | 0.9066 | 0.6637 | 467 | 4219 |
| noise_0.10 | 0.209 | 0.7270 | 0.7561 | 0.8922 | 0.6514 | 0.8607 | 0.7530 | 696 | 3083 |
| color_-0.20 | 0.500 | 0.8506 | 0.8478 | 0.9035 | 0.8577 | 0.8379 | 0.8800 | 810 | 1258 |
| color_-0.20 | 0.209 | 0.8487 | 0.8282 | 0.8667 | 0.9020 | 0.7545 | 0.8840 | 1227 | 867 |
| color_0.20 | 0.500 | 0.8015 | 0.8235 | 0.9310 | 0.7445 | 0.9024 | 0.8274 | 488 | 2259 |
| color_0.20 | 0.209 | 0.8418 | 0.8426 | 0.9056 | 0.8400 | 0.8451 | 0.8716 | 774 | 1415 |
| crop_0.80 | 0.500 | 0.7313 | 0.7822 | 0.9682 | 0.5991 | 0.9652 | 0.7402 | 174 | 3545 |
| crop_0.80 | 0.209 | 0.7816 | 0.8156 | 0.9520 | 0.6931 | 0.9382 | 0.8022 | 309 | 2714 |

## Error analysis

- The weakest ranking condition is `blur_2.0` with ROC AUC 0.8151; this is consistent with strong blur removing or distorting high-frequency forensic cues.
- At the frozen calibrated threshold, clean false positives/false negatives are 363/1322; under `blur_2.0` they become 1899/1504.
- Calibration improves clean balanced accuracy from 0.8703 to 0.8889, but under `blur_2.0` it changes from 0.7333 to 0.7250. The lower threshold recovers more synthetic images but also increases false positives, so it is not a universal fix for severe blur.
- JPEG AUC remains high even when fixed-threshold recall falls, indicating score calibration shift rather than a complete loss of ranking ability.
- The submission interface should continue to emit confidence scores. The calibrated threshold is for binary demo decisions and is not baked into the score output.
- Representative high-confidence errors are listed below and fully exported in `error_cases.csv`.

### clean: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160993.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163818.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161581.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161863.jpg` — score 0.9980

### clean: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0005
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0005
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/129c9496f9fb40acffb64cbdafcdb889.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/c5097ad662b67c9af001724e3a102d3c.jpg` — score 0.0007

### jpeg_90: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9985
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159274.jpg` — score 0.9937
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161863.jpg` — score 0.9917
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160377.jpg` — score 0.9868
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163762.jpg` — score 0.9868

### jpeg_90: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/0763e21106d9bb5adb44fafdebba1337.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0008

### jpeg_70: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9878
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159274.jpg` — score 0.9819
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163785.jpg` — score 0.9780
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160377.jpg` — score 0.9600
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161332.jpg` — score 0.9097

### jpeg_70: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0004
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/0763e21106d9bb5adb44fafdebba1337.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/c5097ad662b67c9af001724e3a102d3c.jpg` — score 0.0007

### jpeg_50: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159274.jpg` — score 0.9937
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161332.jpg` — score 0.7734
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160377.jpg` — score 0.7661
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img158984.jpg` — score 0.7383
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.6982

### jpeg_50: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0004
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/c73db9e07daad5c11b30a46817c09991.jpg` — score 0.0005
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/2bb2b3d160e5311788624d273b772f40.jpg` — score 0.0005
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/ed4740ee61bd78c461b59c8d1ec73304.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/edc0ae92b84620fd57f2a72508c01e67.jpg` — score 0.0006

### jpeg_30: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163785.jpg` — score 0.8960
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159698.jpg` — score 0.7393
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159014.jpg` — score 0.6919
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160377.jpg` — score 0.4492
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161332.jpg` — score 0.4180

### jpeg_30: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/f104d8a186f1cc92ce6b17b5d69506bd.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/0a91eb995bc069847ae7f2bbe17f0af5.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/aca2372360c004ec13e7858d32c2e6bd.jpg` — score 0.0003

### blur_0.5: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160993.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163818.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162600.jpg` — score 0.9980
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161863.jpg` — score 0.9980

### blur_0.5: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0005
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/129c9496f9fb40acffb64cbdafcdb889.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0007

### blur_1.0: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162600.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163762.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159852.jpg` — score 0.9985
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161617.jpg` — score 0.9985

### blur_1.0: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/129c9496f9fb40acffb64cbdafcdb889.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a6fee19a25b72d87e762b8b5ed5986cf.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0010

### blur_2.0: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162986.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163583.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160451.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160876.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161489.jpg` — score 0.9995

### blur_2.0: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a6fee19a25b72d87e762b8b5ed5986cf.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/129c9496f9fb40acffb64cbdafcdb889.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0012

### scale_0.5: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162600.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163762.jpg` — score 0.9985
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162959.jpg` — score 0.9985
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159612.jpg` — score 0.9980

### scale_0.5: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/129c9496f9fb40acffb64cbdafcdb889.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a6fee19a25b72d87e762b8b5ed5986cf.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0009

### scale_0.25: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9961
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162494.jpg` — score 0.9956
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162276.jpg` — score 0.9912
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162283.jpg` — score 0.9907
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161393.jpg` — score 0.9902

### scale_0.25: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0006
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/bc00fc97be21aeb32cf1b02135f5dbed.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/129c9496f9fb40acffb64cbdafcdb889.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a6fee19a25b72d87e762b8b5ed5986cf.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0008

### noise_0.02: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159274.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162925.jpg` — score 0.9980
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161127.jpg` — score 0.9980
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159852.jpg` — score 0.9971

### noise_0.02: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/ecc2b25599ad985c691ba6ab332222d5.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/ce6f87ad22b653aa7bdc1c8636a9800e.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/10fdc474e8f781bc73079bb4894fbf1b.jpg` — score 0.0004

### noise_0.05: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159274.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159099.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162925.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159599.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162844.jpg` — score 0.9990

### noise_0.05: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0004
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0004
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/0763e21106d9bb5adb44fafdebba1337.jpg` — score 0.0004

### noise_0.10: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159099.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159274.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162844.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162937.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161309.jpg` — score 0.9995

### noise_0.10: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/a4c1a07ecb0a5cd6f2ca29572120f434.jpg` — score 0.0004
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/5beae974af6ddc04f197246d47818463.jpg` — score 0.0005
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0005

### color_-0.20: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img160671.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162572.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161559.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163770.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163244.jpg` — score 0.9995

### color_-0.20: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/0763e21106d9bb5adb44fafdebba1337.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/f3969cbaa698d69874ad67eda89fc187.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/e0913d5b9eb0a2c74bc377d336f276c1.jpg` — score 0.0012
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231103102150b17aad067ad7e034/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0018

### color_0.20: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163818.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159707.jpg` — score 1.0000
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161863.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162560.jpg` — score 0.9990

### color_0.20: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/be72f166363120651f31542d4daed408.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/18f690beb924721883a47f1802e44fde.jpg` — score 0.0007
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/50bcdb27830ca8be95298802a0162ba9.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/623c0a74ec9139e8c582f68a1925411a.jpg` — score 0.0008
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/8148d0b6ad70932b3f6c4ec560e8c152.jpg` — score 0.0009

### crop_0.80: false positive

- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159953.jpg` — score 0.9995
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img163864.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img162473.jpg` — score 0.9990
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img161722.jpg` — score 0.9971
- `WildFake_demo/Images/Real/coco/coco2017/val2017/img159852.jpg` — score 0.9946

### crop_0.80: false negative

- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311070924598319e76b1a88ba6c/d1c317f2429de20b96a2effdeab4215b.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/20231102143933b82206831d45b85d/ecc2b25599ad985c691ba6ab332222d5.jpg` — score 0.0002
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/2023110215025084768300d30fc34f/0f92f8331c3c80beee1676140dd006a3.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/044f4f988518587886dc78b7d1da2b45.jpg` — score 0.0003
- `WildFake_demo/Images/Diffusion_based/DALLE/Advanced/DALLE3/dalle3/202311011943129901ca391019566e/fbc64c5d00016f7d60ce41c6e055a9a3.jpg` — score 0.0003

## Calibration details

- Objective: mean condition-balanced accuracy; ties use worst condition then proximity to 0.5
- Mean internal balanced accuracy: 0.8873
- Worst internal balanced accuracy: 0.8483

The WildFake demonstration split is a reference benchmark only and does not contribute to the final score.

## Frozen threshold 0.209 overall summary (all 16 conditions)

The threshold `0.209` was frozen on the internal validation split and is NOT recalibrated
on WildFake. Applied unchanged to all 16 WildFake conditions (13,841 images each):

- Mean balanced accuracy across the 16 conditions: **0.8341**.
- Best condition: `jpeg_90` (balanced accuracy 0.9143); worst: `blur_2.0` (0.7250).
- `clean`: recall 0.8505, specificity 0.9274, balanced accuracy 0.8889, FP/FN = 363/1322.
- Total errors over the 16 conditions: FP = 8,431, FN = 32,038. FN dominate under strong
  JPEG/noise conditions where scores compress below 0.209 (e.g. `jpeg_30` recall 0.5269,
  `noise_0.10` recall 0.6514 at 0.209) while ranking AUC stays comparatively high.
- Per-condition balanced accuracy at 0.209: jpeg_90 0.9143, jpeg_70 0.8962, clean 0.8889,
  blur_0.5 0.8853, scale_0.25 0.8743, blur_1.0 0.8716, scale_0.5 0.8617, color_0.20 0.8426,
  jpeg_50 0.8417, color_-0.20 0.8282, crop_0.80 0.8156, noise_0.02 0.8023, noise_0.05 0.7787,
  jpeg_30 0.7625, noise_0.10 0.7561, blur_2.0 0.7250.

## Score distribution drift observation (old multisource vs final blur_finetune)

Observation only; no threshold is re-selected anywhere. Both models' per-image scores are
compared on the same held-out splits.

Internal validation (validation_multisource.csv, five calibration conditions, real n=7000 /
fake n=9000 per condition):

| Condition | Class | Old mean | New mean | Errors @ 0.209 old -> new |
|---|---|---:|---:|---|
| clean | real | 0.0577 | 0.0534 | FP 542 -> 488 |
| clean | fake | 0.8202 | 0.8339 | FN 1245 -> 1177 |
| blur_2.0 | real | 0.1169 | 0.1054 | FP 1066 -> 958 |
| blur_2.0 | fake | 0.7814 | 0.8010 | FN 1360 -> 1341 |
| jpeg_30 | real | 0.0147 | 0.0116 | FP 117 -> 94 |
| jpeg_30 | fake | 0.7790 | 0.7711 | FN 944 -> 1082 |
| scale_0.25 | real | 0.1089 | 0.1014 | FP 1025 -> 953 |
| scale_0.25 | fake | 0.8058 | 0.8055 | FN 1132 -> 1250 |
| noise_0.10 | real | 0.1012 | 0.0941 | FP 912 -> 870 |
| noise_0.10 | fake | 0.7599 | 0.7449 | FN 753 -> 1004 |

WildFake held-out per-image scores (conditions where both models have predictions):

| Condition | Class | Old mean / median | New mean / median |
|---|---|---|---|
| clean | real | 0.0639 / 0.0028 | 0.0552 / 0.0017 |
| clean | fake | 0.7648 / 0.9648 | 0.7731 / 0.9814 |
| blur_2.0 | real | 0.3674 / 0.1788 | 0.3051 / 0.0552 |
| blur_2.0 | fake | 0.7400 / 0.9531 | 0.7464 / 0.9717 |

Interpretation: the fine-tuned model pushes real-image scores lower (especially under
`blur_2.0`, real median 0.1788 -> 0.0552) and fake-image scores slightly higher, so the
frozen 0.209 threshold transfers without recalibration. The trade-off is concentrated on
the fake side under heavy JPEG/noise, where more fake scores fall below 0.209.

## Conclusions

- blur_2.0 remains the weakest condition (ROC AUC 0.8151), as expected, though the blur
  fine-tune already improved it materially versus the previous multisource model
  (0.7834 -> 0.8151 AUC on WildFake; real-score drift confirms the fix direction).
- The secondary weakness is additive noise: `noise_0.10` now has the lowest noise-family
  AUC (0.8576) and, notably, slightly below `noise_0.02` (0.8598), with recall at 0.209
  falling to 0.6514. Noise robustness regressed a little relative to the old model
  (noise_0.10 AUC 0.8780 -> 0.8576) while blur robustness improved.
- Strong JPEG compression causes score compression below the frozen threshold (jpeg_30
  recall 0.5269 at 0.209) even though AUC stays high (0.9808); ranking ability is retained
  and the binary decision layer is where the loss occurs.
- Recurring hard negatives: a small set of COCO val2017 images (e.g. `img159953.jpg`,
  `img159274.jpg`, `img160377.jpg`, `img161332.jpg`, `img161863.jpg`, `img163818.jpg`)
  appear as high-confidence false positives across nearly all conditions, and a handful of
  DALL·E 3 Advanced images (e.g. `8148d0b6...`, `a4c1a07e...`, `50bcdb27...`,
  `129c9496...`) are missed with scores < 0.001 under almost every condition. These are
  content-driven errors, not degradation-driven ones.
- The 0.209 threshold remains valid for the final submission model; no WildFake-based
  recalibration was performed.
