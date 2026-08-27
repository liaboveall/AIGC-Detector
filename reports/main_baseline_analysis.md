# Main ConvNeXt baseline analysis

Date: 2026-08-26  
Checkpoint: `outputs/baseline/best.pt`  
Best epoch: 8  
Model: ImageNet-pretrained ConvNeXt-Tiny  
Parameters: 27,820,897

## Training and in-domain validation

The model trained on 202,820 clean SID_Set training images. Checkpoint selection
used a fixed, stratified 6,000-image subset of SID_Set validation under clean,
JPEG 30, blur 2.0, scale 0.25, and Gaussian noise 0.10 conditions.

| Epoch | Train loss | Clean AUC | Mean selected-degradation AUC | Worst selected-degradation AUC | Robust score |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3699 | 0.9365 | 0.9215 | 0.9152 | 0.9202 |
| 2 | 0.3626 | 0.9506 | 0.9361 | 0.9313 | 0.9351 |
| 3 | 0.3124 | 0.9507 | 0.9349 | 0.9280 | 0.9335 |
| 4 | 0.2675 | 0.9606 | 0.9471 | 0.9443 | 0.9465 |
| 5 | 0.2209 | 0.9692 | 0.9589 | 0.9567 | 0.9584 |
| 6 | 0.1727 | 0.9773 | 0.9696 | 0.9672 | 0.9691 |
| 7 | 0.1316 | 0.9787 | 0.9708 | 0.9694 | 0.9706 |
| 8 | 0.1135 | **0.9797** | **0.9730** | **0.9722** | **0.9728** |

All selection metrics improved through epoch 8. There is no in-domain validation
rollback, and epoch 8 is the correct saved checkpoint.

The complete 16-condition suite on the same fixed validation sample produced:

- Mean degraded AUC: **0.9732**
- Worst degraded AUC: **0.9666** (`noise_0.02`)
- Robust score: **0.9719**

These results establish a strong **within-SID baseline**.

## Cross-dataset demo evaluation

The frozen checkpoint was evaluated, without training or checkpoint selection, on
all 13,841 WildFake demo images: 4,998 COCO real images and 8,843 DALL-E Advanced
full-synthetic images.

| Condition | AUC | Average precision | Accuracy at 0.5 | Real specificity | Fake recall |
|---|---:|---:|---:|---:|---:|
| clean | 0.6463 | 0.7922 | 0.5859 | 0.6471 | 0.5513 |
| JPEG 50 | 0.6985 | 0.8210 | 0.6264 | 0.6967 | 0.5867 |
| blur 1.0 | **0.5661** | 0.7437 | 0.5300 | 0.5934 | 0.4942 |
| scale 0.5 | 0.5763 | 0.7509 | 0.5419 | 0.5672 | 0.5275 |

Exact values are stored in
`outputs/baseline/evaluation_wildfake_demo_quick.json`.

## Decision

The model is robust to image degradations **within SID_Set**, but it does not yet
generalize across data sources. The low demo AUC cannot be repaired by selecting a
different threshold. The improvement under JPEG also suggests that source/format
statistics are influencing the predictions.

This checkpoint is frozen as the ImageNet ConvNeXt within-domain baseline, not as
the final detector. Do not use the SID evaluation pool yet. The next iteration must:

1. add high-resolution, multi-generator fake training data and matched real data;
2. keep WildFake demo isolated from training and checkpoint selection;
3. introduce a CLIP-pretrained semantic backbone for stronger cross-source features;
4. compare against this checkpoint using the same SID validation and demo tracks;
5. add an NPR/pixel-residual branch only after the cross-source baseline is working.
