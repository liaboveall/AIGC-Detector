# Smoke baseline analysis

Date: 2026-08-26  
Model: ImageNet-pretrained ConvNeXt-Tiny  
Parameters: 27,820,897  
Training data: balanced 8,000-image SID_Set smoke split  
Validation data: balanced 1,600-image SID_Set smoke validation split

## Training result

| Epoch | Train loss | Clean AUC | Mean quick-degraded AUC | Worst quick-degraded AUC | Robust score |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5695 | 0.8921 | 0.8909 | 0.8887 | 0.8904 |
| 2 | 0.3349 | 0.9412 | 0.9425 | 0.9404 | 0.9421 |

The best checkpoint is epoch 2. The loss and every selection metric improved, so
there is no evidence of smoke-scale convergence yet.

## Full deterministic robustness suite

The full suite covers clean images plus 15 degraded conditions. Its aggregate
results are:

- Mean degraded AUC: **0.9388**
- Worst degraded AUC: **0.9271**
- Robust score (`0.8 * mean + 0.2 * worst`): **0.9364**

| Condition | Overall AUC | Accuracy | Real specificity | Full-synthetic recall | Tampered recall | Tampered-vs-real AUC |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.9412 | 0.8650 | 0.8788 | 0.9925 | 0.7100 | 0.8880 |
| jpeg 30 | 0.9401 | 0.8650 | 0.8813 | 0.9900 | 0.7075 | 0.8863 |
| blur 2.0 | **0.9271** | 0.8419 | 0.9213 | 0.9800 | **0.5450** | **0.8607** |
| scale 0.25 | 0.9340 | 0.8538 | 0.9200 | 0.9850 | 0.5900 | 0.8738 |
| noise 0.10 | 0.9352 | 0.8594 | 0.9138 | 0.9775 | 0.6325 | 0.8779 |
| color +20% | 0.9325 | 0.8531 | 0.8588 | 0.9900 | 0.7050 | 0.8726 |
| crop 80% | 0.9411 | 0.8563 | 0.8988 | 0.9800 | 0.6475 | 0.8890 |

The exact values for every condition are stored in
`outputs/baseline_smoke/evaluation_full.json`.

## Interpretation

1. Full-synthetic detection is already strong: clean full-synthetic recall is
   99.25% and its clean contrast AUC against real images is 0.9945.
2. Partial tampering is the dominant error source. Clean tampered recall is 71.0%,
   falling to 54.50% under strong blur.
3. Strong blur and downscaling erase localized manipulation evidence, while JPEG
   compression causes almost no additional decline at the tested qualities.
4. Overall AUC hides the tampered weakness because full-synthetic examples are much
   easier. Future reports must retain the source-contrast and per-class recall rows.

## Main-training decision

Proceed to the 202,820-image SID_Set main training run. Checkpoint selection is
updated to use clean, JPEG 30, blur 2.0, scale 0.25, and noise 0.10. Training
degradation probability is increased from 0.60 to 0.70. These are evidence-driven
changes from the smoke experiment, not tuned against the final evaluation split.

The evaluation pool and WildFake demo remain unused for training and checkpoint
selection.
