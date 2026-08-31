# Base v1 preregistered comparison plan

Status: active on `base-v1`. Tiny vNext and `v1.1.0-rc1` are immutable rollback
artifacts; no Base experiment may overwrite their paths or tags.

## Hypothesis

ConvNeXt-Base may improve cross-generator and worst-condition ranking because it
has more representational capacity than Tiny. Scale alone is not assumed to be
beneficial: Base must beat Tiny vNext under the same audited data and evaluation
protocol before it can be selected.

## Fixed recipe

- Student: `convnext_base.fb_in22k_ft_in1k`, 87,567,489 parameters.
- Initialisation: timm ImageNet-22K -> ImageNet-1K pretrained weights; Tiny
  weights are structurally incompatible and are never loaded into Base.
- Trainable: head plus stages 2-3; stem and stages 0-1 remain frozen.
- Data: the frozen 280,000-row Tiny vNext balanced manifest.
- Loss: label BCE on every source plus temperature-2 binary KD with weight 0.5
  on GenImage and SID_Set only.
- Teacher: frozen Tiny vNext seed-2026/gain-1.60 checkpoint.
- Augmentation and label-independent re-encoding exactly match Tiny vNext.
- Two epochs, effective batch 32 (physical 16, accumulation 2), one warmup
  epoch; head LR 2e-4 and stages 2-3 LR 1e-5.

## Selection and stop-loss

The first full recipe and at most one mechanism-specific retry are allowed.
Modern selection uses only the fresh 12,896-image SuSy/MS-COCOAI development
manifest. Historical validation is a regression anchor. The consumed 16,000
confirmation set and WildFake observation remain sealed from Base selection.

Base is eligible only when all of the following hold versus frozen Tiny vNext:

1. Modern generator-macro robust score improves by at least 0.005.
2. Grouped bootstrap 95% confidence lower bound is above zero.
3. Worst-generator robust score and worst generator-condition AUC do not regress.
4. Historical overall robust score does not regress by more than 0.002.
5. GenImage and SID robust scores each do not regress by more than 0.002.
6. Any historical source-by-degradation-family drop is at most 0.003.
7. Clean AUC and blur-2.0 AUC drops are each at most 0.002.
8. Parameter count remains below 2 billion and the standard prediction smoke
   contract passes.

If the primary recipe fails in several independent families, Base is rejected
without a retry. A retry is allowed only for one diagnosed mechanism and must
change one declared training factor. Failure preserves Tiny vNext as the final
model.
