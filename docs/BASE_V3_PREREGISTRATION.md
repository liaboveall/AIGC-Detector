# Base v3 preregistration

Status: frozen before implementation and training on 2026-08-31.

## Objective

Train one deployable ConvNeXt-Base detector from the declared ImageNet-22K to
ImageNet-1K pretrained weights. Base v1/v2 checkpoints are diagnostics only and
must not initialise Base v3. The accepted Tiny vNext checkpoint is a frozen
teacher and the comparison baseline; it is never overwritten.

## Data and batch contract

- Train manifest: `Dataset/manifests/tiny_vnext_train_balanced_280000.csv`.
- Historical selection: `validation_modern_combined_selection_12000.csv`, all
  16 registered conditions.
- Physical batch size 8, gradient accumulation 4 (effective batch size 32).
- Every physical batch contains exactly one real and one fake image from each
  of CommunityForensics-Small, GenImage, SID_Set, and the combined modern pool.
- The modern real item comes from SuSy. Modern fake generators are shuffled
  independently and sampled round-robin across generator groups from SuSy and
  MS-COCOAI. An epoch contains 28,000 quota batches (224,000 examples).
- Label-independent JPEG/WebP re-encoding probability is 0.35. The existing
  six degradation families remain enabled with the Base v1 weights.

## Optimisation phases

Base starts from `convnext_base.fb_in22k_ft_in1k` pretrained weights.

1. Phase A, one epoch: train head and stage 3; freeze stem and stages 0-2.
2. Phase B, two epochs: train head and stages 2-3; freeze stem and stages 0-1.
3. Phase C, one epoch: train all layers with layer-wise learning rates.

Learning rates in Phase C are stem/stage0 `5e-7`, stage1 `1e-6`, stage2
`3e-6`, stage3 `1e-5`, head `1e-4`. Earlier phases use the same rates for
their enabled groups. Each phase builds a fresh AdamW optimiser; the global
schedule uses five percent warm-up followed by cosine decay within the phase.
AMP, gradient clipping 1.0 and weight decay 0.05 are mandatory.

## Loss contract

- Supervised BCE is computed on every image.
- On GenImage and SID_Set only, preserve Tiny decisions with exact-logit Huber
  distillation (`delta=2.0`). Modern images never receive Tiny distillation.
- On the same historical mask, align L2-normalised pre-logit features. A
  training-only linear projection maps Base 1024-d features to the Tiny 768-d
  feature space. It is not stored in the deployable model checkpoint.
- Phase A uses BCE + `0.75 * logit_KD`; feature loss is disabled.
- Phases B/C use BCE + `0.75 * logit_KD + 0.10 * feature_cosine`.

## Selection and stopping

- Epochs 1-2 are burn-in. Old-domain gate failure during burn-in is recorded
  but cannot stop training.
- Stop only for non-finite loss, unreadable data, invalid quota batches, or a
  clear divergence (`GenImage` or `SID_Set` source robust drop greater than
  0.05 after epoch 2).
- Every epoch is saved. Candidates are ranked only after the full historical
  12,000-image, 16-condition evaluation.
- A deployable replacement must pass the existing strict Tiny-relative gate
  script without threshold changes. The modern selection score is used only
  after historical eligibility is established.
- Confirmation and WildFake remain sealed until a candidate passes selection.
  Confirmation is opened once; WildFake is observational and never feeds back
  into model choice.

## Safety and fallback

All outputs go to `outputs/base_v3/`. `main`, Tiny v1.1.0-rc1, Base v1, Base
v2, and every earlier `best.pt` are read-only. If no Base v3 epoch passes the
strict gates, Tiny vNext remains the deployable model and the proven fixed
Tiny/Base blend remains a diagnostic fallback, not a single-model release.
