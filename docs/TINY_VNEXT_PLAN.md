# Tiny vNext preregistered optimization plan

Status: completed and frozen on `tiny-vnext` (2026-08-30). The selected
checkpoint is seed 2026 at residual gain 1.60. The `v1.0.0` release and its
checkpoint remain immutable rollback artifacts. See `TINY_VNEXT_RESULTS.md`.

## Evidence policy

- The historical 12,000-image selection set is a regression anchor only.
- The historical 16,000-image confirmation set and WildFake observation are
  consumed and cannot select Tiny vNext.
- Development selection uses source-disjoint SuSy validation and MS COCOAI
  validation after exact-image, perceptual-hash, and content-group audits.
- Final test data remains sealed until the Tiny recipe and optional Base recipe
  are frozen. Threshold calibration happens only after checkpoint selection.

## Data policy

- SuSy: all six published sources are retained by user decision, including
  `midjourney_tti`; provenance and upstream license status remain recorded.
- MS COCOAI: train and validation are pinned to revision
  `787334f7857fa54f29027a7f09c30e895ad486ef`. The dataset card does not declare
  a license; it is retained by explicit user decision and marked accordingly.
- New manifests must keep official train/development roles, remove any shared
  prompt/content groups from development, and reject exact overlap with prior
  repository manifests.
- Training batches are source-, label-, and generator-balanced. Raw dataset
  size must not determine the gradient contribution.

## Ordered experiments

1. Evaluate frozen Adapter v2 at gain 1.0, Adapter v1 at gain 0.60, and Adapter
   v2 at gain 0.87 on the fresh development manifests. No training occurs.
2. Train one frozen-backbone adapter with BCE on CommunityForensics plus the new
   modern sources, and squared residual preservation on GenImage and SID_Set.
3. Sweep residual gain after training; allow one targeted penalty/exposure retry
   only when the failure identifies a single mechanism.
4. If the single-scale adapter plateaus, replace it with a frozen multi-scale
   stage-2 + stage-3 adapter. Selective stage-3 unfreezing is the final option.

## Training exposure target

- clean 20%
- JPEG 12%
- blur 24% (blur 2.0 approximately 14%)
- scale 10%
- noise 18% (noise 0.10 approximately 6%)
- color 8% (negative/positive 4% each)
- crop 8%
- label-independent re-encoding probability 0.15

## Acceptance gates

A candidate is eligible only when all of the following hold:

1. Fresh modern-generator macro robust score improves by at least 0.005.
2. The paired/grouped bootstrap 95% confidence lower bound is above zero.
3. Worst-generator robust score and worst condition do not regress.
4. Historical CF, GenImage, and SID robust-score drops are each at most 0.002.
5. Any historical source-by-degradation-family drop is at most 0.003.
6. Clean AUC drop is at most 0.002.
7. Label-blind format diagnostic drop is at most 0.002 versus `v1.0.0`.
8. Single-pass batch-32 latency overhead is at most 10% and Tiny stays below
   30 million parameters.
9. The winning recipe passes a second seed with the same improvement direction.

Failure preserves `v1.0.0`; no candidate overwrites release paths before all
gates, sealed evaluation, calibration, and delivery verification complete.
