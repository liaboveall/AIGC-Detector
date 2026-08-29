# Error Analysis — Frozen Adapter v2

**Scope:** formal `v1.0.0` checkpoint, one-time WildFake post-freeze observation,
continuous ROC AUC plus binary behavior at the frozen threshold 0.209.

WildFake contains 4,998 COCO val2017 real images and 8,843 DALL-E 3 Advanced fake
images. It is a demonstration subset rather than a universal cross-generator test.

## Executive diagnosis

The model's ranking is strong on clean and JPEG-compressed images, but three different
failure mechanisms remain:

1. heavy blur destroys or confounds forensic texture cues and raises real-image scores;
2. strong noise and crop reduce fake recall;
3. strong JPEG preserves ranking but shifts the score distribution below the fixed
   operating threshold.

The adapter improves fake recall consistently without materially changing aggregate
false positives, but it does not eliminate content-driven errors that survive every
transformation.

## Binary error profile at threshold 0.209

| Condition | AUC | bACC | Real FPR | Fake recall | Primary failure |
|---|---:|---:|---:|---:|---|
| clean | 0.9647 | 0.8929 | 7.28% | 85.86% | recurring semantic errors |
| jpeg_30 | 0.9811 | 0.7676 | 0.18% | 53.69% | score shift / false negatives |
| blur_2.0 | 0.8221 | 0.7322 | 37.62% | 84.06% | real false positives |
| noise_0.10 | 0.8604 | 0.7630 | 13.95% | 66.55% | both sides degrade |
| crop_0.80 | 0.9112 | 0.8233 | 6.22% | 70.88% | fake false negatives |

Across all 16 conditions, Adapter v2 reduced false negatives from 32,038 to 30,374
(-1,664) while false positives changed from 8,431 to 8,429 (-2). The residual branch
therefore behaves as intended: it recovers modern fake examples while leaving the real
decision boundary nearly unchanged.

Raw operating-point data are tracked in
[`../reports/final_adapter_v2/wildfake_frozen_threshold_table.csv`](../reports/final_adapter_v2/wildfake_frozen_threshold_table.csv).

## Failure mode 1: heavy blur

`blur_2.0` is the lowest-AUC condition and has the largest real-image false-positive
rate. The score change is not a simple global calibration shift: fake recall remains
reasonably high while authentic images move strongly toward the fake side.

Likely mechanism: the ConvNeXt detector relies on local texture, resampling, and
high-frequency synthesis traces. Severe Gaussian blur removes those cues. Some smooth
authentic scenes then resemble the low-detail statistics of generated images.

Implication: changing the global threshold alone cannot repair this condition without
substantially reducing fake recall. Future work needs new representation or training
evidence, such as multi-scale/frequency features or additional authentic blurred data,
and must be validated on a new development split rather than WildFake.

## Failure mode 2: strong noise

At `noise_0.10`, AUC is 0.8604 and balanced accuracy is 0.7630. Noise corrupts both
the real/fake forensic traces and the low-level statistics used by the detector.

The internal selection result improved globally under all noise conditions, but the
non-binding 0.002 stress test found a GenImage noise-family drop of 0.002583 versus the
frozen base. This is small and below the formal 0.005 bound, yet it identifies the
narrowest old-domain robustness margin in the accepted model.

## Failure mode 3: JPEG score-distribution shift

JPEG q30 has excellent ranking (AUC 0.9811) and only 0.18% real false positives, but
fake recall falls to 53.69% at threshold 0.209. Recompression moves many fake scores
downward while preserving their ordering relative to real images.

This distinction matters:

- AUC says the model can still rank real versus fake well.
- Balanced accuracy says the frozen threshold is not optimal for this transformed
  distribution.

The repository still emits continuous probabilities, allowing a deployment to
calibrate its own operating point on a genuinely held-out target domain. We do not
introduce a condition-specific threshold in the competition submission because the
condition may be unknown and such tuning would require new validation data.

## Persistent content-driven errors

Some COCO real images remain high-scoring false positives under many or all conditions,
and some DALL-E images remain low-scoring false negatives. These persistent cases are
less likely to be caused by a particular codec or transformation; they reflect content,
semantic prior, or generator-style mismatch.

The tracked aggregate error-case table is
[`../reports/final_adapter_v2/error_cases.csv`](../reports/final_adapter_v2/error_cases.csv).
It contains dataset-relative identifiers and scores, not redistributed image bodies.

## Threshold decision

Adapter v2's internal threshold scan produced optima around 0.155-0.170, but the gain
over 0.209 was only:

- +0.00035 mean balanced accuracy over all 16 conditions;
- +0.00122 over the historical five-condition calibration subset.

Both changes lie below the estimated sampling-noise floor, while 0.209 preserves the
pre-existing model's operating contract and enables direct base-versus-adapter
comparison. Threshold 0.209 was therefore retained.

## Format-history shortcut audit

The combined historical datasets have a known label/codec association, so Adapter v2
was trained with 15% label-independent JPEG/WebP re-encoding. A paired 3,000-image audit
then compared decoded, JPEG q75, hash-random re-encoding, and neutralized random
re-encoding views.

Candidate-versus-base absolute AUC differences ranged from -0.00004 to -0.00064, with
paired bootstrap intervals strongly overlapping. No detectable shortcut worsening was
found. However, random re-encoding still lowers the candidate by approximately 0.0093
AUC and stronger neutralization by 0.0162, so format invariance is not claimed.

## Interpretation limits

- Selection results are development results and influenced candidate choice.
- The earlier 16,000-image confirmation set was consumed by a rejected model-soup
  candidate and was not reused for Adapter v2.
- WildFake was observed only after model and threshold freeze, but its fake side covers
  one generator family.
- No additional tuning should use the recorded WildFake errors. Further optimization
  requires a new development set and a new untouched final test.
