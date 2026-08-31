# Error Analysis — Ensemble vNext

**Scope:** frozen `alpha=0.50` candidate, historical selection anchor, and
source-disjoint modern development set. The consumed confirmation/WildFake data were
not reopened.

## What the fusion fixed

Tiny vNext's largest modern weaknesses were strong noise, aggressive resizing, crop,
and some newer generator families. The fixed Base-v1 branch provides complementary
ranking evidence:

- generator-macro robust score improved by `0.176407`
- worst-generator robust score improved by `0.247400`
- worst generator-condition AUC improved by `0.325581`
- every global condition improved, with deltas from `+0.058249` to `+0.239058`

This does not mean either member is independently acceptable. The Base-only attempts
were rejected when they violated Tiny-relative historical guards; only the gated
0.50/0.50 composition passed both historical preservation and modern-gain checks.

## Remaining ranking failures

### Strong additive noise

`noise_0.10` is the lowest global ensemble condition at AUC `0.828625`, followed by
`noise_0.05` at `0.848464`. Noise can erase or mimic high-frequency forensic cues
used by both backbones.

### Generator-specific weaknesses

- Midjourney v6 has the lowest generator-level robust score: `0.796370`.
- Midjourney v1/v2 under `noise_0.05` is the lowest generator-condition pair:
  AUC `0.648097`.
- DALL-E 3 under `noise_0.10` remains weak despite a large improvement.

These results argue for new, source-disjoint noisy development data if further work is
authorized. They do not authorize tuning on already observed WildFake examples.

### Compression and acquisition effects

`jpeg_90` is lower than several more aggressive JPEG settings on the modern set.
That non-monotonic pattern suggests interaction with the source files' existing codec
history, not a simple quality-versus-accuracy curve. Camera, resize, and platform
pipelines may similarly shift scores.

## Calibration and threshold risk

The evaluation gates use AUC and ranking-based robust scores. They do not establish
probability calibration. The Adapter v2 threshold `0.209` belongs to a different
checkpoint and must not be reused automatically. Until a new independent calibration
set exists, downstream users should consume the continuous probability and avoid
hard-decision claims.

## Systems trade-offs

The ensemble has 115,585,507 parameters and performs two forward passes. Compared with
Tiny vNext it increases:

- checkpoint size from one compact model to 462,558,035 bytes
- GPU/CPU memory demand
- latency and energy per image
- operational dependency on the ensemble-aware loader

The release verifier therefore checks the single-file schema, source hashes, parameter
count, deterministic repeated inference, exact output keys, neutral unreadable-image
fallback, and CPU/CUDA paths.

## Numerical reproducibility

The live sweep blends member logits in FP32. The packaged model now does the same.
CUDA direct-versus-packaged comparison aligns samples by `(path, condition)` and uses
an absolute probability tolerance of `1e-3` to accommodate cuDNN execution-path
rounding; identities and static fields must match exactly. Aggregate metrics and the
maximum/mean probability differences are recorded in
`reports/ensemble_vnext/packaged_equivalence.json`.

## Interpretation boundary

An AIGC score is not proof of origin. False positives and false negatives can be
systematic for unseen generators, cameras, edits, semantic categories, and geographic
or platform domains. Consequential use requires human review and independent
target-domain validation.
