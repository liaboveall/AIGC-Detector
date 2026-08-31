# Changelog

## Ensemble vNext candidate — 2026-09-01

Completed on branch `ensemble-vnext`; not yet merged, tagged, pushed, or published as
a new public release.

### Model

- Freeze a 0.50/0.50 FP32 logit ensemble of Tiny vNext and Base v1.
- Package both states and provenance in one 115,585,507-parameter Git LFS asset.
- Record SHA-256
  `B3A2002C6C297D6382D88B0BA83C4059CE75BBA01C14049A26D3CF45D074DC4B`.
- Keep the published Adapter v2 asset as the rollback release.

### Evidence

- Historical robust score: `0.944886 -> 0.978314`.
- Modern global robust score: `0.740077 -> 0.901913`; all 16 condition AUCs improve.
- Modern generator-macro robust score: `0.718331 -> 0.894738`.
- Pass all four modern gates; 1,000-replicate macro-gain 95% CI
  `[0.171015, 0.182232]`.
- Do not reopen the consumed confirmation split or WildFake observation.

### Runtime and packaging

- Add ensemble-aware checkpoint loading to prediction and evaluation.
- Add live alpha sweep, packaging, generator-macro bootstrap, latency, release smoke,
  and direct/package equivalence tooling.
- Promote member logits to FP32 before blending under CUDA autocast.
- Switch the branch default inference asset to Ensemble vNext and update current-facing
  documentation without rewriting historical `v1.0.0` release notes.

## v1.0.0 — 2026-08-29

First public release.

### Model

- Freeze Adapter v2 (`28,018,018` parameters) as the release checkpoint.
- Publish checkpoint SHA-256
  `17FE0D53D4264D93485F91BF11E24733A637280324889E2920B168BC1C7999DE`.
- Preserve the previous ConvNeXt-Tiny base as a rollback point.

### Performance evidence

- Internal 12,000-image, 16-condition robust score: `0.930488 -> 0.942425`.
- CommunityForensics robust score: `0.903910 -> 0.928369` while GenImage and SID
  robust-score drops remain below `0.0006`.
- Pass all 31 pre-registered acceptance checks.
- One-time post-freeze WildFake robust score: `0.904694 -> 0.908171`; all 16 AUC and
  balanced-accuracy conditions are non-decreasing versus the base.

### Code and tooling

- Add adapter-aware loading to training, evaluation, prediction, and audit paths while
  retaining legacy-checkpoint compatibility.
- Add modern-generator acquisition, balanced replay, distillation, interpolation,
  threshold, leakage, and compression-history tooling.
- Add a self-contained release verifier for checkpoint hash, architecture, parameter
  count, unreadable-image behavior, and exact JSON schema.
- Make unit/pipeline tests runnable without committing image datasets.
- Pin the verified top-level dependency versions.
- Rewrite the documentation around the final Adapter v2 metrics, limitations,
  evidence boundaries, and usage instructions.
