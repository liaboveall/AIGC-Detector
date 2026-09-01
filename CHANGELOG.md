# Changelog

## v2.0.0 — 2026-09-01

Promote Ensemble vNext to the public default release.

### Model

- Freeze a 0.50/0.50 FP32 logit ensemble of Tiny vNext and Base v1.
- Package both states and provenance in one 115,585,507-parameter Git LFS asset.
- Record SHA-256
  `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`.
- Keep the v1.0.0 Adapter v2 asset as the rollback release.

### Evidence

- Historical robust score: `0.944886 -> 0.978314`.
- Modern global robust score: `0.740077 -> 0.901913`; all 16 condition AUCs improve.
- Modern generator-macro robust score: `0.718331 -> 0.894738`.
- Pass all four modern gates; 1,000-replicate macro-gain 95% CI
  `[0.171015, 0.182232]`.
- Do not reopen the consumed confirmation split or WildFake observation.

### Runtime and delivery

- Add ensemble-aware checkpoint loading to prediction and evaluation.
- Add live alpha sweep, packaging, generator-macro bootstrap, latency, release smoke,
  and direct/package equivalence tooling.
- Promote member logits to FP32 before blending under CUDA autocast.
- Switch the branch default inference asset to Ensemble vNext and update current-facing
  documentation without rewriting historical `v1.0.0` release notes.

## v1.0.0 — 2026-08-29

Formal TikTok TechJam Track 5 delivery release.

### Model

- Freeze Adapter v2 (`28,018,018` parameters) as the formal checkpoint.
- Publish checkpoint SHA-256
  `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`.
- Preserve the previous ConvNeXt-Tiny base as a rollback point.

### Performance evidence

- Internal 12,000-image, 16-condition robust score: `0.930488 -> 0.942425`.
- CommunityForensics robust score: `0.903910 -> 0.928369` while GenImage and SID
  robust-score drops remain below `0.0006`.
- Pass all 31 pre-registered acceptance checks.
- One-time post-freeze WildFake robust score: `0.904694 -> 0.908171`; all 16 AUC and
  balanced-accuracy conditions are non-decreasing versus the base.

### Code and delivery

- Add adapter-aware loading to training, evaluation, prediction, and audit paths while
  retaining legacy-checkpoint compatibility.
- Add modern-generator acquisition, balanced replay, distillation, interpolation,
  threshold, leakage, and compression-history tooling.
- Add a self-contained release verifier for checkpoint hash, architecture, parameter
  count, unreadable-image behavior, and exact JSON schema.
- Make unit/pipeline tests runnable without committing image datasets.
- Pin the verified top-level dependency versions.
- Replace historical submission documents with final Adapter v2 metrics, limitations,
  evidence boundaries, and demo instructions.
