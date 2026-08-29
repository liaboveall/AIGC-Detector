# Changelog

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
