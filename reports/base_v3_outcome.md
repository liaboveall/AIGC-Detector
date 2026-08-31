# Base v3 controlled restart outcome

Date: 2026-08-31
Branch: `base-v3`
Preregistration: `docs/BASE_V3_PREREGISTRATION.md`

## Verdict

Base v3 completed all four preregistered epochs normally, but no epoch passed
the unchanged Tiny-relative 16-condition selection gates. Therefore Base v3 is
rejected as a deployable replacement. Tiny vNext remains the frozen release
model. Confirmation and WildFake were not opened, because the selection gate
was not passed.

Accepted model (unchanged):

- `outputs/tiny_vnext/final_candidate/tiny_vnext_seed2026_gain1p60.pt`
- SHA-256: `1AF51D00022B9CD3FABD58D65F01C7F728F6F99C2649AAA86ACDBAA9789EDE44`

Best Base v3 diagnostic checkpoint by historical robust score:

- `outputs/base_v3/seed2026/epoch_04.pt`
- SHA-256: `6C4BEBBC013D0D5D1A30BC4F08734901BA123E30BE60377AF19EAA313937B4C6`
- Size: 353,576,137 bytes
- This checkpoint is not approved for prediction or release.

## Reproducibility and safety checks

- Training manifest: 280,000 rows, exactly 140,000 real / 140,000 fake.
- Training/selection SHA-256 overlap: 0.
- Missing training files: 0.
- Quota sampler: 28,000 batches per epoch; every physical batch has one real
  and one fake from CF, GenImage, SID and the modern pool.
- Modern fake pool: 10 generator groups of 2,800 rows each, strict round-robin.
- Sixteen repository tests passed before the run.
- All four phase smoke tests passed; full-run peak allocated GPU memory was
  2.655 GiB on the final epoch.
- Training ended normally; no non-finite loss or divergence stop occurred.
- `main`, Tiny vNext, Base v1/v2 and all earlier best checkpoints were not
  modified.

## Selection results

All scores use the same fixed 12,000-image historical selection manifest and
the registered 16-condition suite.

| Candidate | Overall robust | CF robust | GenImage robust | SID robust | Decision |
|---|---:|---:|---:|---:|---|
| Tiny vNext baseline | 0.944886 | 0.935608 | 0.918252 | 0.958064 | accepted baseline |
| Base v3 epoch 1 | 0.916812 | 0.962727 | 0.834936 | 0.887406 | rejected |
| Base v3 epoch 2 | 0.938522 | 0.962724 | 0.892042 | 0.915059 | rejected |
| Base v3 epoch 3 | 0.938847 | 0.958947 | 0.900346 | 0.919047 | rejected |
| Base v3 epoch 4 | 0.942558 | 0.957507 | 0.913184 | 0.927779 | rejected |

Epoch 4 was closest, but relative to Tiny it changed:

- overall robust: -0.002328;
- CF robust: +0.021899;
- GenImage robust: -0.005068 (allowed source drop: 0.003);
- SID robust: -0.030285 (allowed source drop: 0.003);
- clean AUC: +0.006760.

It also failed the registered global blur/scale/noise guards, multiple
GenImage and SID degradation-family guards, the noise guard, and the
`blur_2.0` guard. Complete machine-readable evidence is in:

- `outputs/base_v3/seed2026/tiny_relative_gates.json`
- `outputs/base_v3/seed2026/tiny_relative_gates.txt`
- `outputs/base_v3/seed2026/metrics_epoch_01.json` through `metrics_epoch_04.json`

## Interpretation

The run successfully tested the strongest single-Base hypothesis that had not
been covered by Base v1/v2: exact per-batch replay quotas, label-independent
re-encoding, Tiny logit preservation, cross-architecture feature alignment,
and progressive shallow-layer unfreezing from clean ImageNet-pretrained Base
weights. The result improves historical domains monotonically after epoch 1,
but the final SID gap remains an order of magnitude larger than the acceptance
budget. The clean and CF gains alongside degraded old-domain losses show that
capacity is not the limiting factor; the Base representation still replaces
robust Tiny forensic cues with source/quality-sensitive cues.

Continuing the same run would violate the preregistered four-epoch protocol and
would use the selection set adaptively. A future Base experiment should be a
new preregistered study, preferably using function-preserving initialisation or
representation transfer that starts Base from Tiny-compatible features rather
than relying on loss-only cross-architecture distillation.

## Final disposition

- Deployable model: Tiny vNext, unchanged.
- Base v3: archived diagnostic, not promoted.
- Confirmation: sealed.
- WildFake: not re-opened.
- Threshold calibration: not repeated because the deployable model did not
  change.
