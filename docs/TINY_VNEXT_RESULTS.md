# Tiny vNext frozen result

Frozen: 2026-08-30 on `tiny-vnext`.

## Decision

The selected model is the single-scale frozen-backbone adapter trained with
seed 2026 and served at a fixed residual gain of 1.60. Multi-scale adaptation
was not opened because the simpler model cleared every preregistered gate.

- Local delivery checkpoint:
  `outputs/tiny_vnext/final_candidate/tiny_vnext_seed2026_gain1p60.pt`
- SHA-256:
  `1AF51D00022B9CD3FABD58D65F01C7F728F6F99C2649AAA86ACDBAA9789EDE44`
- Parameters: 28,018,018
- Frozen binary-demo threshold: 0.209
- Rollback release: `v1.0.0` at commit `574af660`

The checkpoint embeds `adapter.residual_gain: 1.6`; no command-line override is
needed in `predict.py`.

## Data and leakage controls

The training manifest contains 280,000 rows, balanced 140,000 real / 140,000
fake. Source exposure is CommunityForensics 112,000, GenImage 56,000, SID_Set
56,000, SuSy 42,000, and MS-COCOAI 14,000. Ten modern fake strata each receive
2,800 rows. The fresh modern development manifest contains 12,896 unique paths.

Before manifest construction, exact duplicates, train/development exact-image
overlaps, and shared prompt/content groups were removed. The final train/dev
SHA-256 overlap list is empty. Counts are recorded in
`outputs/tiny_vnext/data/manifest_summary.json`.

## Preregistered gate results

### Fresh modern generators

Generator-macro robustness rose from 0.658981 to 0.718331, a gain of
0.059349. The worst-generator robust score improved by 0.112533 and the worst
generator-condition AUC improved by 0.082553. A 1,000-replicate grouped
bootstrap estimated a mean gain of 0.059403 with 95% CI
[0.057693, 0.061231]. All modern gates passed.

Evidence:
`outputs/tiny_vnext/trained_gain_sweep/compare_gain_1p60_bootstrap1000.json`.

### Historical regression anchor

Against the frozen Adapter v2 release on the same 12,000-image, 16-condition
selection manifest:

| Metric | Adapter v2 | Tiny vNext | Delta |
|---|---:|---:|---:|
| Overall robust score | 0.942425 | 0.944886 | +0.002460 |
| CommunityForensics | 0.928369 | 0.935609 | +0.007239 |
| GenImage | 0.919772 | 0.918252 | -0.001520 |
| SID_Set | 0.957819 | 0.958064 | +0.000244 |
| Clean AUC | 0.973125 | 0.972336 | -0.000789 |

All source, source-by-degradation-family, clean, and blur-2.0 guards passed.
Evidence:
`outputs/tiny_vnext/historical_gain_sweep/release_relative_gates.json`.

### Second-seed replication

Seed 2027 reproduced the modern improvement: whole-set robust score rose from
0.684200 to 0.744749. Its generator-macro gain was 0.065108, the weakest
generator improved by 0.140227, and the weakest generator-condition AUC
improved by 0.076791. This satisfies the preregistered same-direction
replication requirement.

Seed 2027 is not the delivery candidate: its GenImage historical robust-score
drop was 0.002485, exceeding the 0.002 guard by 0.000485, and several narrow
JPEG/color guards also failed. This negative result is retained rather than
silently discarded.

Evidence:
`outputs/tiny_vnext/seed2027_modern_gain1p60/compare_generator_macro.json` and
`outputs/tiny_vnext/seed2027_historical_gain1p60/release_relative_gates.json`.

### Format-history sensitivity

On the same 3,000-image paired diagnostic, Tiny vNext changed AUC by -0.010219
under label-blind random re-encoding and -0.017512 under the stronger
neutralized re-encoding view. Relative to v1.0.0, the sensitivity worsened by
only 0.000961 and 0.001327 respectively, both within the preregistered 0.002
limit.

Evidence:
`outputs/tiny_vnext/final_candidate/compression_history_audit/diagnostic_summary.json`.

### Cost and inference contract

An interleaved 40-round, 20-forward-per-round RTX 4080 Laptop benchmark at
batch 32 measured 17.2091 ms for Adapter v2 and 17.2087 ms for Tiny vNext. The
observed relative difference (-0.002%) is measurement noise and passes the 10%
overhead guard. Both checkpoints contain 28,018,018 parameters.

The packaged checkpoint passed the real `predict.py` directory-to-JSON smoke
test, including unreadable-file fallback, finite [0,1] scores, parameter count,
and SHA-256 validation. All nine unit tests passed.

Evidence:
`outputs/tiny_vnext/final_candidate/inference_latency.json`.

## Threshold decision

The post-freeze internal scan found 0.212 with mean balanced accuracy 0.885982,
versus 0.885936 at the existing 0.209 threshold (gain 0.000045). This is below
the measurement-noise floor, so 0.209 remains frozen. Submission inference
continues to emit continuous probabilities.

## Evidence boundary

The historical 16,000-image confirmation set and WildFake observation had
already been consumed before Tiny vNext and were not reused for selection.
Tiny vNext selection used the historical regression anchor plus source-disjoint
SuSy/MS-COCOAI development data. No claim is made about a new sealed organizer
test result.
