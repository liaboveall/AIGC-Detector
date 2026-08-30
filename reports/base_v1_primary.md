# Base v1 primary experiment report

Status: **rejected by the preregistered historical-retention gates**. The
frozen Tiny vNext `v1.1.0-rc1` checkpoint remains the accepted model.

## Experiment

- Branch: `base-v1`
- Student: `convnext_base.fb_in22k_ft_in1k` (87,567,489 parameters)
- Teacher: frozen Tiny vNext seed-2026 / residual-gain-1.60 checkpoint
- Training data: frozen 280,000-row balanced replay manifest
- Primary recipe: BCE on all sources, temperature-2 KD with weight 0.5 on
  GenImage and SID_Set, frozen stem/stages 0-1
- Checkpoint evaluated: `outputs/base_v1/primary/epoch_01.pt`

Epoch 1 triggered the historical early-stop guard, so epoch 2 was not run.
No frozen Tiny checkpoint, release asset, tag, or `main` artifact was changed.

## Historical validation

| Metric | Tiny vNext | Base epoch 1 | Delta | Gate |
|---|---:|---:|---:|---|
| Overall robust | 0.944886 | 0.963958 | +0.019072 | pass |
| CommunityForensics robust | 0.935609 | 0.989385 | +0.053776 | pass |
| GenImage robust | 0.918251 | 0.910638 | -0.007613 | fail |
| SID robust | 0.958064 | 0.925755 | -0.032309 | fail |

The failures are broad rather than a single narrow metric. GenImage regressed
in JPEG, blur, scale, and noise families. SID regressed in JPEG, blur, scale,
noise, and color families. Global clean performance improved, but blur-2.0
dropped by 0.003318, exceeding the 0.002 guard.

## Modern-generator validation

The 12,896-image modern development manifest was evaluated under all 16 fixed
conditions (206,336 predictions).

| Metric | Tiny vNext | Base epoch 1 | Delta |
|---|---:|---:|---:|
| Generator-macro robust | 0.718331 | 0.963147 | +0.244816 |
| Worst-generator robust | 0.548970 | 0.919776 | +0.370805 |
| Worst generator-condition AUC | 0.322516 | 0.875107 | +0.552592 |

A 1,000-replicate grouped bootstrap estimated the generator-macro improvement
as +0.245042 with a 95% interval of **[+0.236006, +0.254703]**. The modern
gain is therefore large and statistically stable on this development set.

## Decision

The Base primary checkpoint is not eligible for confirmation or release. It
solves the modern-generator weakness but fails the preregistered historical
source and source-by-degradation-family guards in several places. Per
`docs/BASE_V1_PLAN.md`, continuing the same recipe is prohibited.

Any further Base work must be registered as a separate historical-preservation
experiment with its own single mechanism, output directory, and stop-loss. It
must not consume the sealed confirmation set or WildFake during model selection.

Evidence:

- `outputs/base_v1/primary/tiny_relative_historical_gates.json`
- `outputs/base_v1/primary/modern_vs_tiny_generator_macro_bootstrap.json`
- `outputs/base_v1/primary/modern_epoch01.json`
