# Ensemble vNext freeze protocol

Branch: `ensemble-vnext`

Status: alpha frozen at `0.50`; formal validation in progress.

## Evidence boundary

This is a retrospective freeze, not a new preregistration. The Tiny/Base blend
diagnostic and the alpha sweep had already been observed before this document
was written. The purpose of this document is to stop further tuning and define
the remaining pass/fail checks without moving their thresholds.

The historical confirmation sets and WildFake observation were consumed by
earlier project stages. They are not reopened for ensemble selection,
calibration, or claims of new sealed performance. The historical 12,000-image
selection anchor and the source-disjoint modern development manifest are the
only evaluation inputs used here.

## Frozen components

Model A, weight `0.50`:

- Tiny vNext checkpoint:
  `outputs/tiny_vnext/final_candidate/tiny_vnext_seed2026_gain1p60.pt`
- SHA-256:
  `1AF51D00022B9CD3FABD58D65F01C7F728F6F99C2649AAA86ACDBAA9789EDE44`

Model B, weight `0.50`:

- Base v1 checkpoint: `outputs/base_v1/primary/best.pt`
- SHA-256:
  `F49D423847B26F26FAF4C2558F1A831658F0F92DF22F37F56B3E33BA51264DD5`

Inference formula:

```text
ensemble_logit = 0.50 * tiny_vnext_logit + 0.50 * base_v1_logit
pred = sigmoid(ensemble_logit)
```

No training, calibration, threshold tuning, or test-time adaptation is applied.

## Alpha selection rule and observed decision

The historical selection sweep evaluated alpha values `0.10` through `0.60`
with unchanged Tiny-relative gates. Among candidates that passed every gate,
the one with the highest overall robust score was selected.

- alpha `0.10` through `0.50`: accepted
- alpha `0.60`: rejected because SID_Set scale-family drop was `0.005079`,
  above the frozen `0.005000` limit
- selected alpha: `0.50`
- selected historical robust score: `0.978314`
- Tiny vNext historical robust score: `0.944886`

The `0.60` gate is not relaxed despite its slightly higher overall score.

## Remaining formal gates

### Modern development

Run the full 12,896-image, 16-condition manifest for the frozen alpha only.
All checks must pass:

- generator-macro robust gain at least `0.005`
- worst-generator robust score does not regress
- worst generator-condition AUC does not regress
- 1,000-replicate content-group bootstrap 95% lower bound is above zero

### Packaged checkpoint

- one self-contained checkpoint with both source states and alpha `0.50`
- source hashes embedded in checkpoint metadata
- exact parameter count below two billion
- packaged evaluation numerically agrees with the direct alpha sweep
- directory-to-JSON output contains exactly `image_path` and `pred`
- unreadable images produce the documented neutral `0.5` fallback
- CPU and CUDA release smoke tests pass
- SHA-256 and latency evidence are recorded

### Rollback

Any failed formal gate preserves Tiny vNext as the accepted model.
