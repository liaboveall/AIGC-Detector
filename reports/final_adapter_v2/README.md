# Frozen Adapter v2 evidence bundle

This directory is the compact, tracked evidence bundle for the formal `v1.0.0`
checkpoint. Large per-image internal predictions and private dataset manifests remain
outside Git; the aggregate tables needed to reproduce the reported conclusions are
included here.

## Model identity

- Architecture: frozen ConvNeXt-Tiny base plus a `768 -> 256 -> 1` residual adapter.
- Parameters: 27,820,897 frozen base + 197,121 adapter = **28,018,018 total**.
- Frozen threshold: **0.209** (continuous scores remain the submission output).
- Release asset: `aigc-detector-adapter-v2.pt`, 112,172,235 bytes.
- SHA-256: `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`.

## Internal selection result

The model was selected on a fixed 12,000-image, three-source development split under
all 16 deterministic conditions. CommunityForensics generator identities are disjoint
between training, selection, and the previously sealed confirmation split.

| Metric | Frozen base | Adapter v2 | Delta |
|---|---:|---:|---:|
| Robust score | 0.930488 | **0.942425** | **+0.011938** |
| Clean AUC | 0.965637 | **0.973125** | +0.007488 |
| Mean degraded AUC | 0.939995 | **0.950238** | +0.010243 |
| Worst degraded AUC (`noise_0.10`) | 0.892458 | **0.911173** | +0.018715 |
| CommunityForensics robust score | 0.903910 | **0.928369** | +0.024459 |
| GenImage robust score | 0.920356 | 0.919772 | -0.000584 |
| SID_Set robust score | 0.958171 | 0.957819 | -0.000352 |

All 16 global condition AUCs improved. The pre-registered acceptance protocol passed
**31/31** checks with the source-by-family regression bound fixed at 0.005. A later,
non-binding 0.002 stress test passed 30/31 checks; its only miss was GenImage noise
family drop 0.002583, exceeding that tighter line by 0.000583. The binding protocol was
not changed after seeing results.

Machine-readable values and the exact evidence boundary are in
[`selection_summary.json`](selection_summary.json).

## One-time WildFake observation

After checkpoint and threshold freeze, the model was observed once on the organizer's
13,841-image WildFake demo subset (4,998 COCO real; 8,843 DALL-E 3 Advanced fake).
WildFake did not influence training, checkpoint selection, or threshold selection.

| Metric | Frozen base | Adapter v2 | Delta |
|---|---:|---:|---:|
| Robust score | 0.904694 | **0.908171** | +0.003477 |
| Clean AUC | 0.963591 | **0.964738** | +0.001148 |
| Mean degraded AUC | 0.927088 | **0.929697** | +0.002610 |
| Worst AUC (`blur_2.0`) | 0.815118 | **0.822067** | +0.006949 |
| Mean balanced accuracy at 0.209 | 0.8341 | **0.8400** | +0.0059 |

All 16 condition AUC and balanced-accuracy values were non-decreasing versus the
frozen base. The gain is consistent but modest; WildFake is a demonstration set with
one fake-generator family and is not evidence of universal or leaderboard-level
generalization.

Raw aggregate artifacts:

- [`wildfake_frozen_threshold_table.csv`](wildfake_frozen_threshold_table.csv)
- [`wildfake_frozen_threshold_comparison.csv`](wildfake_frozen_threshold_comparison.csv)
- [`wildfake_source_class_stats.csv`](wildfake_source_class_stats.csv)
- [`error_cases.csv`](error_cases.csv)

## Additional release checks

- Format-history diagnostic: no detectable shortcut worsening; candidate-versus-base
  changes were within 0.00064 AUC across the four paired views. Raw aggregate tables:
  [`format_diagnostic_metrics.csv`](format_diagnostic_metrics.csv) and
  [`format_paired_auc_deltas.csv`](format_paired_auc_deltas.csv).
- Inference overhead on an RTX 4080 Laptop GPU: +0.24 ms for batch 1, +1.0% for batch
  8, and +0.6% for batch 32. Raw protocol and values:
  [`inference_latency.json`](inference_latency.json).
- Threshold scan: 0.155-0.170 produced at most +0.0012 mean balanced accuracy versus
  0.209 on internal development data, below the measurement-noise floor. The previously
  frozen 0.209 threshold was retained.

## Evidence boundary

The 16,000-image confirmation set was opened exactly once for an earlier model-soup
candidate, which was rejected by one source-family gate. It was therefore permanently
consumed and was not reused for Adapter v2. Adapter v2's evidence is the generator-
disjoint development selection result plus the one-time post-freeze WildFake observation.
No official hidden-test or leaderboard result is claimed.
