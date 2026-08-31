# Ensemble vNext Evidence Bundle

Verdict: **PASS — frozen repository candidate**

This bundle contains aggregate, non-image evidence for the fixed 0.50/0.50 logit
ensemble. It does not contain private manifests or per-image predictions.

## Candidate identity

- Branch: `ensemble-vnext`
- Asset: `weights/aigc-detector-ensemble-vnext.pt`
- Bytes: 462,558,035
- SHA-256:
  `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`
- Tiny vNext source SHA-256:
  `1AF51D00022B9CD3FABD58D65F01C7F728F6F99C2649AAA86ACDBAA9789EDE44`
- Base v1 source SHA-256:
  `F49D423847B26F26FAF4C2558F1A831658F0F92DF22F37F56B3E33BA51264DD5`
- Parameters: 115,585,507 total, 0 trainable

## Gate results

### Historical selection anchor

- Images / conditions: 12,000 / 16
- Tiny robust score: `0.944885600`
- Ensemble robust score: `0.978314324`
- Delta: `+0.033428724`
- CommunityForensics / GenImage / SID_Set robust deltas:
  `+0.049908648 / +0.042714220 / +0.009567727`
- Alpha `0.50`: accepted
- Alpha `0.60`: rejected because SID_Set scale-family drop
  `0.005079 > 0.005000`

### Modern development

- Images / prediction rows / conditions: 12,896 / 206,336 / 16
- Global robust score: `0.740077431 -> 0.901912934`
- Generator-macro robust score: `0.718330697 -> 0.894737560`
- Worst-generator robust score: `0.548970215 -> 0.796369796`
- Worst generator-condition AUC: `0.322515532 -> 0.648096966`
- Grouped-bootstrap replicates / seed: 1,000 / 2026
- Macro-gain bootstrap mean: `0.176475480`
- 95% interval: `[0.171015078, 0.182231866]`
- All four modern gates: pass

### Packaged equivalence

- Rows aligned by `(path, condition)`: 192,000
- Key sets and static fields: exact
- Probability tolerance: `1e-3`
- Maximum / mean absolute difference:
  `0.000975102 / 0.000017555`
- Rows over tolerance: 0
- Robust-score difference: `2.70e-7`
- Maximum condition-AUC difference: `8.37e-7`

The probability tolerance covers cuDNN execution-path rounding between separately
instantiated direct and packaged graphs. Member logits are promoted to FP32 before
blending in both paths.

### Release smoke

CPU and CUDA both pass:

- checkpoint hash, fixed alpha, and source hashes
- 115,585,507 parameters and zero trainable parameters
- repeated inference equality
- exact `image_path,pred` output keys
- finite probabilities in `[0,1]`
- unreadable-image fallback `0.5`

### Paired CUDA latency

Device: NVIDIA GeForce RTX 4080 Laptop GPU. Protocol: 10 warmups, 40 alternating
rounds, 20 forwards per burst; median milliseconds per forward.

| Batch | Tiny vNext | Ensemble vNext | Relative latency | Ensemble throughput |
|---:|---:|---:|---:|---:|
| 1 | 5.014 ms | 13.874 ms | 2.77× | 72.08 images/s |
| 32 | 27.572 ms | 92.884 ms | 3.37× | 344.52 images/s |

## Files

- `alpha_gate_report.json` — complete historical candidate gates
- `modern_full_evaluation.json` — frozen-alpha live 16-condition result
- `modern_generator_macro_bootstrap1000.json` — generator strata and bootstrap gates
- `packaged_historical_evaluation.json` — full packaged-checkpoint result
- `packaged_equivalence.json` — direct/package row and metric agreement
- `latency_batch1.json`, `latency_batch32.json` — paired CUDA timings
- `release_verify_cpu.json`, `release_verify_cuda.json` — release smoke reports
- `checkpoint_metadata.json` — artifact and source provenance

## Evidence boundary

The historical anchor and modern manifest are development/model-selection data. The
previously consumed confirmation set and WildFake observation were not reopened. No
new sealed, official-hidden-test, threshold-calibration, or universal-generator claim
is made.
