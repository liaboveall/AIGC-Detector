# Compression-history shortcut audit

## Scope and protocol

- Internal data only; no WildFake rows were read. Diagnostic manifest: `Dataset/manifests/validation_multisource.csv`.
- Frozen checkpoint: `outputs/multisource_blur_finetune/best.pt`; paired stratified sample: 3,000 images; seed: 2026.
- `decoded`: ordinary decoded input.
- `jpeg_q75`: every decoded image is saved once more as JPEG quality 75.
- `hash_random_reencode`: codec (JPEG/WebP) and quality (50/65/80/95) are assigned only by a seeded SHA-256 hash of the relative path, never by label.
- `neutralized_random_reencode`: adds mild resize round-trip and one-LSB deterministic noise before the same label-blind re-encoding. This is a sensitivity stress test, not a pure causal intervention.

## Static association

| Split | Rows | JPEG suffix real | JPEG suffix fake | V(suffix,label) | PIL probe rows | PIL JPEG real | PIL JPEG fake | V(PIL format,label) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 482,820 | 1.0000 | 0.0000 | 1.0000 | 500 | 1.0000 | 0.2900 | 0.7034 |
| validation | 16,000 | 1.0000 | 0.0000 | 1.0000 | 500 | 1.0000 | 0.3167 | 0.6806 |

## Frozen-model paired diagnostic

| View | ROC AUC | AUC delta vs decoded | 95% paired-bootstrap interval | Mean score real | Mean score fake |
|---|---:|---:|---:|---:|---:|
| decoded | 0.9713 | +0.0000 | reference | 0.0672 | 0.8769 |
| jpeg_q75 | 0.9875 | +0.0163 | [0.0119, 0.0206] | 0.0417 | 0.8873 |
| hash_random_reencode | 0.9624 | -0.0088 | [-0.0133, -0.0031] | 0.0831 | 0.8480 |
| neutralized_random_reencode | 0.9557 | -0.0156 | [-0.0213, -0.0095] | 0.0871 | 0.8275 |

## Interpretation

- Filename suffix is perfectly associated with label here, but suffix is not the same as the decoded container: most sampled SID tampered files have a `.png` name while PIL identifies JPEG bytes. The sampled PIL-format association is therefore the more relevant static signal and remains strong but not perfect. A model cannot read the suffix directly; it can exploit codec residue correlated with the decoded history.
- Label-blind random re-encoding changed AUC by -0.0088; the stronger neutralization stress test changed it by -0.0156. Large negative shifts support sensitivity to codec/resampling history, but do not prove that all lost signal was a shortcut because valid forensic cues are also perturbed.
- A single static JPEG conversion is not a remedy: original JPEG real images become effectively double-compressed, while original PNG fake images are typically compressed once. The original history can therefore survive uniform JPEG re-encoding.
- Recommended mitigation, if used in training, is label-independent randomized codec/quality assignment with exposure balance verified by label, while retaining a no-extra-reencoding branch. Evaluate it only on internal selection/confirmation splits before freezing the candidate.

## Reproducibility

```powershell
python scripts/audit_compression_history.py
```

Aggregate tables are stored beside this report. Per-image predictions and the sampled
manifest are intentionally excluded from Git and can be regenerated with the command
above when the private dataset is available.
