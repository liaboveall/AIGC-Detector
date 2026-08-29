# v1.0.0 — Frozen Track 5 Submission Model

This release freezes Adapter v2 as the formal TikTok TechJam Track 5 checkpoint.

## Asset

- `aigc-detector-adapter-v2.pt`
- Size: 112,172,235 bytes
- SHA-256: `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`
- Architecture: ConvNeXt-Tiny + residual MLP adapter
- Parameters: 28,018,018
- Binary-demo threshold: 0.209; submission output remains a continuous score

After downloading the asset into `weights/`, run:

```powershell
python scripts/verify_release.py
python predict.py --input-dir path/to/images --output predictions.json
```

## Evidence summary

- Internal development selection, 12,000 images × 16 conditions: robust score
  `0.930488 -> 0.942425`; all 31 pre-registered gates passed.
- One-time post-freeze WildFake observation: robust score `0.904694 -> 0.908171`;
  all 16 AUC and balanced-accuracy conditions were non-decreasing versus the base.
- Batch-32 inference overhead: approximately 0.6%; adapter parameters: 0.70% of total.

## Important limits

- The internal selection result influenced model choice.
- The earlier confirmation split was consumed by a rejected model-soup candidate and
  did not independently test Adapter v2.
- WildFake is a narrow demonstration set, not an official hidden-test score.
- Heavy blur and strong noise remain the primary weak conditions.
- Review upstream dataset terms before use; datasets are not included in this release.

See `MODEL_CARD.md`, `docs/ROBUSTNESS_SUMMARY.md`, `docs/ERROR_ANALYSIS.md`, and
`reports/final_adapter_v2/` in the tagged source for full detail.
