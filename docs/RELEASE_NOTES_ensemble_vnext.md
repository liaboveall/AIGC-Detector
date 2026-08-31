# Ensemble vNext — Frozen Fusion Candidate

Date: 2026-09-01

Status: complete on branch `ensemble-vnext`; not yet merged, tagged, pushed, or
published as a new GitHub release.

## Asset

- `weights/aigc-detector-ensemble-vnext.pt`
- 462,558,035 bytes
- SHA-256
  `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`
- 115,585,507 parameters
- fixed `alpha=0.50` on the Base-v1 logit

The checkpoint is self-contained and tracked through Git LFS. It embeds both source
model states, source hashes, and the blend weight.

## Validation outcome

- Historical selection-anchor robust score: `0.944886 -> 0.978314`
- Modern global robust score: `0.740077 -> 0.901913`
- Modern generator-macro robust score: `0.718331 -> 0.894738`
- Modern worst-generator robust score: `0.548970 -> 0.796370`
- Modern worst generator-condition AUC: `0.322516 -> 0.648097`
- Grouped-bootstrap macro-gain 95% CI: `[0.171015, 0.182232]` (1,000 replicates)
- 16/16 modern global condition AUCs improved
- CPU and CUDA deterministic directory-inference smoke tests passed
- unreadable-image fallback remains exactly `0.5`
- Paired RTX 4080 Laptop latency: `13.87 ms` at batch 1 and `92.88 ms` at
  batch 32 (2.77× / 3.37× Tiny vNext)

See `reports/ensemble_vnext/` for machine-readable evidence, the grouped-bootstrap
interval, packaged/direct equivalence, and latency measurements.

## Evidence boundary

The 12,000-image historical anchor and 12,896-image source-disjoint modern development
set are model-selection/development evidence. The previously consumed confirmation set
and WildFake observation were not reopened. No new sealed or official hidden-test claim
is made, and no ensemble-specific binary threshold was calibrated.

## Verify

```powershell
git lfs pull
python test.py
python scripts/verify_ensemble_release.py --device cuda
python scripts/verify_ensemble_release.py --device cpu
```
