# Delivery Checklist — Ensemble vNext

This checklist separates the completed public repository release from account-bound
competition submission actions. The model is frozen at `alpha=0.50`; no further tuning
on the recorded evidence belongs in this release.

## Branch and provenance

- [x] Fusion work was completed and reviewed on `ensemble-vnext`.
- [x] The verified fusion history is promoted to the public `main` branch.
- [x] Tiny vNext and Base v1 source paths and SHA-256 values are frozen.
- [x] Alpha selection rule and the rejected `0.60` gate are documented.
- [x] Consumed confirmation/WildFake data were not reopened.

## Validation

- [x] Historical 12,000-image × 16-condition alpha sweep completed.
- [x] Alpha `0.50` passed all unchanged Tiny-relative gates.
- [x] Modern 12,896-image × 16-condition live evaluation completed.
- [x] Modern macro gain, worst-generator, and worst-condition gates passed.
- [x] 1,000-replicate grouped-bootstrap lower bound is positive
  (`0.171015 > 0`).
- [x] Packaged checkpoint full historical evaluation completed after the final FP32
  blending fix.
- [x] Direct/package identities and probabilities pass the recorded numerical
  equivalence tolerance.
- [x] Paired CUDA latency evidence recorded for batch sizes 1 and 32.

## Artifact and runtime

- [x] One self-contained checkpoint contains both model states and fixed alpha.
- [x] Asset size is 462,558,035 bytes.
- [x] SHA-256 is
  `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`.
- [x] Parameter count is 115,585,507, below the two-billion cap.
- [x] Source checkpoint hashes are embedded in metadata.
- [x] Default `predict.py` checkpoint points to Ensemble vNext.
- [x] Directory output contains exactly `image_path` and `pred`.
- [x] Unreadable-image fallback is exactly `0.5`.
- [x] Final CPU and CUDA release verifiers pass after all runtime changes.
- [x] Unit tests, compile checks, dependency check, and
  `git diff --check` pass.
- [x] Fusion checkpoint is committed through Git LFS with metadata and checksum.
- [x] Tracked evidence and all current-facing documents match the final candidate.

## Publication and account-bound actions

- [x] Push `ensemble-vnext` and the promoted `main` branch to the public remote.
- [x] Create and verify the public `v2.0.0` release tag.
- [x] Upload the checkpoint and checksum manifest as GitHub release assets.
- [x] Set the public repository description, topics, license, CI, and security policy.
- [ ] Confirm the real Devpost team roster and contribution statements.
- [ ] Record/upload the demo video and add its final URL.
- [ ] Paste/review the Devpost text and submit before the real deadline.
- [ ] Save the official submission receipt or screenshot.

## Final verification commands

```powershell
git lfs pull
python test.py
python scripts/verify_ensemble_release.py --device cuda
python scripts/verify_ensemble_release.py --device cpu
python -m compileall -q src scripts train.py train_adapter.py `
  train_base_v3.py train_repair_adapter.py train_replay_distill.py `
  evaluate.py predict.py test.py
python -m pip check
git diff --check
```
