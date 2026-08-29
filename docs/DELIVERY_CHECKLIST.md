# Delivery Checklist — v1.0.0

This file separates repository-controlled release gates from account-bound Devpost
actions. The model is frozen; no further training or WildFake-driven tuning belongs in
this release.

## Repository-controlled gates

- [x] Formal checkpoint fixed as Adapter v2 epoch 1.
- [x] Source checkpoint, read-only backup, and release-staging copy have identical
  SHA-256 `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`.
- [x] Source and backup checkpoints are read-only; legacy base remains available for
  rollback.
- [x] Training entry point refuses to write into a non-empty output directory unless
  the caller explicitly overrides the guard.
- [x] Adapter-aware loading works while legacy-checkpoint behavior remains compatible.
- [x] Directory-to-JSON inference emits exactly `image_path` and `pred`.
- [x] Unreadable-image fallback is 0.5.
- [x] Unit/pipeline suite passes.
- [x] Python compile check and all public CLI `--help` entry points pass.
- [x] Release smoke test passes on CUDA and CPU.
- [x] Dependency health check passes; verified top-level versions are pinned.
- [x] README, model card, robustness summary, error analysis, Devpost draft, demo script,
  dataset documentation, aggregate evidence bundle, and release notes use Adapter v2.
- [x] Dataset images, private manifests, per-image internal predictions, local planning
  files, and raw training outputs remain excluded from Git.
- [ ] Public `main` contains the delivery commits.
- [ ] Public `v1.0.0` Release contains the checkpoint and checksum assets.
- [ ] Publicly downloaded asset metadata matches the frozen byte size and SHA-256.

The final three publication checks are completed and updated during the GitHub release
step; they cannot be truthfully checked before the corresponding public state exists.

## Devpost/account-bound actions

These require the team owner's identity, media, or authenticated Devpost account and are
not fabricated by repository automation:

- [ ] Confirm the final project title and enter the real team roster/contributions.
- [ ] Record the three-minute demo with licensed display images using
  `DEMO_VIDEO_SCRIPT.md`.
- [ ] Upload the video and add its final URL to Devpost.
- [ ] Paste/review `DEVPOST_DESCRIPTION.md` within Devpost field limits.
- [ ] Add the public GitHub repository and `v1.0.0` Release links.
- [ ] Submit before the competition deadline and save a submission receipt/screenshot.

## Release verification commands

```powershell
python test.py
python scripts/verify_release.py --device cuda
python scripts/verify_release.py --device cpu
python -m pip check
git diff --check
```
