# Frozen model weights

## Ensemble vNext — v2.0.0

The current public release commits the self-contained fusion checkpoint through Git
LFS and mirrors it as a GitHub release asset:

- Asset: `aigc-detector-ensemble-vnext.pt`
- Size: 462,558,035 bytes
- SHA-256: `B3A2002C6C297D6382D88B0BA83C4059CE75BBA01C14049A26D3CF45D074DC4B`
- Composition: 0.50 Tiny vNext logit + 0.50 Base v1 logit
- Parameters: 115,585,507
- Release: <https://github.com/liaboveall/AIGC-Detector/releases/tag/v2.0.0>

```powershell
git lfs pull
Get-FileHash weights/aigc-detector-ensemble-vnext.pt -Algorithm SHA256
python scripts/verify_ensemble_release.py --device cuda
python scripts/verify_ensemble_release.py --device cpu
```

Load the checkpoint through `predict.py`, `evaluate.py`, or
`src.adapter.build_checkpoint_model`. The embedded source paths are provenance only;
inference does not require separate member checkpoint files.

## Published rollback release

The earlier Adapter v2 checkpoint remains available from the public `v1.0.0` release:

- Release: <https://github.com/liaboveall/AIGC-Detector/releases/tag/v1.0.0>
- Asset: `aigc-detector-adapter-v2.pt`
- Size: 112,172,171 bytes
- SHA-256: `17FE0D53D4264D93485F91BF11E24733A637280324889E2920B168BC1C7999DE`

Both expected digests are listed in `SHA256SUMS.txt`. Dataset images are not
redistributed; checkpoint use remains subject to the upstream dataset terms described
in the project README.
