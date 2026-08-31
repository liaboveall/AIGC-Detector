# Frozen model weights

## Ensemble vNext

The current branch commits the self-contained fusion checkpoint through Git LFS:

- Asset: `aigc-detector-ensemble-vnext.pt`
- Size: 462,558,035 bytes
- SHA-256: `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`
- Composition: 0.50 Tiny vNext logit + 0.50 Base v1 logit
- Parameters: 115,585,507

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
- Size: 112,172,235 bytes
- SHA-256: `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`

Both expected digests are listed in `SHA256SUMS.txt`. Dataset images are not
redistributed; checkpoint use remains subject to the upstream dataset terms described
in the project README.
