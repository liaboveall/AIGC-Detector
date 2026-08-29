# Frozen model weights

The formal Track 5 release checkpoint is distributed as a GitHub Release asset rather
than committed to Git:

- Release: <https://github.com/liaboveall/AIGC-Detector/releases/tag/v1.0.0>
- Asset: `aigc-detector-adapter-v2.pt`
- Size: 112,172,235 bytes
- SHA-256: `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`

Download it into this directory:

```powershell
Invoke-WebRequest `
  https://github.com/liaboveall/AIGC-Detector/releases/download/v1.0.0/aigc-detector-adapter-v2.pt `
  -OutFile weights/aigc-detector-adapter-v2.pt
```

Then verify both the checksum and the inference contract:

```powershell
Get-FileHash weights/aigc-detector-adapter-v2.pt -Algorithm SHA256
python scripts/verify_release.py
```

The checkpoint contains both the frozen ConvNeXt-Tiny base and the trained residual
adapter. Load it only through the repository's adapter-aware `predict.py`, `evaluate.py`,
or `src.adapter.build_checkpoint_model` path.

The source datasets are not redistributed. Use of the checkpoint remains subject to
the terms of the datasets described in the project README, including the
CommunityForensics-Small CC-BY-NC-SA-4.0 terms.
