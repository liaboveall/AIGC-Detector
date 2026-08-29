# Model Card — AIGC Detector Adapter v2

## Model details

- Release: `v1.0.0`
- Asset: `aigc-detector-adapter-v2.pt`
- SHA-256: `C5E0C7EC9E39B505A7269826F034969E53340D8CA2C74D60CC9B1868E43F44EC`
- Architecture: ConvNeXt-Tiny binary classifier plus residual MLP adapter
  (`768 -> 256 -> 1`)
- Parameters: 28,018,018 total; 197,121 adapter parameters
- Input: RGB images resized/cropped to 224×224 by the repository transform
- Output: one sigmoid score in `[0, 1]`, where higher means more likely AI-generated
- Documented binary-demo threshold: 0.209

## Intended use

Research, education, and hackathon demonstration of robust AI-generated-image
detection under deterministic JPEG, blur, resize, noise, color, and crop transforms.
The preferred interface is `predict.py`, which emits continuous scores for a directory.

## Out-of-scope use

- Treating a score as proof of authorship, fraud, or misconduct.
- High-stakes automated moderation without human review and target-domain validation.
- Claiming universal generator coverage or authenticity guarantees.
- Recalibrating or selecting new checkpoints on the recorded WildFake observation.
- Commercial use without reviewing every upstream dataset's terms.

## Training data summary

The final adapter uses a 560,000-row balanced replay manifest:

| Source | Rows | Share |
|---|---:|---:|
| CommunityForensics-Small | 280,000 | 50% |
| GenImage | 140,000 | 25% |
| SID_Set | 140,000 | 25% |

Each source is internally balanced between real and fake labels. CommunityForensics
generator identities are split disjointly between train, selection, and confirmation.
Image bodies are not included in this repository.

## Evaluation summary

Internal development selection (12,000 images × 16 conditions): robust score 0.942425,
clean AUC 0.973125, mean degraded AUC 0.950238, worst degraded AUC 0.911173. The model
passed all 31 pre-registered acceptance checks.

One-time post-freeze WildFake observation (13,841 images × 16 conditions): robust score
0.908171, clean AUC 0.964738, mean degraded AUC 0.929697, worst AUC 0.822067. WildFake
contains COCO real images and DALL-E 3 Advanced fake images only and is not an official
hidden-test result.

See [`reports/final_adapter_v2/`](reports/final_adapter_v2/README.md) for aggregate raw
tables and evidence boundaries.

## Limitations and risks

- Severe blur produces a high authentic-image false-positive rate.
- Strong noise and cropping reduce fake recall.
- Strong JPEG compression shifts score calibration even when AUC remains high.
- Content and semantic priors can produce persistent false positives/negatives.
- Generators, cameras, editing pipelines, languages, regions, and content types outside
  the evaluated data may behave differently.
- The development selection split influenced model choice. The earlier confirmation set
  was consumed by a rejected candidate and did not independently test Adapter v2.

Outputs should be interpreted as uncertain model scores, not factual authenticity
certificates. Human review and domain-specific validation remain necessary.

## Reproducibility

Top-level dependency versions are pinned in `requirements.txt`. Download the frozen
asset from the GitHub `v1.0.0` release, verify `weights/SHA256SUMS.txt`, and run:

```powershell
python test.py
python scripts/verify_release.py
```

Training reproduction additionally requires the upstream datasets and private manifests
documented in `Dataset/README_DATASET.md`.
