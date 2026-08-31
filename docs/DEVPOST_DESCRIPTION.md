# Devpost Description — Ensemble vNext Draft

This technical text matches the frozen `ensemble-vnext` branch. Team names, real
contributions, repository/release URLs, media, and submission metadata must be supplied
by the team owner.

## Inspiration

AI-image detection often looks strong on clean images and then fails after ordinary
platform transformations. JPEG recompression, blur, resize, noise, color changes, and
crop can remove forensic traces or shift a model's score distribution. We built a
detector and evaluation protocol around those real deployment failures rather than a
single clean benchmark.

## What it does

The project recursively scores supported images and returns one continuous probability:

```json
[
  {"image_path": "subfolder/example.jpg", "pred": 0.9731}
]
```

Higher scores mean more likely AI-generated. Unreadable files receive a neutral `0.5`
instead of terminating the batch.

## How we built it

The final candidate is a fixed logit-space ensemble:

```text
image
  ├─ Tiny vNext adapter ─┐
  └─ ConvNeXt Base v1 ──┴─ 0.50 / 0.50 FP32 logit blend ─ sigmoid ─ score
```

The 28.0M-parameter Tiny member preserves the strongest historical cross-source
behavior. The 87.6M-parameter Base member contributes complementary evidence for newer
generators. The complete 115.6M-parameter model is frozen and packaged in one
hash-pinned checkpoint; inference performs no training, calibration, or test-time
adaptation.

We evaluate clean plus 15 deterministic degradations. The robustness score gives 80%
weight to mean degraded AUC and 20% to the worst degraded AUC. Source and
degradation-family guards prevent a large model from winning globally while silently
regressing an older dataset.

## Fusion decision

On the historical 12,000-image development anchor, alpha values from 0.10 to 0.50
passed. Alpha 0.60 achieved a slightly higher overall score but was rejected because
its SID_Set scale-family drop exceeded the frozen limit
(`0.005079 > 0.005000`). We selected and froze alpha 0.50.

| Historical metric | Tiny vNext | Ensemble |
|---|---:|---:|
| Robust score | 0.944886 | **0.978314** |
| Clean AUC | 0.972336 | **0.991032** |
| Worst degraded AUC | 0.916339 | **0.961712** |

We then ran the frozen alpha on a 12,896-image source-disjoint modern development set:

| Modern metric | Tiny vNext | Ensemble |
|---|---:|---:|
| Global robust score | 0.740077 | **0.901913** |
| Clean AUC | 0.850770 | **0.958501** |
| Generator-macro robust score | 0.718331 | **0.894738** |
| Worst-generator robust score | 0.548970 | **0.796370** |
| Worst generator-condition AUC | 0.322516 | **0.648097** |

All 16 global condition AUCs improved. A 1,000-replicate content-group bootstrap gave
a generator-macro gain 95% interval of `[0.171015, 0.182232]`.

## Challenges and lessons

Other Base-only candidates also failed preservation gates. The useful lesson was that
capacity alone did not solve domain shift; complementary models plus strict regression
guards did.

Packaging introduced another subtle issue: CUDA autocast initially blended half-
precision logits, while the direct sweep blended FP32 logits. We made the production
wrapper explicitly promote both logits before arithmetic and added direct/package
equivalence checks.

## Limitations

Strong noise remains the global bottleneck. Midjourney v1/v2 under `noise_0.05` is the
weakest generator-condition pair. The ensemble is larger and slower than Tiny vNext,
and its probabilities have not been recalibrated on a new independent set.

The historical confirmation set and WildFake had already been consumed in earlier
project stages, so we did not reopen them. These are internal development results, not
an official hidden-test or universal-generator claim. The output should support human
review, not act as proof of image origin.

## Built with

Python, PyTorch, timm, scikit-learn, Pillow, pandas, NumPy, Git LFS, deterministic
robustness transforms, grouped bootstrap validation, and SHA-256 provenance checks.
