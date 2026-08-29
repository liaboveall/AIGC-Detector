# Devpost Project Description — Copy-ready Draft

The technical content below is final for the frozen `v1.0.0` release. Team roster,
submission URL, and uploaded video are Devpost account fields and are intentionally not
invented in this repository.

## Project name

**Robust AIGC Image Detector**

## One-line summary

A compact ConvNeXt detector with a residual domain adapter that distinguishes authentic
from AI-generated images after JPEG compression, blur, resizing, noise, color shifts,
and cropping.

## Inspiration

AI-image detection is easy to overestimate when evaluation uses clean images from the
same generators as training. Real sharing pipelines recompress, resize, filter, and crop
content, while new generators arrive continuously. TikTok TechJam Track 5 asks for a
detector that remains useful under those transformations rather than one that wins only
on clean laboratory data.

Our earliest SID-only model looked strong in-domain but reached only about 0.65 clean
AUC on the WildFake demo subset. That failure changed the project: generator coverage,
evaluation discipline, and protection against dataset shortcuts became first-class
design requirements.

## What it does

The submission accepts a directory, recursively decodes common image formats, and emits
a JSON array of continuous scores:

```json
[{"image_path": "folder/example.jpg", "pred": 0.9731}]
```

Scores near 0 indicate authentic and scores near 1 indicate AI-generated. Unreadable
supported files receive a neutral 0.5 rather than crashing the batch. The repository
also contains a deterministic 16-condition evaluation harness covering the complete
Track 5 transformation suite.

## How we built it

### 1. Multi-source forensic base

We trained ConvNeXt-Tiny across SID_Set and a multi-generator GenImage subset, then
repaired heavy-blur behavior. Moving from a single source to multiple sources produced
the largest generalization gain: WildFake clean AUC rose from approximately 0.646 to
0.961 before the final adaptation stage.

### 2. Modern-generator coverage with leakage controls

We acquired all 186 CommunityForensics-Small shards and exported 553,531 usable images
covering 4,780 fake-generator identities. Exact SHA-256 checks, WildFake exact/perceptual
overlap exclusion, NSFW exclusion, and generator-disjoint train/selection/confirmation
splits were applied before training.

The final replay manifest contains 560,000 samples, balanced real/fake and composed of
50% CommunityForensics-Small, 25% GenImage, and 25% SID_Set.

### 3. Rejecting shortcuts and catastrophic forgetting

A modern-only full-model fine-tune almost perfectly learned its new source but collapsed
on GenImage and SID_Set. Replay-only, knowledge distillation, and model-soup candidates
also produced tempting headline scores while failing one or more pre-registered
source-by-degradation safeguards.

The accepted solution freezes the 27.82M-parameter ConvNeXt base and trains a small
197,121-parameter residual MLP. CommunityForensics examples optimize final-label BCE;
GenImage and SID examples penalize non-zero residuals, preserving the old detector's
behavior. Fifteen percent label-independent JPEG/WebP re-encoding reduces reliance on
historical label/codec correlations.

### 4. Robustness-aware acceptance

We select with `0.8 * mean degraded AUC + 0.2 * worst degraded AUC`, not clean accuracy.
The final candidate had to pass 31 pre-registered checks covering overall score, modern
generator improvement, old-source retention, clean performance, every global
degradation family, and every source-by-family combination.

## Results

### Internal 12,000-image selection set, 16 conditions

- Robust score: **0.9424** (base 0.9305).
- Clean AUC: **0.9731** (base 0.9656).
- Mean degraded AUC: **0.9502** (base 0.9400).
- Worst degraded AUC: **0.9112** (base 0.8925).
- CommunityForensics robust score: **0.9284** (+0.0245).
- GenImage and SID robust changes: -0.0006 and -0.0004.
- Acceptance gates: **31/31 passed**.

### WildFake one-time observation after freeze

- Clean AUC: **0.9647**.
- Mean degraded AUC: **0.9297**.
- Worst condition (`blur_2.0`): **0.8221**.
- Robust score: **0.9082** (base 0.9047).
- Mean balanced accuracy at threshold 0.209: **0.8400** (base 0.8341).
- All 16 condition AUC and balanced-accuracy values were non-decreasing versus the base.

WildFake contains 4,998 COCO real and 8,843 DALL-E 3 Advanced images. It was used once,
only after checkpoint and threshold freeze, and never fed back into training or model
selection. It is demonstration evidence, not an official final score.

## Accomplishments we are proud of

- Solved the catastrophic-forgetting failure without expanding to a large backbone.
- Added only 0.70% parameters while keeping batch-32 inference overhead near 0.6%.
- Improved all 16 global internal AUC conditions and did not regress any WildFake
  condition versus the base.
- Built deterministic acquisition, manifest, leakage, compression-history, threshold,
  robustness, and release-verification tooling.
- Preserved rejected experiments and evidence boundaries instead of selecting only the
  most flattering scalar result.

## Challenges

### Cross-generator generalization

High in-domain performance did not transfer automatically. More model capacity would
not have fixed the source mismatch; diverse training sources did.

### Modern adaptation versus old knowledge

The strongest modern-only and Replay+KD candidates sacrificed narrow old-source
degradation behavior. The frozen-base residual adapter produced a better multi-objective
trade-off than continued full-model fine-tuning.

### Ranking versus operating threshold

At JPEG q30, WildFake AUC remains 0.9811 while balanced accuracy at threshold 0.209 is
0.7676. The model still ranks well, but recompression shifts fake scores downward. We
therefore emit continuous scores and document the binary operating point honestly.

## What we learned

Generator coverage contributed more than a larger backbone was likely to contribute at
this stage. Robust evaluation must also be source-aware: a global average can hide a
small but meaningful regression for one dataset and one degradation family.

We also learned that a sealed validation set is a consumable resource. Our 16,000-image
confirmation set was opened once for an earlier model-soup candidate; that candidate
failed, and we did not reuse the set for the adapter. The final evidence statement
explicitly preserves that limitation.

## What's next

The frozen release is the submission model. Further research will start only with a new
development split and a new untouched final test. The highest-value directions are:

1. new authentic and modern-generator coverage with license and leakage controls;
2. multi-scale or frequency-domain features for severe blur;
3. better target-domain calibration for compressed-score shift;
4. controlled backbone/semantic-feature comparisons after the current method is fully
   reproduced.

## Built with

Python, PyTorch, torchvision, timm, pandas, scikit-learn, Pillow, PyYAML, Hugging Face
datasets, PyArrow, TensorBoard, tqdm, conda, Git, and GitHub Releases.

## Supporting material

- [`ROBUSTNESS_SUMMARY.md`](ROBUSTNESS_SUMMARY.md)
- [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)
- [`DEMO_VIDEO_SCRIPT.md`](DEMO_VIDEO_SCRIPT.md)
- [`../reports/final_adapter_v2/`](../reports/final_adapter_v2/README.md)
