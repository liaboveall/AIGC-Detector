# Deliverable 1 — Devpost Project Description (Draft)

> Copy-ready draft for the Devpost submission page. Sections map to the typical Devpost
> fields; adjust lengths to the platform's limits. Placeholders marked `[...]` must be
> replaced before publishing.

---

## Project name

**[Proposed] BlurProof — a robust AI-generated-image detector** *(working title, replace
if the team prefers another)*

## One-line summary

A ConvNeXt-Tiny detector that keeps telling real from AI-generated images even after
JPEG recompression, heavy blur, downscaling, noise, color shifts, and cropping —
trained on multi-source real+synthetic data with a blur-focused fine-tune, evaluated
on a strictly held-out WildFake split.

## Inspiration / how we addressed the problem statement

TikTok TechJam Track 5 asks for detection that survives the real world: social-media
pipelines recompress, resize, and filter images, and detectors tuned on clean lab data
collapse under those transformations. We attacked the problem from three angles:

1. **Training for the transformation space.** Our augmentation engine applies the
   same six transformation families (JPEG, blur, downscale, noise, brightness, crop)
   used by the evaluation protocol, so the model learns forensic cues that persist
   through them.
2. **Training for generator diversity.** We mixed SID_Set with a multi-generator
   GenImage subset (exact-hash deduplicated), because a single-source model
   catastrophically failed cross-source in our first iteration (WildFake clean AUC ~0.646).
3. **Targeted repair of the weakest link.** After diagnosing heavy blur as the worst
   condition, we fine-tuned the frozen multi-source checkpoint with `blur_2.0` exposure
   raised from ~5.8% to 20% — lifting the hardest condition from 0.7834 to 0.8151 AUC
   without touching the held-out evaluation set.

## What it does

Given a directory of images, the detector emits a JSON array of
`{"image_path": ..., "pred": ...}` confidence scores (0 = real, 1 = AI-generated),
recursively scanning the folder and gracefully handling unreadable files. A frozen,
calibrated threshold (0.209) is available for binary decisions, but continuous scores
are always provided so deployments can pick their own precision/recall operating point.

## How we built it

**Tools:** VS Code with the Qoder agent for pair programming, conda for environment
management, PyTorch + torchvision for training, timm for the ConvNeXt-Tiny backbone,
TensorBoard for experiment tracking, pandas/scikit-learn for calibration and error
analysis, and the Hugging Face `datasets` library for GenImage ingestion. All code is
plain Python — no heavyweight frameworks beyond PyTorch.

**Model & libraries:** ConvNeXt-Tiny (27.8M parameters — far under the 2B limit),
ImageNet-pretrained, trained with BCE on a degradation-augmented pipeline; checkpoint
selection uses a robustness-aware criterion (`0.8 × mean degraded AUC + 0.2 × worst
degraded AUC`) instead of clean accuracy.

## Datasets

- **SID_Set** (train/val/test splits of real, tampered, and fully synthetic images) —
  primary backbone of the training mixture.
- **GenImage subset** — multi-generator synthetic images with matched real counterparts,
  downloaded via the official dataset shards and hash-deduplicated against SID_Set.
- **CIFAKE** — held exclusively for ablation studies; not used in the final model.
- **WildFake demo subset** (4,998 COCO val2017 real + 8,843 DALL·E 3 Advanced images) —
  provided by the organizers; used **only** for demonstration evaluation. It never
  touched training, checkpoint selection, or threshold calibration.

Licensing note: all datasets are publicly released research datasets used under their
respective research-purpose licenses. `[Team to verify and list exact license terms
before publishing if Devpost requires it.]`

## Results (WildFake held-out, ROC AUC)

- Clean: **0.9636**
- Mean degraded AUC across 15 transformed conditions: **0.9271**
- Worst condition (`blur_2.0`): **0.8151**
- Robust score: **0.9047**
- Notably, all four JPEG qualities score *above* clean (0.9808–0.9897): recompression
  does not break the detector's ranking.

## Challenges we ran into

- **Cross-source collapse.** Our first SID-only model scored 0.97+ in-domain but ~0.65
  AUC on WildFake — a sobering reminder that degradation robustness ≠ source
  generalization. Multi-source training was the fix.
- **The blur/noise trade-off.** Repairing heavy blur cost a little noise robustness
  (`noise_0.10` 0.8780 → 0.8576); we accepted the trade because the net robust score
  improved.
- **Calibration drift under JPEG.** Ranking survives strong compression, but the score
  distribution shifts, so fixed-threshold accuracy drops at q30 — an honest failure mode
  we document rather than hide.

## What we learned

Evaluation integrity matters as much as accuracy: keeping WildFake fully held out and
freezing the threshold before final scoring made every reported number trustworthy. We
also learned that error analysis should be content-aware — our worst errors are a small
set of images that fail under *every* transformation, pointing at semantic rather than
signal-level blind spots.

## What's next

Restore noise robustness with a joint blur+noise schedule, attack `blur_2.0` with
frequency-domain or multi-scale branches, and add semantic (CLIP-style) features to
eliminate the recurring content-driven false positives/negatives.

## Team

[Member 1], [Member 2], [Member 3] — [one-line roles]

---

*Related deliverables: [`ROBUSTNESS_SUMMARY.md`](ROBUSTNESS_SUMMARY.md),
[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md), [`DEMO_VIDEO_SCRIPT.md`](DEMO_VIDEO_SCRIPT.md).*
