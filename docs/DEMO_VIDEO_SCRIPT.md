# Two-Minute Demo Script — Ensemble vNext

Target length: 130–140 seconds. Solo entrant: all narration is first-person singular
(no "we"/"our"). Use only licensed display images. Preload the Git LFS checkpoint so
network speed is not part of the recording.

## Preflight

```powershell
conda activate jam
git lfs pull
python scripts/verify_ensemble_release.py --device cuda
python predict.py --input-dir demo_images --output demo_predictions.json
```

Do not show private dataset paths, per-image development predictions, credentials, or
unverified leaderboard claims.

## 0:00–0:15 — Problem

**Visual:** the same image through clean, JPEG, blur, resize, noise, color, and crop
(`demo_visuals/01–07`, ~2 s each).

**Narration:**

> AI-image detectors often look accurate on clean files, then break after ordinary
> social-media processing. I built this detector and its validation suite around 15
> deterministic degradations, not one clean benchmark.

## 0:15–0:38 — Model

**Visual:**

```text
image
  ├─ Tiny vNext adapter ─┐
  └─ ConvNeXt Base v1 ──┴─ equal FP32 logit blend ─ probability
```

**Narration:**

> The final model combines a compact Tiny branch that preserves historical
> cross-source behavior with a Base branch that is stronger on newer generators. Both
> branches are frozen, and I blend their logits fifty-fifty into a single
> 115.6-million-parameter checkpoint.

## 0:38–0:50 — Fusion guardrails

**Visual:** the alpha gate report.

**Narration:**

> Even alpha 0.60 was rejected for exceeding the SID_Set scale guard. The accepted
> alpha is 0.50.

## 0:50–1:15 — Results

**Visual:** two concise tables.

**Narration:**

> On the historical 12,000-image anchor, robust score rose from 0.9449 to 0.9783. On a
> source-disjoint modern development set, it rose from 0.7401 to 0.9019, all sixteen
> condition AUCs improved, and the bootstrap macro-gain interval is 0.1710 to 0.1822.

**On-screen disclaimer:** “Internal development evidence; not an official hidden-test
score.”

## 1:15–1:40 — Live inference

**Visual:** run `predict.py`, then show the JSON in the terminal.

**Narration:**

> The interface scans a directory and outputs exactly two fields: relative image path
> and a continuous AI probability. Unreadable files return a neutral 0.5 instead of
> crashing the run.

Show one authentic-looking example, one synthetic-looking example, and one deliberately
corrupt file. Do not describe any single score as proof.

## 1:40–1:55 — Reproducibility

**Visual:** checksum manifest and CPU/CUDA verifier reports.

**Narration:**

> The 462.6-megabyte checkpoint is Git-LFS-tracked with a SHA-256 checksum, and my
> release verifier checks the embedded source hashes, fixed alpha, parameter count,
> deterministic repeated inference, and CPU and CUDA execution.

## 1:55–2:10 — Limitations and close

**Visual:** `noise_0.10` and Midjourney v1/v2 `noise_0.05` callouts.

**Narration:**

> Strong noise remains the main weakness, and this is a review signal, not an
> authenticity certificate. The repository ships the code, negative experiments,
> validation evidence, and a reproducible checkpoint.

## Accuracy guardrails

- Say “internal development set,” never “official score.”
- Do not claim a fresh confirmation or WildFake result for the ensemble.
- Do not reuse Adapter v2's threshold 0.209 for the ensemble.
- Do not claim state of the art, universal detection, or guaranteed authenticity.
- Show `aigc-detector-ensemble-vnext.pt`, not the v1.0.0 Adapter asset, as the
  `v2.0.0` release checkpoint.
- First-person singular only: never say “we” or “our” in narration.
