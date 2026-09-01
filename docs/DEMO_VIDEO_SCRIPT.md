# Three-Minute Demo Script — Ensemble vNext

Target length: 175–180 seconds. Use only licensed display images. Preload the Git LFS
checkpoint so network speed is not part of the recording.

## Preflight

```powershell
git lfs pull
python scripts/verify_ensemble_release.py --device cuda
python predict.py --input-dir demo_images --output demo_predictions.json
```

Do not show private dataset paths, per-image development predictions, credentials, or
unverified leaderboard claims.

## 0:00–0:20 — Problem

**Visual:** the same image through clean, JPEG, blur, resize, noise, color, and crop.

**Narration:**

> AI-image detectors can look accurate on clean files and break after ordinary social
> media processing. We built a detector and validation suite around 15 deterministic
> degradations, not just one clean benchmark.

## 0:20–0:50 — Model

**Visual:**

```text
image
  ├─ Tiny vNext adapter ─┐
  └─ ConvNeXt Base v1 ──┴─ equal FP32 logit blend ─ probability
```

**Narration:**

> Our final candidate combines a compact Tiny model that preserves historical
> cross-source behavior with a Base model that is stronger on newer generators. Both
> are frozen. We blend their logits fifty-fifty and package both states in one
> checkpoint with 115.6 million parameters.

## 0:50–1:15 — Fusion guardrails

**Visual:** the alpha gate report.

**Narration:**

> Even alpha 0.60 was rejected because it exceeded the SID_Set scale guard. The
> accepted alpha is 0.50.

## 1:15–1:50 — Results

**Visual:** two concise charts or tables.

**Narration:**

> On the historical 12,000-image selection anchor, robust score increased from 0.9449
> to 0.9783. On a source-disjoint modern development set, it increased from 0.7401 to
> 0.9019. Generator-macro robust score rose from 0.7183 to 0.8947, and all sixteen
> global condition AUCs improved. A thousand-replicate grouped bootstrap put the macro
> gain interval between 0.1710 and 0.1822.

**On-screen disclaimer:** “Internal development evidence; not an official hidden-test
score.”

## 1:50–2:25 — Live inference

**Visual:** run `predict.py`, then open the JSON.

**Narration:**

> The interface recursively scans a directory and outputs exactly two fields: relative
> image path and a continuous AI probability. Unreadable files return a neutral 0.5
> instead of crashing the run.

Show one authentic-looking example, one synthetic-looking example, and one deliberately
corrupt file. Do not describe any single score as proof.

## 2:25–2:45 — Reproducibility

**Visual:** checksum manifest and CPU/CUDA verifier reports.

**Narration:**

> The 462.6-megabyte checkpoint is tracked with Git LFS and a SHA-256 checksum. Our
> release verifier checks the embedded source hashes, fixed alpha, parameter count,
> deterministic repeated inference, output schema, and CPU and CUDA execution.

## 2:45–3:00 — Limitations and close

**Visual:** `noise_0.10` and Midjourney v1/v2 `noise_0.05` callouts.

**Narration:**

> Strong noise remains the main weakness, the ensemble is slower than Tiny alone, and
> we did not reuse already observed confirmation or WildFake data. This is a review
> signal, not an authenticity certificate. The repository includes the code, negative
> experiments, validation evidence, and reproducible checkpoint.

## Accuracy guardrails

- Say “internal development set,” never “official score.”
- Do not claim a fresh confirmation or WildFake result for the ensemble.
- Do not reuse Adapter v2's threshold 0.209 for the ensemble.
- Do not claim state of the art, universal detection, or guaranteed authenticity.
- Show `aigc-detector-ensemble-vnext.pt`, not the v1.0.0 Adapter asset, as the
  `v2.0.0` release checkpoint.
