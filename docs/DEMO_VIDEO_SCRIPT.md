# Final Demo Video Script — 3 Minutes

This shot list matches the frozen `v1.0.0` Adapter v2 release. Target length: 175-180
seconds. Record at 1080p or higher with terminal font at least 18 pt.

## Pre-recording gate

Run these commands before recording:

```powershell
python test.py
python scripts/verify_release.py
python predict.py --input-dir demo_images --output predictions.json --num-workers 0
```

Use only images that the team has permission to display. Keep the actual inference run
live, but pre-download the release checkpoint so network speed is not part of the demo.

## Scene 1 — Problem and objective (0:00-0:22)

**Visual:** title, then one authentic and one generated image followed by icons for
JPEG, blur, resize, noise, color, and crop.

**Narration:**

> AI-generated images are easy to detect only when the test looks like the training
> set. Track 5 asks for something harder: a detector that survives real sharing
> pipelines—compression, blur, resizing, noise, color shifts, and crop. We built a
> compact detector and evaluated every one of those conditions.

## Scene 2 — Method (0:22-0:55)

**Visual:** pipeline diagram:

```text
SID + GenImage + CommunityForensics
        -> balanced replay and degradation augmentation
        -> frozen ConvNeXt-Tiny
        -> 0.7% residual adapter
        -> continuous real/fake score
```

**Narration:**

> Our first single-source model collapsed to about 0.65 clean AUC on WildFake. The
> biggest fix was generator diversity. The final training view balances 560 thousand
> examples across three sources. To learn modern generators without forgetting older
> domains, we freeze the 27.8-million-parameter base and train a 197-thousand-parameter
> residual adapter. The whole model is 28 million parameters—far below the two-billion
> limit.

## Scene 3 — Live directory-to-JSON inference (0:55-1:38)

**Visual:** show six licensed demo images in `demo_images`, then the terminal.

Run:

```powershell
python predict.py --input-dir demo_images --output predictions.json --num-workers 0
```

Open `predictions.json` and highlight `image_path` and `pred`.

**Narration:**

> The submission interface is one command. Point it at a directory and it recursively
> returns exactly the required image path and continuous AI probability. Scores near
> zero mean authentic and near one mean generated. We output scores rather than hiding
> the operating point behind a hard label. An unreadable supported file safely receives
> a neutral 0.5 instead of terminating the batch.

## Scene 4 — Robustness evidence (1:38-2:18)

**Visual:** first show the internal base-versus-adapter summary, then a 16-bar WildFake
chart. Use a badge reading “model + threshold frozen before observation.”

**Narration:**

> On our fixed 12-thousand-image, 16-condition development split, robust score rises
> from 0.9305 to 0.9424, clean AUC reaches 0.9731, and the worst degraded AUC improves
> by 0.0187. The candidate passes all 31 pre-registered source and degradation guards.
> After freezing the model, we observed WildFake once: clean AUC 0.9647, mean degraded
> AUC 0.9297, and robust score 0.9082. All 16 conditions are non-decreasing versus the
> frozen base.

## Scene 5 — Honest failure analysis (2:18-2:47)

**Visual:** show three callouts: blur sigma 2, noise 0.10, JPEG q30 threshold shift.

**Narration:**

> The model is not perfect. Heavy blur remains the hardest condition at 0.822 AUC and
> creates too many real-image false positives. Strong noise also reduces recall. JPEG
> q30 is different: AUC stays at 0.981, but fake recall at the fixed threshold falls to
> 54 percent, showing calibration shift rather than ranking collapse. These limitations
> are documented in the repository.

## Scene 6 — Close (2:47-3:00)

**Visual:** GitHub repository, release tag, SHA-256, and links to robustness/error docs.

**Narration:**

> The complete code, frozen checkpoint with checksum, release self-test, robustness
> tables, and error analysis are public and reproducible. Our main lesson: generator
> coverage and evaluation discipline matter more than chasing a larger headline model.

## Recording notes

- Do not describe WildFake as an official score or broad generator benchmark.
- Do not say the 16,000-image confirmation set validated Adapter v2; it was consumed by
  an earlier rejected candidate.
- Show continuous scores. Threshold 0.209 is a documented binary-demo operating point.
- Keep the final GitHub Release asset name visible:
  `aigc-detector-adapter-v2.pt`.
- Upload the video through the Devpost/team account and use that account's final URL;
  credentials and external submission actions are not stored in this repository.
