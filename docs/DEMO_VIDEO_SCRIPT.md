# Demo Video Script — 3 minutes

Target: 180 seconds, screen-recording friendly. Terminal font ≥ 18pt. Pre-stage all
commands before recording. Times are approximate; keep transitions snappy.

---

## Scene 1 — The problem (0:00 – 0:25)

**Visual:** Title card → slide with two near-identical photos (one real, one AI-generated).

**Narration (~25s):**
> "AI-generated images now pass casual inspection. Track 5 of TikTok TechJam asks a
> harder question: can a detector survive the real world — JPEG recompression, blur,
> downscaling, noise? We built one that does."

**Notes:** show the six transformation icons (JPEG / blur / scale / noise / color /
crop) one by one as they are named.

## Scene 2 — The approach (0:25 – 0:55)

**Visual:** simple pipeline diagram: SID_Set + GenImage → degradation-aware augmentation
→ ConvNeXt-Tiny → blur fine-tune → frozen threshold 0.209.

**Narration (~30s):**
> "ConvNeXt-Tiny — 27.8 million parameters, far under the 2-billion cap — trained on a
> multi-source mixture of real and synthetic images, with the same six transformation
> families applied during training that the organizers use for evaluation. Our first
> single-source model collapsed cross-source, at 0.65 AUC; multi-source training fixed
> that, and a blur-focused fine-tune repaired our weakest condition."

**Notes:** flash the numbers: 0.646 → multi-source → blur fine-tune. Keep it visual,
not a wall of text.

## Scene 3 — Live inference demo (0:55 – 1:45)

**Visual:** terminal, pre-opened in the project directory. A small demo folder
(e.g. `demo_images/`) contains ~6 mixed real/fake thumbnails visible in a file browser.

**Actions:**
1. Show the folder briefly: `ls demo_images`
2. Run inference (command staged in advance):
   ```powershell
   python predict.py --checkpoint outputs/multisource_blur_finetune/best.pt --input-dir demo_images --output predictions.json
   ```
3. Open `predictions.json` — zoom in on the `{"image_path", "pred"}` entries.

**Narration (~50s):**
> "Inference is one command: point it at any folder, get back a JSON array of
> confidence scores. Scores near 1 mean AI-generated, near 0 means real. Corrupt or
> unreadable files get a neutral 0.5 instead of crashing the pipeline — and note we
> always output scores, not just hard labels, so you can pick your own operating point."

**Notes:** highlight one clearly-fake image with a high score and one real image with a
low score. Pre-select demo images that are unambiguous.

## Scene 4 — Robustness results (1:45 – 2:25)

**Visual:** bar chart of the 16 conditions (grouped by family), clean baseline drawn as
a horizontal reference line.

**Narration (~40s):**
> "On the held-out WildFake subset — 13,841 images the model never saw — clean AUC is
> 0.96. Every JPEG quality actually scores above clean. Blur and noise are the hard
> cases, and our worst condition, blur sigma 2.0, sits at 0.82. Mean degraded AUC 0.93,
> robust score 0.90."

**Notes:** animate the bars by family. Emphasize "held out — never used for training
or thresholding" with an on-screen badge.

## Scene 5 — Error analysis honesty (2:25 – 2:50)

**Visual:** split screen: one recurring false-positive real image vs one recurring
false-negative DALL·E image, both with scores.

**Narration (~25s):**
> "Our honest failure modes: a handful of COCO photos the model calls fake under every
> transformation, and a few DALL·E images it misses everywhere. These are semantic, not
> signal-level errors — our roadmap attacks them with CLIP-style features."

## Scene 6 — Limitations & close (2:50 – 3:00)

**Visual:** 3-bullet slide: noise regression acknowledged · blur_2.0 still weakest ·
generator generalization to be tested. End card with repo/team name.

**Narration (~10s):**
> "Noise robustness dipped slightly as the price of the blur fix — we document that.
> Code, weights, and full analysis are in the repo. Thanks."

---

## Production checklist

- [ ] Prepare `demo_images/` with 6–8 unambiguous images (mix of real/fake)
- [ ] Pre-place the `predict.py` command in the terminal (history or clip file)
- [ ] Chart assets for scenes 4–5 (can be generated from
      `reports/wildfake_analysis_blur_finetune/robustness_table.csv`)
- [ ] Verify checkpoint path exists on the recording machine
- [ ] Record at 1080p, single take per scene, edit cuts over live retakes
