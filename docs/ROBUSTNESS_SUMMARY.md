# Robustness Summary — Ensemble vNext

- **Status:** frozen public release `v2.0.0`
- **Checkpoint:** `aigc-detector-ensemble-vnext.pt`
- **SHA-256:** `DE3C8C6E44C445278D6A47A9BC7F9E96B3CC9D02EFA675587F6329D46148587A`
- **Model:** 0.50 Tiny vNext logit + 0.50 Base v1 logit
- **Parameters:** 115,585,507

## Protocol

The robustness score is `0.8 * mean degraded AUC + 0.2 * worst degraded AUC`.
The suite contains clean plus 15 deterministic degradations across JPEG, blur,
resizing, noise, color, and crop.

Alpha was selected on the historical 12,000-image development anchor using the
unchanged Tiny-relative acceptance gates. The selected alpha was then frozen and run
once on the full 12,896-image modern development manifest. Generator comparisons use
macro-averaging and a 1,000-replicate content-group bootstrap.

This is retrospective development validation, not a new preregistration or official
hidden evaluation.

## Historical selection anchor

| Metric | Tiny vNext | Ensemble vNext | Delta |
|---|---:|---:|---:|
| Robust score | 0.944886 | **0.978314** | **+0.033429** |
| Clean AUC | 0.972336 | **0.991032** | +0.018696 |
| Mean degraded AUC | 0.952022 | **0.982465** | +0.030443 |
| Worst degraded AUC | 0.916339 | **0.961712** | +0.045373 |
| CommunityForensics robust score | 0.935609 | **0.985517** | +0.049909 |
| GenImage robust score | 0.918252 | **0.960966** | +0.042714 |
| SID_Set robust score | 0.958064 | **0.967631** | +0.009568 |

Alpha values `0.10` through `0.50` passed. Alpha `0.60` had a slightly higher
overall score but failed the frozen SID_Set scale-family guard
(`0.005079 > 0.005000`), so it was rejected.

## Modern development set

| Metric | Tiny vNext | Ensemble vNext | Delta |
|---|---:|---:|---:|
| Global robust score | 0.740077 | **0.901913** | **+0.161836** |
| Clean AUC | 0.850770 | **0.958501** | +0.107731 |
| Mean degraded AUC | 0.772745 | **0.920235** | +0.147490 |
| Worst degraded AUC | 0.609406 | **0.828625** | +0.219219 |
| Generator-macro robust score | 0.718331 | **0.894738** | **+0.176407** |
| Worst-generator robust score | 0.548970 | **0.796370** | +0.247400 |
| Worst generator-condition AUC | 0.322516 | **0.648097** | +0.325581 |

Grouped-bootstrap macro-gain mean: `0.176475`; 95% interval:
`[0.171015, 0.182232]`. All four frozen modern gates passed.

### Per-condition global AUC

| Condition | Tiny | Ensemble | Delta |
|---|---:|---:|---:|
| `clean` | 0.850770 | 0.958501 | +0.107731 |
| `jpeg_90` | 0.758150 | 0.903483 | +0.145333 |
| `jpeg_70` | 0.845841 | 0.942067 | +0.096226 |
| `jpeg_50` | 0.808069 | 0.928055 | +0.119987 |
| `jpeg_30` | 0.793364 | 0.926427 | +0.133063 |
| `blur_0.5` | 0.828991 | 0.944393 | +0.115403 |
| `blur_1.0` | 0.813097 | 0.935456 | +0.122359 |
| `blur_2.0` | 0.777852 | 0.925641 | +0.147789 |
| `scale_0.5` | 0.803798 | 0.941215 | +0.137418 |
| `scale_0.25` | 0.750335 | 0.916748 | +0.166413 |
| `noise_0.02` | 0.751638 | 0.937406 | +0.185768 |
| `noise_0.05` | 0.609406 | 0.848464 | +0.239058 |
| `noise_0.10` | 0.611602 | 0.828625 | +0.217023 |
| `color_-0.20` | 0.920021 | 0.978271 | +0.058249 |
| `color_0.20` | 0.799087 | 0.940248 | +0.141162 |
| `crop_0.80` | 0.719933 | 0.907025 | +0.187092 |

All 16 global AUC deltas are positive. Strong noise remains the absolute bottleneck,
despite producing the largest improvements.

## Evidence boundary

- Both manifests are development/model-selection data.
- The historical confirmation split was already consumed by an earlier rejected
  candidate and was not reopened.
- WildFake was already observed during the v1.0.0 stage and was not reopened.
- No ensemble threshold was tuned.
- No official leaderboard or hidden-test result is claimed.

Machine-readable evidence is under `reports/ensemble_vnext/`. Historical Adapter v2
evidence remains under `reports/final_adapter_v2/`.
