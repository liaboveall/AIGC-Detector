# Modern-generator fine-tune v1: selection result

> Historical rejected experiment. The later accepted release is Adapter v2; see
> [`final_adapter_v2/`](final_adapter_v2/README.md).

Date: 2026-08-28

Training data: 293,792 format-balanced CommunityForensics-Small images

Selection data: 6,000 CommunityForensics-Small generator-held-out images plus the
fixed 6,000-image legacy robustness selection set

Protocol: 16 conditions; confirmation and WildFake were not opened

## Decision

Reject epochs 1 and 2. Keep `outputs/multisource_blur_finetune/best.pt` as the
accepted model. The modern-only fine-tune learned the new distribution extremely
well but catastrophically forgot both legacy internal sources.

The accepted checkpoint was copied before training to the read-only backup
`outputs/checkpoint_backups/multisource_blur_finetune_best_pre_modern_20260828.pt`.
Both files had SHA-256
`99E8F456564B0EE0FC5C97779DF5AF36F01ED17F9BD93E2E57FDA6F5094F1F05`
after training.

## Combined selection result

| Epoch | Clean AUC | Robust score | Worst degraded AUC |
|---|---:|---:|---:|
| 0 (accepted baseline) | 0.965637 | 0.930488 | 0.892458 |
| 1 | 0.934446 | 0.910358 | 0.885167 |
| 2 | 0.938005 | 0.914935 | 0.892308 |

## Per-source 16-condition robust score

| Source | Epoch 0 | Epoch 1 | Epoch 2 |
|---|---:|---:|---:|
| CommunityForensics-Small (unseen generators) | 0.903910 | 0.998023 | 0.998418 |
| GenImage | 0.920356 | 0.751810 | 0.772455 |
| SID_Set | 0.958171 | 0.739655 | 0.745438 |

Epoch 2 recovered slightly from epoch 1 but remained far below the accepted model on
both legacy sources. No candidate qualifies for the sealed confirmation stage.

## Interpretation and next experiment

Modern generator coverage is highly informative—the candidate nearly saturated its
generator-disjoint CommunityForensics selection set—but modern-only full-model
fine-tuning is too aggressive. The next defensible experiment should retain legacy
knowledge explicitly, using a source-balanced replay mixture and/or teacher
distillation from the accepted model. Simply training more epochs on the modern-only
set is not justified by these results.
