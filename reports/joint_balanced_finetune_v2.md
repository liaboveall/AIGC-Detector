# Joint balanced fine-tune v2: selection result

> Historical rejected experiment. The later accepted release is Adapter v2; see
> [`final_adapter_v2/`](final_adapter_v2/README.md).

Date: 2026-08-28

Decision set: fixed, source-stratified internal selection split (6,000 images)

Protocol: 16 conditions; WildFake and the sealed 10,000-image confirmation split were not opened

## Decision

Neither retry checkpoint passed the pre-registered selection gates. The accepted
submission checkpoint therefore remains:

`outputs/multisource_blur_finetune/best.pt`

No threshold recalibration or WildFake re-evaluation was performed. This follows the
pre-registered stop rule and prevents confirmation/WildFake feedback from influencing
model development.

## Experiment

The retry started directly from the accepted checkpoint and trained for two epochs.
Its label-independent re-encoding rate was reduced to 15%, the learning rate to
1.5e-5, and the degradation mixture emphasized blur and noise while retaining small
color/crop exposure. The exact resolved configuration and epoch checkpoints are in
`outputs/multisource_joint_finetune_v2/`.

## Selection results

| Metric | Baseline | Epoch 1 | Epoch 2 |
|---|---:|---:|---:|
| Robust score | 0.939260 | 0.934607 | 0.936258 |
| Robust-score delta | -- | -0.004653 | -0.003002 |
| Clean AUC delta | -- | -0.004346 | -0.003913 |
| `blur_2.0` AUC delta | -- | -0.003729 | -0.002242 |
| Noise-family mean delta | -- | -0.001350 | +0.000087 |
| Color-family mean delta | -- | -0.001131 | +0.001838 |
| Crop-family mean delta | -- | +0.011517 | +0.012855 |

Epoch 2 recovered much of epoch 1's loss and improved color/crop, but it still failed
the required robust-score gain, clean guard, blur guard, and explicit noise-gain gate.
Its JPEG-family mean also regressed by 0.014573, exceeding the allowed 0.005 family
drop. It was therefore rejected.

## Interpretation

Adding color/crop exposure successfully improved those conditions, but the joint
mixture still diluted or disturbed the accepted model's clean/JPEG/blur decision
boundary. Noise was restored only to statistical parity, not the required clear gain.
The result supports keeping the simpler accepted checkpoint rather than accepting a
trade-off that lowers the registered aggregate objective.

Machine-readable gate output:
`outputs/multisource_joint_finetune_v2/selection_gate_report.json`.
