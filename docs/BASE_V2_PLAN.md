# Base v2 preregistered repair plan

Status: active on `base-v2`. Tiny vNext `v1.1.0-rc1`, `main`, and all Base-v1
artifacts are immutable inputs and rollback points.

## Evidence from Base v1

Base v1 proved that capacity is useful on the 12,896-image modern development
set: generator-macro robust score rose from 0.718331 to 0.963147, with a
1,000-replicate 95% interval for the gain of [0.236006, 0.254703]. It was
rejected because GenImage robust score fell 0.007613 and SID fell 0.032309,
with broad JPEG, blur, scale, and noise regressions.

A paired 3,000-image audit measured Base-v1 AUC changes of -0.029293 under
label-independent random re-encoding and -0.039435 under stronger neutralised
re-encoding, versus -0.010219 and -0.017512 for Tiny vNext. The standalone
Base recipe is therefore not continued.

## Mechanism

The first Base-v2 experiment freezes Base v1 completely and adds a
zero-initialised `1024 -> 256 -> 1` residual branch (262,657 trainable
parameters). Source routing is deliberately the inverse of Tiny vNext:

- GenImage/SID: supervised BCE plus Smooth-L1 matching to the frozen Tiny
  teacher's exact logits;
- CommunityForensics/SuSy/MS-COCOAI: squared residual penalty, preserving the
  already strong Base-v1 modern prediction;
- all samples retain the fixed degradation exposure schedule; label-independent
  random re-encoding rises from 0.15 to 0.35 to address the diagnosed shortcut.

The Base and Tiny teacher receive no gradients. The zero-init wrapper must be
numerically identical to Base v1 before training. One epoch and one seed are
opened first. Residual gain is swept after training without additional fitting.

## Ordered experiments and stop-loss

1. Zero-training Tiny/Base logit blending is a diagnostic upper-bound only; it
   cannot become the preferred submission because it doubles backbone cost.
2. Train the single-scale frozen Base repair adapter at seed 2026.
3. Sweep gains on historical selection and modern development predictions.
4. If and only if old-domain repair is insufficient while modern preservation
   succeeds, allow one multi-scale frozen repair adapter using intermediate Base
   features. Do not unfreeze Base in this stage.
5. Full Base old-domain alignment is a separate last resort and requires a new
   preregistration; Base-v1 is never simply continued.

## Acceptance gates

The selected Base-v2 checkpoint must satisfy all gates against frozen Tiny
vNext on already-open internal selection data:

1. modern generator-macro robust gain at least 0.005 with grouped-bootstrap
   95% lower bound above zero;
2. worst-generator robust score and worst generator-condition AUC do not regress;
3. historical overall robust score drops at most 0.002;
4. CommunityForensics, GenImage, and SID robust scores each drop at most 0.002;
5. every historical source-by-degradation-family drop is at most 0.003;
6. clean and blur-2.0 AUC drops are each at most 0.002;
7. paired random-reencode sensitivity is no worse than Tiny vNext by more than
   0.002;
8. standard prediction contract passes, scores are finite, and total parameters
   stay below two billion.

Base-v1 modern performance is an additional preservation diagnostic: the
repair candidate should not lose more than 0.005 generator-macro robust score
relative to Base v1, but Tiny-relative gates remain the formal eligibility rule.

## Data boundary and missing evidence

Training uses the audited 280,000-row balanced manifest. Historical selection
and the fresh official SuSy/MS-COCOAI development splits may select the model.
The historical confirmation set and WildFake were consumed before Base v2 and
cannot influence selection. No organiser-hidden data is available.

The project does not currently have a contemporary validation set that is
simultaneously image-, content-, repository-, and generator-family-disjoint
from all training sources. Consequently, internal success cannot guarantee the
hidden-test result. This is an evidence limitation, not a blocker to the
controlled experiment.
