# Base v2 multi-scale historical repair experiment

Status: preregistered before smoke or full training on `base-v2`. Frozen Tiny
vNext, Base v1, `main`, and all prior Base-v2 outputs remain immutable.

## Why the pooled adapter was insufficient

The single-scale `1024 -> 256 -> 1` repair run completed cleanly, but strict
historical retention still failed. Against frozen Tiny, GenImage robust score
dropped 0.005957 and SID dropped 0.028302. Relative to Base v1, it recovered
only 0.001656 on GenImage and 0.004007 on SID.

This was not an optimisation crash. Teacher MAE improved from 2.8325 over the
first 500 batches to 2.4845 over the final 500, while the modern protection
penalty rose from 0.0021 to 0.0156. The branch was still learning, but the
single final pooled feature was already encountering a repair-versus-protect
conflict. A gain sweep is therefore run first to test whether simple amplitude
can solve the issue.

## Single changed mechanism

If no single-scale gain passes the frozen historical gates, replace only the
repair branch with a zero-initialised multi-scale statistics adapter. The Base
v1 backbone remains entirely frozen. The branch pools channel mean and standard
deviation from ConvNeXt stages 0-3 (128/256/512/1024 channels), layer-normalises
the resulting 3,840-dimensional vector, then applies `3840 -> 512 -> 1`.

The intent is to recover low- and mid-level codec, resampling, noise, and
texture evidence that final semantic pooling may discard. Zero initialisation
keeps the wrapper numerically identical to Base v1 before training, and the
modern sources continue to receive an explicit zero-residual penalty.

Two bounded optimisation adjustments accompany the new representation:

- two epochs at learning rate 2e-4 because the first run was demonstrably not
  converged after one epoch;
- Huber delta 2.0, which widens the quadratic teacher-matching region while
  retaining bounded gradients for large logit outliers.

No Base parameter is unfrozen. No confirmation or WildFake data is opened.

## Selection and stop-loss

Epochs and post-training gains are assessed only on the already-open historical
selection and modern development sets. Formal acceptance remains exactly the
eight gates in `docs/BASE_V2_PLAN.md`; in particular, every historical source
and source-by-degradation-family bound must pass against frozen Tiny, modern
generator-macro gain must be at least 0.005 with a positive grouped-bootstrap
lower bound, and modern macro may lose at most 0.005 versus Base v1.

If neither multi-scale epoch nor its gain sweep passes, this mechanism is
rejected. The next experiment may no longer be described as a lightweight
adapter fix and requires a separate preregistration before any Base parameter
is unfrozen.
