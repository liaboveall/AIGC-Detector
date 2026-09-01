# Contributing

Thanks for helping improve AIGC-Detector. Changes should preserve the repository's
evidence boundaries and keep inference reproducible.

Participation is governed by the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test.py
```

Use `source .venv/bin/activate` on macOS or Linux.

## Before opening a pull request

Run:

```powershell
python test.py
python -m compileall -q src scripts train.py train_adapter.py `
  train_base_v3.py train_repair_adapter.py train_replay_distill.py `
  evaluate.py predict.py test.py
python -m pip check
git diff --check
```

If a change affects the release checkpoint or inference path, also run the CPU release
verifier and, when CUDA is available, the CUDA verifier.

## Evidence and data rules

- Do not commit dataset image bodies, private manifests, per-image development
  predictions, credentials, or machine-local caches.
- Do not tune on the consumed confirmation split or previously observed WildFake data.
- Distinguish development metrics from sealed, official, or hidden-test results.
- Update the model card, robustness summary, error analysis, checksum manifest, and
  release notes whenever a release artifact or claim changes.
- New checkpoints must use Git LFS and include a SHA-256 digest.

Small, focused commits with tests and a clear explanation of changed evidence are
preferred.
