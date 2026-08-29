# PR 3 — `feat(mtp): add ragged cache rollback adapter`

Local commit: `f061e090e30f8379829e32f705c5cb625f1cd172`

Depends on: PR 2 / `203cb5be39b9deb391aee76a5d793187b7693559`

Publication branch: `feat/mtp-ragged-cache-rollback`

Status: local-only; not pushed or submitted.

## Why

Speculative lanes accept different draft lengths. A scalar rollback across a
batched cache can therefore restore the wrong state or leave stale cells in a
live row. Qwen4 makes the transaction wider than transformer KV: QSA ledgers
and recurrent/PLE arrays must land on the same per-row boundary. Unsupported
cache shapes must fail before any member is mutated.

## Scope

- Add a version-gated, idempotent adapter for the known Rapid/mlx-lm cache
  classes used by the planned fixed-membership engine.
- Separate preflight from mutation so a heterogeneous cache tree is validated
  atomically.
- Support per-row rollback for batched KV, declared auxiliary ledgers,
  `ArraysCache`, Qwen4 atomic recurrent state, and QSA retained raw groups.
- Preserve existing scalar trim methods.
- Refuse unknown subclasses, foreign runtime classes, pending-padding
  geometries, partial atomic state, over-trim, and unsupported cache families.

Files:

- `vllm_mlx/spec_decode/mtp/ragged_cache.py`
- `tests/test_mtp_ragged_cache.py`

## Non-goals

- No quantized or windowed batched-cache support.
- No model execution, scheduler wiring, APC changes, or continuous engine.
- No silent generic fallback for unknown cache implementations.
- No GPU/model benchmark or Rapid performance qualification.

## Acceptance

- Installation is strictly runtime-version gated and idempotent.
- Known scalar methods remain intact.
- A cache tree fully preflights before any mutation occurs.
- Uniform cursor movement and residual row reconciliation are distinct.
- Qwen4 atomic state refuses partially staged snapshots.
- QSA raw/index state rewinds consistently with retained logical rows.
- Unsupported cache families fail loudly.

## Verification

- Current stacked checkout:
  `python3 -m pytest -q tests/test_mtp_ragged_cache.py` — 17 test cases pass.
- Combined focused stack: 70 tests pass in 0.29 seconds.
- Stack-tip qualification battery: 163 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- Model-free/fake-array evidence only. No MLX model, GPU, service, stress, or
  performance run.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, real MLX cache tests, and
  model-cache equality gates.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/ragged_cache.py` and
`tests/test_mtp_ragged_cache.py` under Pierre Lamy's direction. The focused
fake-cache tests were rerun locally. Human line-by-line review, mutation
testing, and real-runtime cache verification remain pending; update this
disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (17 focused fake-cache cases).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **not required before the adapter has a live caller**.
- Existing API compatibility: **existing scalar trim methods are preserved**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
