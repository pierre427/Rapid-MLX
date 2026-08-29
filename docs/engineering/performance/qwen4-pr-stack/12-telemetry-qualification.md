# PR 12 — `feat(mtp): add bounded qualification telemetry`

Local commit: `3290480649731ea9ebaa3237eb7212ea256609da`

Depends on: PR 11 / `7ed6326cf2ab3e5aeb1eeae058816049d5a776c1`

Publication branch: `feat/mtp-qualification-telemetry`

Status: local-only; not pushed or submitted.

## Why

Continuous speculative work needs bounded lifecycle accounting and an offline
qualification schema that cannot turn synthetic tests into performance claims.
Unbounded request/model labels would make production metrics unsafe, while
aggregate throughput without identity, cache, lane, and transaction
reconciliation would be too weak for acceptance.

## Scope

- Add fixed-enum admission outcome/reason counters.
- Add single-use proposal tickets and proposed, committed, aborted, failed,
  commit-kind, cleanup, and token totals.
- Export immutable snapshots with a statically bounded key space.
- Add exact candidate, model revision, config, prompt manifest, and environment
  identities for offline qualification.
- Reconcile N=1/2/4 lane output, proposal, acceptance, commit, cache-equality,
  digest, memory, and finish data.
- Require declared Apple-Silicon evidence plus a raw-artifact digest before
  `performance_qualified` can become true.

Files:

- `vllm_mlx/spec_decode/mtp/continuous_telemetry.py`
- `tests/test_continuous_mtp_telemetry.py`

## Non-goals

- No scheduler/config/model/backend/wrapper wiring, request-cardinality labels,
  route change, default enablement, hardware execution, or Rapid performance
  claim.
- No claim that source-prototype 120.3 or 82.0 token/s applies to Rapid.

## Acceptance

- Metric dimensions are fixed enums and snapshots contain no request/model IDs.
- Proposal tickets are single-use; invalid commits do not close a transaction.
- Abort/failure paths publish no commit totals.
- Cohort and suite totals reconcile exactly for N=1/2/4.
- Missing identity, divergent/not-run digest, cache inequality, open/failed
  transactions, or missing raw hardware artifact refuses qualification.
- Synthetic/model-free evidence can be structurally valid but never
  performance-qualified.

## Verification

- Recorded focused result: `tests/test_continuous_mtp_telemetry.py` — 28 passed.
- Recorded combined core+telemetry result — 40 passed.
- `py_compile`, import isolation (`mlx` and `mlx.core` unloaded), and an
  88-column scan passed.
- Stack-tip qualification battery: 163 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- CPU/static/model-free only. No GPU, model, service, hardware artifact, or
  Rapid performance qualification was produced.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, integration, and serialized
  hardware batteries.

## AI assistance disclosure

Codex wrote and reviewed `vllm_mlx/spec_decode/mtp/continuous_telemetry.py` and
`tests/test_continuous_mtp_telemetry.py` under Pierre Lamy's direction. Focused
model-free checks were rerun locally. Human line-by-line review, full
validation, production wiring review, and independent hardware qualification
remain pending; update this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (28 focused cases; 40 core-plus-telemetry cases).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **qualification limitations are documented in this body**.
- Existing API compatibility: **module is additive and not live-wired**.
- Hardware/model acceptance: **pending and intentionally impossible to infer from synthetic evidence**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
