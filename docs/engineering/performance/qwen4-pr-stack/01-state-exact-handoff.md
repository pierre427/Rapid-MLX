# PR 1 — `fix(mtp): make batch handoff state-exact`

Local commit: `de2c5c9a5047edfacbd3b847dbd8015568a698b1`

Base: Rapid-MLX `746522837c2cde5deca3784786ce06d10b45e66c`

Publication branch: `fix/mtp-state-exact-handoff`

Status: local-only; not pushed or submitted.

## Why

When a second request joins a batch whose only active lane is using vendored
self-MTP, the prior fallback closed the MTP generator and entered the ordinary
batch step with `_next_tokens` still representing the last emitted MTP token.
That can duplicate a visible token or continue from a stale cache position.
Continuous batching must not trade request correctness for admission.

## Scope

- Before admitting a waiting request, ask the active MTP lane to reach a
  cache-safe handoff boundary.
- Drain already accepted draft tokens at batch size one before expansion.
- Stage one not-yet-emitted target token in the ordinary batch generator's
  expected input/return shape.
- Verify the staged token survives batch extension, then disable the old MTP
  generator and continue through the baseline path.
- Fail closed if admission bypasses preparation or loses/replaces the staged
  token.
- Add focused scheduler/handoff tests in
  `tests/test_mtp_batch_handoff_exact.py`.

Files:

- `vllm_mlx/scheduler.py`
- `tests/test_mtp_batch_handoff_exact.py`

## Non-goals

- No batched self-MTP execution.
- No new scheduler policy, memory model, APC behavior, or model-family support.
- No dynamic MTP lane joins; this makes the existing fallback exact.
- No throughput claim or Rapid performance qualification.

## Acceptance

- A prepared B=1 to B>1 handoff emits its staged token exactly once.
- Accepted drafts drain before a waiting request is admitted.
- Unprepared batch growth raises before the baseline step can consume stale
  `_next_tokens`.
- The scheduler defers the waiting request until the handoff hook reports a
  safe boundary.
- Existing non-MTP admission behavior remains unchanged.

## Verification

- Current stacked checkout:
  `python3 -m pytest -q tests/test_mtp_batch_handoff_exact.py` — 4 tests pass.
- Combined focused stack: 70 tests pass in 0.29 seconds.
- Stack-tip qualification battery: 163 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- Model-free only. No GPU, MLX model, service, stress, or performance run.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, and the live B=1 to B>1
  model gate.

## Behaviour delta

Before: a live B=1 MTP lane could be closed after batch expansion while the
ordinary generator still held a last-emitted placeholder, permitting a
duplicate/stale continuation.

After: admission waits for a not-yet-emitted, cache-safe target token; the
ordinary path receives that exact token or the transition fails closed.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/scheduler.py` and `tests/test_mtp_batch_handoff_exact.py` under
Pierre Lamy's direction. The model-free focused tests above were rerun on the
local stacked checkout. Human line-by-line review, mutation testing, full
validation, and live hardware verification remain pending; update this section
with the human review actually performed before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (4 focused cases on the stacked checkout).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **not required for this focused correctness fix**.
- Existing API compatibility: **no intentional breaking change**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
