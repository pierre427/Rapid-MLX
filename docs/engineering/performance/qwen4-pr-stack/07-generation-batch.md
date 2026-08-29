# PR 7 — `feat(mtp): add continuous generation batch`

Local commit: `e7a20e733389cb66aba13e45030db17e7788015b`

Depends on: PR 6 / `57df9e28ee38a6f15549a4d14eae6d34e7b40626`

Publication branch: `feat/mtp-continuous-generation-batch`

Status: local-only; not pushed or submitted.

## Why

Rapid's scheduler consumes a GenerationBatch-shaped lifecycle, while the
continuous transaction core exposes prepare, attach, propose, commit, and
detach primitives. A scheduler-neutral wrapper is needed to preserve token
delivery, stop/max-token decisions, logprobs, and cache extraction without
allowing half-committed proposals to escape.

## Scope

- Add a fixed-cohort generation wrapper around the continuous core.
- Emit every prepared first token exactly once.
- Limit each call to one proposal and commit only the delivered prefix.
- Preserve per-lane token/logprob ledgers and stop/max-token finish reasons.
- Package detached lane/cache state at a closed transaction boundary.
- Make manual detach idempotent and refuse calls after detachment.
- Refuse incremental membership, including Flash even when lower-level dynamic
  capabilities are attested.

Files:

- `vllm_mlx/spec_decode/mtp/continuous_batch.py`
- `tests/test_continuous_mtp_generation_batch.py`

## Non-goals

- No MLX compute backend, scheduler route, config flag, APC integration,
  telemetry, dynamic Flash join, or performance claim.
- No change to Rapid's incumbent GenerationBatch path.

## Acceptance

- Initial and proposal tokens are emitted once and in lane order.
- Stop and length boundaries commit the exact delivered prefix.
- A failed commit publishes no delivery ledger.
- Cancellation/manual teardown returns one coherent detach package.
- Incremental join always fails closed.

## Verification

- Commit contains focused pure-Python lifecycle tests for initial delivery,
  proposal/commit, stop, length, failure, detach, and Flash join refusal.
- The focused file passed within the 163-test stack-tip model-free battery; all
  changed Python files pass `ruff check` and `ruff format --check`.
- No GPU, MLX model, service, stress, or Rapid performance qualification was
  performed for this packet.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, scheduler compatibility,
  and model gates.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/continuous_batch.py` and
`tests/test_continuous_mtp_generation_batch.py` under Pierre Lamy's direction.
Human line-by-line review, full validation, and live model verification remain
pending; update this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed within the 163-test stack-tip battery**.
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **deferred to the documentation PR**.
- Existing API compatibility: **incumbent GenerationBatch remains unchanged**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
