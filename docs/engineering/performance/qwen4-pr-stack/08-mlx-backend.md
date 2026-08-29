# PR 8 — `feat(mtp): add continuous MLX backend`

Local commit: `f48fadb21237c7e94cddacd6fd3dfb20a204b83e`

Depends on: PR 7 / `e7a20e733389cb66aba13e45030db17e7788015b`

Publication branch: `feat/mtp-continuous-mlx-backend`

Status: publication branch not pushed and nothing submitted upstream; this
commit is mirrored only on the private Forgejo integration branch.

## Why

The core and generation wrapper deliberately perform no array or model work.
The data plane must translate one fixed cohort into Rapid's target/MTP forward
seams, maintain pending hidden/token pairs, and merge, roll back, filter, and
extract real cache groups without weakening the transaction boundary.

## Scope

- Add a lazily loaded MLX array adapter and an injectable array-ops seam for
  model-free testing.
- Implement the fixed-cohort K=2 recursive draft and target-verify data plane.
- Use only Rapid's explicit target and MTP forward seams.
- Carry accepted outputs, hidden-state progression, and rollback metadata into
  the core's propose/commit protocol.
- Flush pending pairs before the next cycle and on terminal detach.
- Add a Rapid ragged-cache adapter with preflighted merge, extend, extract,
  filter, finalize, and rollback operations.
- Refuse quantized/windowed caches and transformed sampling without exact
  residual hooks.

Files:

- `vllm_mlx/spec_decode/mtp/mlx_backend.py`
- `tests/test_continuous_self_mtp_mlx_backend.py`

## Non-goals

- No transformed-distribution implementation; PR 9 supplies that hook.
- No scheduler/config route, APC integration, telemetry, dynamic Flash join,
  automatic enablement, or performance claim.

## Acceptance

- K=2 draft/verify/commit reconciles per-lane outputs and cache debt.
- A following cycle flushes persistent pending pairs before new drafts.
- Terminal delivery advances cur/hidden state and detaches at one boundary.
- Cache mutations preflight before partial application.
- Unsupported cache topologies and missing transformed hooks fail closed.
- Importing the module does not eagerly import MLX.

## Verification

- Commit contains CPU/mock tests for recursive draft/verify, pending-state
  flushing, terminal detach, cache operations, refusal gates, and import
  isolation.
- The focused file passed within the 261-test current-tip model-free battery; all
  changed Python files pass `ruff check` and `ruff format --check`.
- No GPU, MLX model, service, stress, or Rapid performance qualification was
  performed for this packet.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, and real MLX cache
  equality, digest, memory, and throughput gates.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/mlx_backend.py` and
`tests/test_continuous_self_mtp_mlx_backend.py` under Pierre Lamy's direction.
Human review and independent real-model validation remain pending; update this
disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed within the 261-test current-tip battery**.
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **deferred to the documentation PR**.
- Existing API compatibility: **backend remains unreachable from live routing**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
