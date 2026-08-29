# PR 11 — `feat(mtp): plan continuous routing and APC restore`

Local commit: `7ed6326cf2ab3e5aeb1eeae058816049d5a776c1`

Depends on: PR 10 / `6f9d4d706c61a2b79c99331c07350888661cc158`

Publication branch: `feat/mtp-continuous-routing-planner`

Status: publication branch not pushed and nothing submitted upstream; this
commit is mirrored only on the private Forgejo integration branch. **At this
PR head, planning only: no live continuous token delivery.**

## Why

Before the scheduler can select a continuous cohort, it needs a default-off,
immutable decision layer that combines family/sampling/cache capability,
existing memory admission, and exact target-cache/MTP-cache/seed-hidden APC
state. That policy must be reviewable without handing token delivery to the new
wrapper prematurely.

## Scope

- Add `continuous_batching` as a strict boolean MTP-only speculative-config
  option, defaulting to false.
- Carry the opt-in through CLI and SchedulerConfig.
- Add a pure integration planner for fixed-cohort continuous, legacy-MTP,
  plain-decode, and queue decisions.
- Reuse the batched admission contract rather than adding a second memory
  authority.
- Validate exact APC prepared-state sidecars and carry eligible target/MTP
  caches plus seed-hidden state in immutable lane plans.
- Install only planner metadata on the BatchGenerator after static admission.
- Retain the vendored single-request MTP/plain data plane as authoritative.

Files:

- `vllm_mlx/spec_decode/mtp/continuous_routing.py`
- `vllm_mlx/spec_decode/config.py`
- `vllm_mlx/cli.py`
- `vllm_mlx/scheduler.py`
- `tests/test_continuous_mtp_routing.py`

## Non-goals

- **No live call to `ContinuousMTPGenerationBatch` and no continuous token
  delivery.**
- No scheduler cohort execution, live attach/detach, APC persistence rewrite,
  new allocator, dynamic Flash join, telemetry wiring, automatic enablement,
  or performance claim.

## Acceptance

- The absent flag preserves incumbent behavior; non-boolean or non-MTP use is
  rejected.
- Static refusal occurs before mutating BatchGenerator.
- Supported metadata produces an immutable fixed-cohort plan.
- Exact APC state is carried; stale/mismatched state falls back safely.
- Unsupported sampling/cache/membership routes to legacy MTP, plain decode, or
  queue according to the existing admission outcome.
- AST coverage proves the scheduler installs metadata only and continues to
  call the vendored live path.

## Verification

- Commit contains model-free planner/APC tests and AST checks for the explicit
  default-off CLI/config/scheduler seam.
- Source inspection confirms scheduler comments and control flow retain
  `_install_mtp_vendored` as the live data plane.
- The focused file passed within the 261-test current-tip model-free battery; all
  changed Python files pass `ruff check` and `ruff format --check`.
- No GPU, model, service, stress, live continuous delivery, or Rapid
  performance qualification was performed for this packet.
- Pending before submission: per-PR full unit suite, scheduler contract
  mutation spot-check, PR-number `pr_validate`, human review, and the separate
  live-delivery integration.

## Behaviour delta

With the flag absent or false, behavior is unchanged. With it true, Rapid may
attach a metadata-only planner after static admission, but token generation
still runs through the incumbent vendored MTP/plain path. This commit does not
make continuous self-MTP operational in serving.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/continuous_routing.py`,
`vllm_mlx/spec_decode/config.py`, `vllm_mlx/cli.py`,
`vllm_mlx/scheduler.py`, and `tests/test_continuous_mtp_routing.py` under
Pierre Lamy's direction. Human review, mutation coverage, full validation, and
the separate live-delivery integration remain pending; update this disclosure
before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed within the 261-test current-tip battery**.
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **planning-only boundary is documented in this body**.
- Existing API compatibility: **absent/false flag preserves incumbent behavior**.
- Hardware/model acceptance: **pending; live delivery is not implemented here**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
