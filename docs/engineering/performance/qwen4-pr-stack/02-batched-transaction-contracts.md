# PR 2 — `feat(mtp): add batched transaction contracts`

Local commit: `203cb5be39b9deb391aee76a5d793187b7693559`

Depends on: PR 1 / `de2c5c9a5047edfacbd3b847dbd8015568a698b1`

Publication branch: `feat/mtp-batched-transaction-contracts`

Status: local-only; not pushed or submitted.

## Why

A continuous speculative batch cannot safely be represented as independent
generators. Proposal output is valid only for one ordered membership epoch and
one verification width; membership changes, partial delivery, double commit,
and stale proposal reuse must be rejected before cache integration exists.
The server also needs an explicit default-off capability and admission contract
instead of treating "MTP available" as "batched MTP safe."

## Scope

- Add model-free capability, route, sampling, lane, memory-estimate, and
  admission data contracts.
- Keep every capability fail-closed and the feature disabled by default.
- Define deterministic degradation across draft depth, plain decode, and queue
  outcomes under a supplied memory budget.
- Add a membership-epoch bookkeeper with attach/detach, proposal, commit, and
  abort lifecycle validation.
- Bind proposal tickets to lane order, draft width, and epoch.
- Validate committed/terminal token counts without importing MLX.

Files:

- `vllm_mlx/spec_decode/mtp/batched.py`
- `tests/test_batched_self_mtp_contract.py`

## Non-goals

- No model forwards, cache mutation, scheduler integration, APC integration,
  server route, CLI flag, or telemetry.
- No XTC or arbitrary logits-processor support; both remain fail-closed.
- No performance claim and no automatic feature enablement.

## Acceptance

- Missing or truthy-non-boolean capabilities cannot open the batched route.
- Sampled lanes require lane RNG and an exact transformed verifier.
- Dynamic membership requires a separate explicit capability.
- Admission lowers depth before falling back to plain or queueing, respects
  configured lane limits, and reserves memory.
- Membership cannot mutate while a proposal is open.
- Stale, double, incomplete, or impossible commits are rejected.
- Inputs remain testable without MLX or array dependencies.

## Verification

- Current stacked checkout:
  `python3 -m pytest -q tests/test_batched_self_mtp_contract.py` — 24 test
  cases pass.
- Combined focused stack: 70 tests pass in 0.29 seconds.
- Stack-tip qualification battery: 163 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- Model-free only. No GPU, MLX model, service, stress, or performance run.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, and production integration
  tests.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/batched.py` and
`tests/test_batched_self_mtp_contract.py` under Pierre Lamy's direction. The
focused model-free tests were rerun on the local stacked checkout. Human
line-by-line review and the repository's complete validation pipeline remain
pending; update this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (24 focused cases on the stacked checkout).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **not required for this additive contract module**.
- Existing API compatibility: **no intentional breaking change**.
- Hardware/model acceptance: **not claimed by this model-free PR**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
