# PR 5 — `feat(mtp): add continuous transaction core`

Local commit: `0a3be1757f8c904adcebc024bd1021ffab2bd18b`

Depends on: PR 4 / `07f8beea1750ef51d2451c9c0045082fa926a693`

Publication branch: `feat/mtp-continuous-transaction-core`

Status: local-only; not pushed or submitted.

## Why

The existing MTP generator executes one request at a time. Continuous
self-MTP needs one coordinator to bind lane order, membership epoch, proposal,
ragged rollback, delivery, and commit without baking scheduler or model APIs
into the transaction. A model-neutral core makes those invariants reviewable
before a real MLX backend is allowed to mutate arrays.

## Scope

- Add a scheduler-neutral continuous self-MTP transaction core.
- Define default-off configuration and fail-closed runtime capabilities.
- Add explicit Rapid target/MTP forward seams using `return_hidden=True` and
  `n_confirmed=...`.
- Define lane specifications/state, target+draft cache pairs, prepared lane
  data, proposal results, commit results, and detached lane packages.
- Inject compute and cache adapters through protocols rather than importing
  MLX or assuming one model/cache topology.
- Implement prepare, attach, propose, commit, and detach lifecycle operations.
- Bind every cycle to membership epoch, lane order, acceptance/drop vectors,
  and delivered-token counts.
- Refuse partial fixed-cohort membership changes, unsupported sampled paths,
  XTC, and unattested Flash dynamic membership.

Files:

- `vllm_mlx/spec_decode/mtp/continuous_engine.py`
- `tests/test_continuous_self_mtp_engine.py`

## Non-goals

- No MLX compute backend, Qwen tensor operations, GenerationBatch wrapper,
  scheduler route, APC caller, CLI/config flag, or telemetry.
- No automatic enablement or Rapid performance claim.
- No claim that Qwen3.5 or Flash dynamic membership is executable; capability
  data and a concrete backend are separate patches.

## Acceptance

- The feature defaults off and refuses before calling compute.
- Rapid forward seams request hidden state and pass explicit confirmed depth.
- Fixed-cohort prepare/attach/propose/commit/detach completes through fake
  adapters with one coherent lane/cache lifecycle.
- XTC remains unconditionally fail-closed.
- Incremental attach and partial detach refuse in fixed-membership mode.
- Flash dynamic membership requires its own architecture attestation.
- Partial nonterminal delivery and coercible/non-boolean vector values are
  rejected before closing a cycle.

## Verification

- Current stacked checkout:
  `python3 -m pytest -q tests/test_continuous_self_mtp_engine.py` — 12 test
  cases pass.
- Stack-tip qualification battery: 163 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- Model-free protocol/fake-adapter evidence only. No MLX model, GPU, service,
  stress, or performance run.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, backend integration, and
  hardware cache/digest gates.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/continuous_engine.py` and
`tests/test_continuous_self_mtp_engine.py` under Pierre Lamy's direction. The
focused model-free tests were rerun locally. Human line-by-line review, full
lint/validation, and integration review remain pending; update this disclosure
before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (12 focused fake-adapter cases).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **deferred to the documentation PR**.
- Existing API compatibility: **additive and default-off**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
