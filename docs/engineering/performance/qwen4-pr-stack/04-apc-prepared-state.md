# PR 4 — `feat(mtp): validate APC prepared-state sidecars`

Local commit: `07f8beea1750ef51d2451c9c0045082fa926a693`

Depends on: PR 3 / `f061e090e30f8379829e32f705c5cb625f1cd172`

Publication branch: `feat/mtp-apc-prepared-state`

Status: publication branch not pushed and nothing submitted upstream; this
commit is mirrored only on the private Forgejo integration branch.

## Why

An APC hit is insufficient to resume persistent self-MTP. The target cache,
MTP draft cache, and final covered token's hidden state must have been captured
together. A stale or foreign sidecar can otherwise draft one position behind
or pair cache state from different model/configuration identities. Very small
prefix hits should fail open to normal MTP initialization rather than disable
speculation for the request.

## Scope

- Add a pure-Python prepared-state metadata and eligibility module.
- Bind state to immutable model/revision/adapter/tokenizer identity,
  speculative-config fingerprint, target/MTP cache layouts, and seed-hidden
  layout.
- Fingerprint the exact covered token boundary without retaining prompt text.
- Enforce capture invariant: target covers `N`, MTP covers `N-1`, and seed
  hidden is present.
- Revalidate schema, identity, age, token boundary, object presence, and live
  target/MTP counts before restoration.
- Return stable, non-throwing refusal reasons for persisted/lookup-time errors.
- Mark a valid prefix below the default 64-token usefulness floor with
  `bypass_hit=True`.

Files:

- `vllm_mlx/spec_decode/mtp/prepared_state.py`
- `tests/test_mtp_prepared_state.py`

## Non-goals

- No changes to Rapid APC, radix indexing, persistence, eviction, scheduler,
  generator, model, or cache implementations.
- No serialization format or disk write.
- No model load, GPU run, or Rapid performance claim.

## Acceptance

- Exact model/config/layout and `N`/`N-1` boundaries are restore-eligible.
- Incomplete target/MTP/hidden captures are rejected at the producer boundary.
- Stale, malformed, model-mismatched, config/layout-mismatched, token-diverged,
  or live-offset-mismatched state refuses restoration without mutating caches.
- A valid trivial hit returns `TRIVIAL_HIT` with `bypass_hit=True`.
- Canonical configuration fingerprints are key-order independent and
  value-sensitive.
- Importing the module does not import `mlx` or `mlx.core`.

## Verification

- Current stacked checkout:
  `python3 -m pytest -q tests/test_mtp_prepared_state.py` — 25 test cases pass.
- Isolated import assertion confirms `mlx` and `mlx.core` remain unloaded.
- Combined focused stack: 70 tests pass in 0.29 seconds.
- Current-tip qualification battery: 261 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- No GPU, model, service, stress, APC integration, or performance run.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, and persistence/APC
  integration tests.

## AI assistance disclosure

Codex wrote and reviewed `vllm_mlx/spec_decode/mtp/prepared_state.py` and
`tests/test_mtp_prepared_state.py` under Pierre Lamy's direction. The focused
tests and no-MLX import assertion were rerun locally. Human line-by-line review,
full lint/validation, and APC integration review remain pending; update this
disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (25 focused cases plus no-MLX import assertion).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **not required for this additive metadata seam**.
- Existing API compatibility: **Rapid APC remains untouched**.
- Hardware/model acceptance: **not claimed by this pure-Python PR**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
