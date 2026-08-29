# PR 9 — `feat(mtp): verify transformed distributions`

Local commit: `7d4a9370215d9e35d11bf7642aaf1c7ee44df56a`

Depends on: PR 8A / `f4d8aef9528287706007637b622cf6bc3b49f319`

Publication branch: `feat/mtp-transformed-verifier`

Status: publication branch not pushed and nothing submitted upstream; this
commit is mirrored only on the private Forgejo integration branch.

## Why

Sampling-mode speculative decoding is exact only when draft and target logits
receive identical transforms and rejected drafts are replaced from the target
residual distribution. Greedy acceptance cannot be reused for temperature,
top-p, top-k, or min-p without changing the output distribution.

## Scope

- Add an import-safe transformed-sampling profile and residual-verification
  hook.
- Apply temperature, top-p, top-k, and min-p consistently to draft and target
  distributions.
- Accept by the exact target/draft probability ratio and sample rejection from
  the normalized positive residual.
- Preserve a target-distribution fallback when numeric residual mass is zero.
- Inject the hooks into the MLX backend without changing its greedy path.
- Refuse XTC, profile mismatch, and logits processors without an exact shared
  processor hook.

Files:

- `vllm_mlx/spec_decode/mtp/residual_sampling.py`
- `vllm_mlx/spec_decode/mtp/mlx_backend.py`
- `tests/test_continuous_self_mtp_residual_sampling.py`

## Non-goals

- No XTC approximation, arbitrary processor emulation, scheduler routing, APC,
  telemetry, default enablement, or performance claim.

## Acceptance

- Supported transformed distributions normalize and respect filter support.
- Boundary behavior for top-p, top-k, and min-p is explicit and tested.
- Disjoint support forces rejection and uses the target residual.
- Statistical evidence preserves the transformed target marginal.
- Greedy execution bypasses residual hooks.
- The module does not eagerly import MLX.

## Verification

- Commit contains NumPy-only normalization, boundary, residual, marginal,
  fallback, refusal, backend-integration, and import-isolation tests.
- The focused file passed within the 261-test current-tip model-free battery; all
  changed Python files pass `ruff check` and `ruff format --check`.
- No GPU, MLX model, service, stress, or Rapid performance qualification was
  performed for this packet.
- Pending before submission: per-PR full unit suite, mathematical mutation
  spot-check, PR-number `pr_validate`, human mathematical review, larger seeded
  statistical batteries, and real-model parity.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/residual_sampling.py`,
`vllm_mlx/spec_decode/mtp/mlx_backend.py`, and
`tests/test_continuous_self_mtp_residual_sampling.py` under Pierre Lamy's
direction. Human mathematical review and independent statistical/model
verification remain pending; update this disclosure before submission.

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
- Existing API compatibility: **greedy behavior and live routes are unchanged**.
- Hardware/model acceptance: **pending**.
- Human mathematical review: **pending**.

## Author

X handle (optional, external contributor):
