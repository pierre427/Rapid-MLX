# PR 6 — `feat(mtp): attest batched family capabilities`

Local commit: `57df9e28ee38a6f15549a4d14eae6d34e7b40626`

Depends on: PR 5 / `0a3be1757f8c904adcebc024bd1021ffab2bd18b`

Publication branch: `feat/mtp-family-capabilities`

Status: local-only; not pushed or submitted.

## Why

The model-neutral core cannot infer whether an injected model exposes a
recursive hidden-state draft seam, fixed-cohort cache semantics, or qualified
dynamic membership. Family claims must be immutable, explicit, conservative,
and validated by the existing injectors. Flash and Qwen3.5 also require
different join claims: the source evidence qualifies 27B joins but not
Flash-Next joins.

## Scope

- Add immutable protocol-v1 batched-MTP capability descriptors to the Qwen4
  and Qwen3.5 injectors.
- Expose a `mtp_batch_forward` seam that delegates to the existing recursive
  `mtp_forward(..., return_hidden=True)` path.
- Publish a separate recursive batch draft depth of two without changing the
  legacy single-request `mtp_max_speculative_tokens` cap.
- Attest fixed membership for both families.
- Keep dynamic join false for Qwen4/Flash and true for Qwen3.5.
- Explicitly refuse quantized cache, windowed cache, and XTC capability.
- Extend injector validation to require the descriptor, forward seam, and
  recursive depth.

Files:

- `vllm_mlx/spec_decode/mtp/qwen3_5_inject.py`
- `vllm_mlx/spec_decode/mtp/qwen4_exp_inject.py`
- `tests/test_mtp_batched_family_capability.py`

## Non-goals

- No actual multi-lane MLX backend, cache batching, GenerationBatch wrapper,
  scheduler route, or APC integration.
- No dynamic Flash join and no qualification of quantized/windowed caches.
- No change to the existing single-request MTP depth or default behavior.
- No Rapid performance claim.

## Acceptance

- Both descriptors are immutable and enumerate every supported/refused field.
- Qwen4 advertises fixed membership and refuses dynamic join.
- Qwen3.5 advertises fixed membership and declares dynamic join capability for
  later backend/qualification gates.
- Each batch forward delegates to the existing MTP path with
  `return_hidden=True`.
- Injected classes expose descriptor, forward seam, and recursive depth while
  leaving the legacy single-request cap unchanged.
- Injector validation refuses incomplete batched capability attachment.

## Verification

- Current stacked checkout:
  `python3 -m pytest -q tests/test_mtp_batched_family_capability.py` — 7 test
  cases pass.
- Tests use mocked forwards and AST/descriptor checks. No model weights, GPU,
  service, or performance run.
- Stack-tip qualification battery: 163 model-free tests pass; all changed
  Python files pass `ruff check` and `ruff format --check`.
- Pending before submission: per-PR full unit suite, contract mutation
  spot-check, PR-number `pr_validate`, human review, real injector smoke,
  backend use, and hardware qualification.

## Behaviour delta

Before: injected Qwen4/Qwen3.5 models exposed only the existing single-request
MTP surfaces.

After: they additionally expose immutable, conservative batched-family
capability metadata and a recursive hidden-state forward seam. No scheduler or
generator consumes those fields yet, so serving defaults remain unchanged.

## AI assistance disclosure

Claude/Codex assisted in writing and reviewing
`vllm_mlx/spec_decode/mtp/qwen3_5_inject.py`,
`vllm_mlx/spec_decode/mtp/qwen4_exp_inject.py`, and
`tests/test_mtp_batched_family_capability.py` under Pierre Lamy's direction.
The focused mocked/AST tests were rerun locally. Human line-by-line review,
full validation, and real-model injector verification remain pending; update
this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (7 mocked-forward and AST cases).
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **deferred to the documentation PR**.
- Existing API compatibility: **legacy single-request limits are preserved**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
