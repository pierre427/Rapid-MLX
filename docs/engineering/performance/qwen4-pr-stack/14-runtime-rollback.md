# PR 14 — `feat(mtp): assemble the continuous runtime and exact GDN rollback`

Source integration commit: `d7cf5bbb11656f0ed90ce9b1749454dc9eaf3657`
(publication split required).

Depends on: PR 13 / `be0b3e8c863c3364876c6ac738aedad856650d8f`

Publication branch: `feat/mtp-continuous-runtime-rollback`

Status: draft split only; no publication branch is pushed and nothing is
submitted upstream. The combined source commit is mirrored on private Forgejo.

## Why

The model-free coordinator needs a fail-closed adapter from injected Rapid
models to target/MTP forward seams, real cache factories, family capability
metadata, and Qwen3.5 GatedDeltaNet rollback semantics.

## Scope

- Assemble `ContinuousSelfMTPRuntime` from an MTP-injected model.
- Bind target and recursive MTP forwards plus target/draft cache factories.
- Derive capabilities from immutable family descriptors.
- Add opt-in speculation rollback so Qwen3.5 records an exact per-forward GDN
  snapshot for ragged rewind.
- Arm ragged speculation after merges and avoid unsupported cache call
  signatures.
- Keep Qwen4/Flash dynamic membership refused.

Primary files to extract from `d7cf5bbb`:

- `vllm_mlx/spec_decode/mtp/continuous_runtime.py`
- `vllm_mlx/spec_decode/mtp/mlx_backend.py`
- `vllm_mlx/spec_decode/mtp/ragged_cache.py`
- `vllm_mlx/spec_decode/mtp/qwen3_5_inject.py`
- `vllm_mlx/spec_decode/mtp/qwen4_exp_inject.py`
- configuration/CLI wiring and focused tests

## Non-goals

- No scheduler response delivery, live queue mutation, or dynamic turnover.
- No APC prepared-state consumption; the current live scheduler does not pass
  an APC hit into runtime assembly.
- No non-greedy live route, Flash joins, quantized/windowed cache support, or
  performance claim.

## Acceptance

- Missing/incomplete injection surfaces fail before runtime creation.
- Rollback is off by default and valid only with continuous MTP.
- Qwen3.5 and Qwen4 capability differences remain explicit.
- Ragged rollback restores target, recurrent, and MTP state consistently.
- Unsupported cache ABI or family capability fails closed.

## Verification

- Runtime/rollback behavior is covered by model-free fake-forward/cache tests.
- The current 261-test model-free battery passes; all 34 changed Python files
  pass Ruff lint and format.
- A later real-target smoke used a deliberately random MTP head with zero
  acceptance; it does not qualify this PR's trained-head rollback path.
- Pending: a clean split commit, per-PR suite, mutation checks, `pr_validate`,
  real trained-head cache equality, and human review.

## Behaviour delta

The new assembly path exists only when continuous MTP is explicitly enabled.
Absent flags preserve the incumbent vendored MTP/plain-decode path.

## AI assistance disclosure

Claude/Codex assisted with the listed runtime, cache, family, configuration,
and test files under Pierre Lamy's direction. Model-free checks were rerun;
human line-by-line and real trained-head review remain pending.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed in the 261-test current-tip battery**.
- Lint/format: **passed on the current tip**; rerun after extraction.
- Publication split and PR validation: **pending**.
- Hardware trained-head acceptance: **pending; not claimed**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
