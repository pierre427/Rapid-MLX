# PR 8A — `feat(mtp): preserve QSA selection across draft steps`

Local commit: `f4d8aef9528287706007637b622cf6bc3b49f319`

Depends on: PR 8 / `f48fadb21237c7e94cddacd6fd3dfb20a204b83e`

Publication branch: `feat/mtp-qsa-selection-sharing`

Status: local-only; not pushed or submitted.

## Why

Qwen4's K=2 recursive MTP draft performs two QSA forwards inside one logical
proposal. Recomputing the selected sparse blocks for the second draft wastes
indexer work and can select a different block set after padding/finalization.
Source commit `393eddf` fixed this by preserving each lane's first-step QSA
selection across the second draft while clearing it at every transaction or
membership boundary. Rapid's backend already preferred specialized
prepare/finalize hooks, but its QSA cache did not implement the lifecycle.

## Scope

- Add an explicit begin/end self-MTP selection cycle to `QSAIndexCache`.
- Preserve the selected block IDs across the two recursive draft-step
  prepare/finalize pairs.
- Capture the last valid query's selection independently for every ragged row.
- Reuse that per-row selection on the second draft step.
- Clear shared selection on cycle completion, state restore, trim, filter, and
  extend boundaries.
- Add a default-off `share_qsa_indices` backend option and conservative family
  capability fields: true for Qwen4, false for Qwen3.5.
- Add mock lifecycle, AST/source, and capability regressions.
- Traverse Rapid's production `CacheList(KVCache, QSAIndexCache)` topology so
  the specialized lifecycle reaches the nested sidecar without double
  prepare/finalize.

Files:

- `vllm_mlx/models/qwen4_exp.py`
- `vllm_mlx/models/qwen4_exp_cache.py`
- `vllm_mlx/spec_decode/mtp/mlx_backend.py`
- `vllm_mlx/spec_decode/mtp/qwen4_exp_inject.py`
- `vllm_mlx/spec_decode/mtp/qwen3_5_inject.py`
- `tests/test_continuous_self_mtp_mlx_backend.py`
- `tests/test_mtp_batched_family_capability.py`

## Non-goals

- No NAX block-sparse kernel, dynamic Flash join, scheduler wiring, default
  enablement, or throughput claim.
- No claim that source QSA cache classes and Rapid's `QSAIndexCache` are
  layout-identical; the port preserves the lifecycle invariant using Rapid's
  current abstractions.
- No hardware/model acceptance in this PR draft.

## Acceptance

- QSA sharing is off unless the backend option and cache hooks are present.
- The first recursive draft captures one selection per physical batch row.
- The second recursive draft reuses the same selection.
- Ordinary prepare/finalize and every membership/state boundary clear it.
- Error cleanup disarms every cache that entered the cycle.
- Qwen3.5 never advertises QSA-sharing capability.

## Verification

- `python3 -m pytest -q tests/test_continuous_self_mtp_mlx_backend.py
  tests/test_mtp_batched_family_capability.py tests/test_continuous_mtp_telemetry.py`
  — 51 passed in 0.34 seconds.
- Changed Python files pass Rapid's existing Ruff binary and formatting check.
- Model-free NumPy/mock/AST/source evidence only. No MLX import, model, service,
  GPU workload, or throughput run.
- Pending before submission: per-PR full suite, contract mutation spot-check,
  PR-number `pr_validate`, human review, and real Qwen4 selection/cache parity.

## AI assistance disclosure

Codex assisted in porting source commit `393eddf` into the seven files listed
under Scope and in writing the focused regressions. The source-to-Rapid mapping
and model-free tests were reviewed in the local stack. Human line-by-line and
real-model review remain pending; update this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed** (51 combined QSA/backend/capability/telemetry cases).
- Lint/format: **passed** for the changed Python files.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **updated in the final documentation PR**.
- Existing API compatibility: **additive and default-off**.
- Hardware/model acceptance: **pending**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
