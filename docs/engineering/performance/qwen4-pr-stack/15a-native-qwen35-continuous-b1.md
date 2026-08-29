# PR 15A — `fix(mtp): adopt native Qwen3.5 heads for continuous B1`

Local source commits:

- `b1be6d00a5ec32510aec3e43e920b2b9303dbac3`
- `77eda8334c17493f5c5a52938aa41136cc5ab721`

Depends on: PR 15 / publication live fixed-cohort delivery split.

Publication branch: `fix/mtp-native-qwen35-continuous-b1`

Status: local draft only; no publication branch is pushed and nothing is
submitted upstream. The dense live-service benchmark is in progress. Serial
batched-B1 hardware validation is pending, so this draft makes no live-model,
quality, or performance claim.

## Why

Self-contained Qwen3.5 checkpoints can carry their trained MTP module under
`language_model.mtp.*`, including mixed quantization metadata and packed
weights. Rapid currently ignores that native loaded module, infers one uniform
quantization contract from the first projection, and attempts to rebuild and
inject an external sidecar. That rejects valid mixed-packed native heads and
prevents the dense Qwen3.8-27B artifact from starting with MTP enabled.

The continuous scheduler also requires at least two lanes. A serial full-stack
run therefore falls through to the legacy single-request MTP path and does not
exercise the same continuous engine used by multi-lane runs. Explicitly
enabled continuous integration needs a batched-B1 route so serial comparisons
measure the intended engine without changing the generic planner default.

## Scope

- Detect a compatible native Qwen3.5 MTP module already loaded by the model.
- Validate its callable surface and layer count, then adopt the module by
  identity so its native mixed quantization and packed shapes are preserved.
- Adapt the native `mtp_step` and cache-construction contract to Rapid's
  `mtp_forward` integration boundary.
- Keep the external sidecar injector as the fallback for models that do not
  carry a native compatible head.
- Allow `BatchedMTPConfig(min_batch_lanes=1)` only when explicitly requested;
  keep the generic default at two lanes.
- Configure the explicitly enabled continuous scheduler integration for a
  one-lane cohort and emit an observable admission receipt:
  `[MTP-continuous] admitted continuous cohort route=continuous_planned lanes=1 draft_depth=2`.

AI-touched files in the two local commits:

- `vllm_mlx/spec_decode/mtp/qwen3_5_inject.py`
- `vllm_mlx/spec_decode/mtp/batched.py`
- `vllm_mlx/spec_decode/mtp/continuous_routing.py`
- `vllm_mlx/scheduler.py`
- `tests/test_mtp_native_embedded_contract.py`
- `tests/test_mtp_spec_decode.py`
- `tests/test_batched_self_mtp_contract.py`
- `tests/test_continuous_mtp_routing.py`

## Non-goals

- No model load, GPU run, throughput result, quality claim, or live-service
  qualification is included in this draft.
- No default-on change for continuous self-MTP.
- No change to the generic planner's two-lane default.
- No removal of the external sidecar path.
- No acceptance of missing, partial, incompatible, or wrong-depth native MTP
  contracts; those cases remain fail-closed or use the compatible fallback.
- No Flash dynamic membership, APC restore, sampled delivery,
  quantized/windowed batched cache support, PLE change, or memory-admission
  policy change.

## Acceptance

- A complete native Qwen3.5 MTP module is adopted by identity without
  rebuilding or requantizing its weights.
- Native hidden-state normalization, `mtp_step`, and cache construction are
  presented through the existing Rapid MTP adapter contract.
- Missing, incomplete, wrong-depth, or incompatible native contracts do not
  silently enable an invalid MTP path.
- The existing external sidecar path remains available when no compatible
  native head is embedded.
- Generic `BatchedMTPConfig` construction still defaults to two lanes.
- With continuous MTP explicitly enabled, one eligible request is planned as
  `CONTINUOUS_PLANNED`, executes the continuous driver, and does not route
  through legacy or plain decode.
- The service emits the stable one-lane admission receipt shown above so a
  benchmark harness can assert the executed engine.

## Verification

- Exact model-free CPU-pinned test command:

  ```bash
  env PYTHONPATH=/Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX-worktrees/qwen35-native-embedded-mtp-fix-20260829:/Users/pierrelamy/Desktop/mlx-uag/mlx-lm-worktrees/qwen4-performance-benchmark-dependency-final-20260829 \
    /Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX/.venv/bin/python -c \
    'import mlx.core as mx; mx.set_default_device(mx.cpu); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_mtp_native_embedded_contract.py", "tests/test_batched_self_mtp_contract.py", "tests/test_continuous_mtp_routing.py", "tests/test_continuous_mtp_driver.py", "tests/test_continuous_mtp_generation_batch.py", "tests/test_continuous_mtp_runtime.py", "tests/test_continuous_mtp_telemetry.py"]))'
  ```

  Result: **115 passed in 1.28 seconds**; no model was loaded and no GPU work
  was requested.
- `ruff check` and `ruff format --check` pass on all eight AI-touched Python
  files listed in Scope.
- `git diff --check` passes at the two-commit implementation tip.
- Dense live-service benchmark: **in progress; no result claimed**.
- Serial batched-B1 exact-checkpoint hardware validation: **pending and
  blocking for a runtime claim**.
- Publication-head full suite, mutation spot-checks, `pr_validate`, and human
  line-by-line review remain pending.

## Behaviour delta

Before: a valid embedded mixed-quantized Qwen3.5 MTP head is discarded in
favor of a uniformly inferred external-sidecar reconstruction, and an
explicitly enabled continuous scheduler with one eligible request falls back
to the legacy single-request engine.

After: Rapid adopts a compatible embedded native head without altering its
packing, and explicit continuous integration may admit a one-lane cohort
through the continuous driver. Disabled/refused configurations and the generic
planner's default two-lane policy are unchanged.

## AI assistance disclosure

Codex wrote and reviewed the eight implementation and test files named in
Scope under Pierre Lamy's direction. The output was verified with the exact
CPU-pinned 115-test command, Ruff lint/format checks, and `git diff --check`.
Hardware validation, mutation spot-checks, `pr_validate`, and Pierre's final
line-by-line review are still pending and are not represented as complete.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted CPU-pinned model-free tests: **115 passed**.
- Lint/format and diff check: **passed**.
- Dense live-service benchmark: **in progress; no result claimed**.
- Serial batched-B1 hardware validation: **pending**.
- Publication-head full suite and `pr_validate`: **pending**.
- Critical-line mutation spot-checks: **pending**.
- Human line-by-line review: **pending**.
- Breaking API/default change: **none; continuous mode remains explicit and
  the generic minimum remains two lanes**.

## Author

X handle (optional, external contributor):
