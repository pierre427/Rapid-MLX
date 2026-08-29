# PR 15B — `feat(mtp): expose continuous engine metrics`

Local source commit: `001b1919170c56e2c8638bb9f4a9639608d49ab0`.

Depends on: PR 15A / native Qwen3.5 head and continuous-B1 correction.

Publication branch: `feat/mtp-continuous-observability`

Status: local draft only; no publication branch is pushed and nothing is
submitted upstream. No model or GPU was used for this change. The live dense
benchmark has not been rerun on this metrics commit, so no performance result
is claimed.

## Why

Rapid's legacy single-request MTP generator owns the
`rapid_mlx_spec_decode_*` counters. A request routed through the continuous
self-MTP engine bypasses that generator, so its attempts and accepts correctly
remain zero even while continuous transactions are active. That makes a live
operator unable to distinguish low MTP acceptance from a missing metric path,
or separate draft cost from target verification at long context.

The continuous engine already has bounded transaction telemetry and clean
draft/target-verify execution regions. It needs its own process-global metric
family, wired to the actual proposal and atomic commit boundaries rather than
aliasing the legacy counters.

## Scope

- Wire the existing bounded continuous telemetry registry into production
  runtime assembly.
- Record validated proposal transactions before commit so a failed commit does
  not erase completed draft/verify work.
- Record target-accepted draft tokens, atomically committed output tokens,
  transaction outcomes, successful verify/delivery cache rollbacks, and
  proposal/commit failure phases.
- Measure the existing required draft and target-verify regions with
  `time.perf_counter()` without adding `mx.eval`, a device synchronization, or
  an array read solely for measurement.
- Export a separate `rapid_mlx_continuous_mtp_*` Prometheus family before the
  engine-ready gate, including zero-valued cold-start series.
- Keep metric label cardinality fixed through enums; request IDs, model IDs,
  exception text, and other unbounded labels are excluded.

Exported metric names:

- `rapid_mlx_continuous_mtp_proposals_total`
- `rapid_mlx_continuous_mtp_draft_tokens_proposed_total`
- `rapid_mlx_continuous_mtp_draft_tokens_accepted_total`
- `rapid_mlx_continuous_mtp_accept_ratio`
- `rapid_mlx_continuous_mtp_output_tokens_committed_total`
- `rapid_mlx_continuous_mtp_transactions_total{outcome=...}`
- `rapid_mlx_continuous_mtp_rollbacks_total{phase=...}`
- `rapid_mlx_continuous_mtp_failures_total{phase=...}`
- `rapid_mlx_continuous_mtp_draft_seconds_total`
- `rapid_mlx_continuous_mtp_target_verify_seconds_total`

AI-touched files:

- `vllm_mlx/spec_decode/mtp/continuous_telemetry.py`
- `vllm_mlx/spec_decode/mtp/continuous_engine.py`
- `vllm_mlx/spec_decode/mtp/mlx_backend.py`
- `vllm_mlx/spec_decode/mtp/continuous_runtime.py`
- `vllm_mlx/spec_decode/mtp/continuous_batch.py`
- `vllm_mlx/routes/metrics.py`
- `tests/test_continuous_mtp_telemetry.py`
- `tests/test_continuous_mtp_generation_batch.py`
- `tests/test_continuous_self_mtp_mlx_backend.py`
- `tests/test_metrics_route.py`

## Non-goals

- No reuse or mutation of legacy `rapid_mlx_spec_decode_*` counters.
- No model load, live-service restart, GPU benchmark, throughput claim, or
  acceptance-rate claim.
- No device-event profiler or claim that the wall-clock split is isolated
  kernel time. Each timer includes the required cache preparation/finalization
  and host-side verification work in its existing execution region.
- No new `mx.eval`, `.item()`, cache materialization, or synchronization solely
  for telemetry.
- No per-request/per-lane labels, histograms, traces, or raw exception labels.
- No change to admission, MTP output, cache rollback, sampling, or scheduling
  policy.

## Acceptance

- A committed continuous cycle reconciles accepted drafts at or below proposed
  drafts and records committed output independently; terminal clipping may make
  accepted drafts exceed delivered output tokens without corrupting the ratio.
- A commit failure retains its validated proposal/draft work, publishes no
  committed output, and increments the bounded commit-failure counter.
- Verify and terminal-delivery rollback operations remain distinguishable.
- `/metrics` renders every continuous series at cold start and after synthetic
  counter updates, and its output parses as Prometheus text exposition.
- Continuous metrics move without changing the legacy single-request counter.
- Backend timings are positive in the model-free NumPy seam test and require no
  MLX synchronization added for measurement.

## Verification

- Exact model-free CPU-pinned test command:

  ```bash
  env PYTHONPATH=/Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX-worktrees/qwen35-native-embedded-mtp-fix-20260829:/Users/pierrelamy/Desktop/mlx-uag/mlx-lm-worktrees/qwen4-performance-benchmark-dependency-final-20260829 \
    /Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX/.venv/bin/python -c \
    'import mlx.core as mx; mx.set_default_device(mx.cpu); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_continuous_mtp_telemetry.py", "tests/test_continuous_mtp_generation_batch.py", "tests/test_continuous_self_mtp_engine.py", "tests/test_continuous_self_mtp_mlx_backend.py", "tests/test_continuous_mtp_runtime.py", "tests/test_metrics_route.py"]))'
  ```

  Result: **112 passed in 1.42 seconds**; no model was loaded and no GPU work
  was requested.
- `ruff check` and `ruff format --check` pass on all ten AI-touched Python
  files listed in Scope.
- `git diff --check` passes.
- Live `/metrics` service observation and exact-candidate benchmark rerun:
  **pending; no result claimed**.
- Publication-head full suite, mutation spot-checks, `pr_validate`, and human
  line-by-line review remain pending.

## Behaviour delta

Before: continuous self-MTP transactions leave the legacy attempt/accept
counters at zero and publish no replacement operational signal.

After: the legacy counters remain exclusive to the single-request generator,
while continuous multi-lane and batched-B1 work updates its own clearly named
counter/timing family. Model output and routing policy are unchanged.

## AI assistance disclosure

Codex wrote and reviewed the ten implementation and test files named in Scope
under Pierre Lamy's direction. The output was verified with the exact
CPU-pinned 112-test command, Prometheus parser coverage, Ruff lint/format, and
`git diff --check`. Live service verification, mutation spot-checks,
`pr_validate`, and Pierre's final line-by-line review remain pending.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted CPU-pinned model-free tests: **112 passed**.
- Prometheus rendering/parser coverage: **passed**.
- Lint/format and diff check: **passed**.
- Live metrics and benchmark observation: **pending; no result claimed**.
- Publication-head full suite and `pr_validate`: **pending**.
- Critical-line mutation spot-checks: **pending**.
- Human line-by-line review: **pending**.
- Breaking API/default change: **none; only new metric series are added**.

## Author

X handle (optional, external contributor):
