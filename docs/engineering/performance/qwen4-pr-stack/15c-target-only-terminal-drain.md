# PR 15C — `fix(mtp): drain target-only terminal cycles`

Local source commit: `91af1da5643de4dc3a1d6e61afe8d045f01e1709`.

Depends on: PR 15B / continuous-engine observability.

Publication branch: `fix/mtp-target-only-terminal-drain`

Status: local draft only; no publication branch is pushed and nothing is
submitted upstream. The pre-fix failure was observed in a live dense-27B
warmup, but this correction has only model-free CPU verification. The exact
live rerun and serial batched-B1 hardware validation are pending and blocking.

## Why

A request with one output slot left cannot form a K=2 draft proposal, but it is
not yet terminal: the target still owes one final token and the committed
hidden/token pair must reach the draft cache before detach. Treating this state
as either a normal proposal or an exhausted lane aborts the continuous engine
at short `max_tokens` boundaries.

## Scope

- Detect a cohort whose lanes have exactly one output slot remaining.
- Execute one batched width-one target forward without an MTP draft forward.
- Commit the target-only output through the normal transaction and response
  path, then flush the owed hidden/token pair during normal detach.
- Reconcile the target-only transaction in continuous-engine telemetry with
  zero proposed and accepted draft tokens.
- Preserve fail-closed refusal for a lane with no output budget remaining.

AI-assisted files:

- `vllm_mlx/spec_decode/mtp/mlx_backend.py`
- `vllm_mlx/spec_decode/mtp/continuous_telemetry.py`
- `tests/test_continuous_self_mtp_mlx_backend.py`
- `tests/test_continuous_mtp_generation_batch.py`
- `tests/test_continuous_mtp_driver.py`

## Non-goals

- No change to K=2 proposal, verification, or acceptance semantics when at
  least one complete draft-plus-target cycle fits.
- No new GPU synchronization or timing barrier.
- No model, throughput, acceptance, or quality claim.
- No dynamic-membership policy, APC restore, quantized/windowed cache, or Flash
  capability change.

## Acceptance

- `max_tokens=1` closes after the prepared initial token without a proposal.
- `max_tokens=2` emits the prepared token plus one target-only final token and
  closes normally for batched-B1.
- An all-boundary B>1 cohort uses one batched target forward and preserves
  per-lane response and detach ownership.
- The target-only cycle performs no draft forward; detach alone flushes each
  lane's committed hidden/token pair to its draft cache.
- Continuous telemetry records one proposed and committed transaction, zero
  draft proposals/accepts, one committed output per lane, zero draft time, and
  target-verify time without aliasing legacy single-request counters.
- Direct entry with zero remaining output budget is refused before mutation.

## Verification

CPU-pinned, model-free focused suite:

```bash
env PYTHONPATH=/Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX-worktrees/qwen35-native-embedded-mtp-fix-20260829:/Users/pierrelamy/Desktop/mlx-uag/mlx-lm-worktrees/qwen4-performance-benchmark-dependency-final-20260829 \
  /Users/pierrelamy/Desktop/mlx-uag/Rapid-MLX/.venv/bin/python -c \
  'import mlx.core as mx; mx.set_default_device(mx.cpu); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_continuous_mtp_generation_batch.py", "tests/test_continuous_mtp_driver.py", "tests/test_continuous_self_mtp_mlx_backend.py", "tests/test_continuous_mtp_telemetry.py", "tests/test_continuous_self_mtp_engine.py", "tests/test_continuous_mtp_runtime.py", "tests/test_metrics_route.py"]))'
```

Result: **126 passed in 1.36s**. Ruff lint and formatting pass for all five
changed files, and `git diff --check` passes.

Pending and blocking: rerun the exact dense-27B `max_tokens=2` service warmup
and serial batched-B1 benchmark. The rerun must retain the one-lane continuous
admission log, finish the request normally, omit the prior all-terminal
exception and recovery log, and expose the target-only metric deltas described
above. No live post-fix claim is made here.

## Behaviour delta

Before, a continuous cohort at the final one-token budget attempted another
K=2 proposal and aborted with `all lanes are terminal; no K=2 proposal can be
formed`. After, the same cohort commits one batched target-only transaction and
drains through normal response/detach ownership. A truly exhausted lane still
fails closed.

## AI assistance disclosure

Claude/Codex assisted with the backend boundary handling, telemetry contract,
tests, and this draft under Pierre Lamy's direction. The named model-free tests
and static checks were rerun on CPU. Human line-by-line review, the exact live
rerun, and hardware qualification remain pending.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- CPU-pinned model-free focused suite: **passed, 126 tests**.
- Ruff lint, Ruff formatting, and whitespace validation: **passed**.
- Exact dense-27B boundary rerun: **pending and blocking**.
- Serial batched-B1 hardware validation: **pending and blocking**.
- Publication-branch split, `pr_validate`, mutation checks, and human
  line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
