# PR 15 — `feat(mtp): deliver continuous fixed-cohort responses live`

Source integration commit: `d7cf5bbb11656f0ed90ce9b1749454dc9eaf3657`
(publication split required).

Depends on: PR 14 / publication runtime split.

Publication branch: `feat/mtp-live-fixed-cohort-delivery`

Status: draft split only; no publication branch is pushed and nothing is
submitted upstream. The combined source commit is mirrored on private Forgejo.

## Why

PR 11 deliberately stopped at metadata planning. A scheduler-facing driver is
needed to drain multi-token, multi-lane bursts into Rapid's one-response-per-UID
contract while preserving legacy fallback and cohort cleanup.

## Scope

- Add a response driver for fixed cohorts.
- Bind the runtime assembly to `BatchGenerator.next()` behind the existing
  default-off continuous-MTP option.
- Deliver at most one response per live UID per scheduler iteration.
- Preserve stop/length/abort handling and detach-package cleanup.
- Keep the incumbent vendored path as fallback when admission refuses.

Primary files to extract from `d7cf5bbb`:

- `vllm_mlx/spec_decode/mtp/continuous_driver.py`
- fixed-cohort portions of `vllm_mlx/scheduler.py`
- configuration/CLI wiring
- `tests/test_continuous_mtp_driver.py`
- fixed-cohort live routing tests

## Non-goals

- No boundary joins or replacement admission; those belong to PR 16.
- No live APC restore: prepared state is attached as metadata but not consumed.
- No sampled live route: current capabilities make live continuous delivery
  greedy-only.
- No claim that queue mutation is failure-atomic. The current combined source
  removes admitted requests before driver creation; this split must construct
  first or restore on exception, with a fault-injection regression.
- No performance claim or default-on change.

## Acceptance

- The option is default-off and refusal preserves the incumbent route.
- Driver creation failure leaves every request recoverably queued.
- Each scheduler iteration returns no more than one response per UID.
- EOS, max-token, abort, and disconnect cleanly release lane state.
- Fixed-cohort execution does not admit a replacement lane.

## Verification

- The current combined tip passes the 261-test model-free battery and Ruff on
  all 34 changed Python files.
- Existing driver/runtime tests use fakes and routing tests include AST checks;
  they are not a production-service qualification.
- Failure-atomic queue fault injection is **pending and blocking**.
- Per-PR suite, mutation checks, `pr_validate`, real trained-head delivery,
  and human review remain pending.

## Behaviour delta

With continuous MTP disabled or refused, behavior is unchanged. With it
enabled on a supported greedy fixed cohort, the scheduler may use the new
driver as its authoritative response source.

## AI assistance disclosure

Claude/Codex assisted with the driver, scheduler, configuration, and test
surfaces under Pierre Lamy's direction. Model-free checks were rerun. The
blocking queue-atomicity correction and human review remain pending.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted model-free tests: **passed on the combined tip**.
- Failure-atomic queue regression: **pending and blocking**.
- Lint/format: **passed on the combined tip**; rerun after extraction.
- Publication split and PR validation: **pending**.
- Hardware trained-head acceptance: **pending; not claimed**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
