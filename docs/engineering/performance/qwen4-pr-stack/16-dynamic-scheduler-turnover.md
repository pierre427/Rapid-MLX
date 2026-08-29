# PR 16 — `feat(mtp): admit and retire Qwen3.5 lanes at cycle boundaries`

Source integration commit: `d7cf5bbb11656f0ed90ce9b1749454dc9eaf3657`
(publication split required).

Depends on: PR 15B / continuous-engine observability.

Publication branch: `feat/mtp-dynamic-scheduler-turnover`

Status: draft split only; no publication branch is pushed and nothing is
submitted upstream. The combined source commit is mirrored on private Forgejo.

## Why

Once fixed-cohort delivery owns the live response path, native Qwen3.5 MTP
heads enter that path correctly, and transaction failures are observable,
Qwen3.5 can reuse the wrapper's closed-boundary attach/detach transaction to
replace finished lanes without restarting survivor requests.

## Scope

- Queue eligible Qwen3.5 joiners only at closed transaction boundaries.
- Deliver a joiner's prepared initial token before proposal participation.
- Detach terminal lanes while companions continue and admit replacements.
- Preserve per-UID response, stop, and cleanup ownership through turnover.
- Keep Qwen4/Flash dynamic membership capability-refused.

Primary files to extract from `d7cf5bbb`:

- dynamic-turnover portions of `vllm_mlx/scheduler.py`
- `vllm_mlx/spec_decode/mtp/continuous_driver.py`
- `vllm_mlx/spec_decode/mtp/continuous_batch.py`
- Qwen3.5/Qwen4 family descriptors and focused tests

## Non-goals

- No Flash dynamic join.
- No production-head throughput or acceptance claim.
- No incremental draft-token memory accounting: the combined live source
  currently passes `bytes_per_draft_token=0`, so K-depth degradation is not yet
  calibrated from real incremental cost.
- No APC restore, non-greedy dynamic route, or default-on behavior.

## Acceptance

- Join/leave occurs only between transactions and only for attested Qwen3.5.
- Flash/Qwen4 join is rejected before mutation.
- Join failure preserves or restores scheduler queue ownership.
- Survivor token streams remain valid through companion leave and replacement.
- Admission includes nonzero measured draft-token cost before merge readiness.

## Verification

- Model-free wrapper/driver tests cover boundary join, per-lane leave, and
  survivor continuation on the combined tip.
- A real Qwen3.8-27B target smoke used an explicitly test-only random MTP head,
  observed zero acceptance, a solo-identical joined lane, and companion
  continuation. It has no checked-in raw artifact and does not qualify a
  trained-head join, accepted-draft commit, throughput uplift, or production
  service.
- The diagnostic lane ladder was 18.4 token/s at N=1 and 87.3 at N=16
  (4.74x), all at zero acceptance; it is not a self-MTP speedup result.
- Pending: queue fault injection, nonzero memory calibration, split-PR suite,
  `pr_validate`, trained-head matrix, and human review.

## Behaviour delta

When all dynamic controls and Qwen3.5 attestation are present, the live cohort
may replace lanes at cycle boundaries. Flash and all default configurations
remain fixed-cohort/incumbent.

## AI assistance disclosure

Claude/Codex assisted with the scheduler, driver, wrapper, family descriptors,
and tests under Pierre Lamy's direction. Model-free checks were rerun; the
hardware smoke is diagnostic only. Human review and trained-head qualification
remain pending.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Model-free join/leave tests: **passed on the combined tip**.
- Queue atomicity and nonzero memory calibration: **pending and blocking**.
- Publication split, PR validation, and mutation checks: **pending**.
- Hardware trained-head acceptance: **pending; not claimed**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
