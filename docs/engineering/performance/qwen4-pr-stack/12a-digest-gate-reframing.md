# PR 12A — `fix(mtp): separate digest qualification oracles`

Local commit: `9393de76fd5e684c971b50bf631d2ce969b52b9c`

Depends on: PR 12 / `3290480649731ea9ebaa3237eb7212ea256609da`

Publication branch: `fix/mtp-digest-gate-oracles`

Status: publication branch not pushed and nothing submitted upstream; this
commit is mirrored only on the private Forgejo integration branch.

## Why

One digest classification cannot distinguish engine correctness from BF16
batch-shape numerics. Source commit `0995cbc` showed that the blocking oracle
must be the batched engine at B=1, where no batch-shape confound exists. B>1
lanes are then compared with their own batched-B1 executions inside an explicit
shape-noise band. The legacy single-lane-vs-B>1 comparison is informative,
while cache and transaction equality remain independently exact.

## Scope

- Replace the ambiguous per-lane digest field with three externally produced
  classification fields:
  batched-B1 exactness, B>1-vs-own-B1 batch-shape classification, and legacy
  single-lane compatibility.
- Require every batched-B1 result to be exact.
- Forbid a distributional excuse at cohort size one.
- Permit exact or distributional B>1-vs-own-B1 results; reject divergent or
  unrun blocking comparisons.
- Keep legacy single-lane comparison non-blocking.
- Prove that a distributional token band cannot override cache equality.
- Leave numeric distance calculation, comparison-identity binding, and raw
  artifact persistence to the external qualification runner.

Files:

- `vllm_mlx/spec_decode/mtp/continuous_telemetry.py`
- `tests/test_continuous_mtp_telemetry.py`

## Non-goals

- No implementation of the hardware logit-distance diagnostic itself.
- No change to generation, sampling, cache mutation, scheduler routing, or
  performance thresholds.
- No claim that the Rapid candidate has passed the hardware digest battery.

## Acceptance

- Batched-B1 distributional, divergent, or unrun results fail qualification.
- A B=1 cohort cannot claim a batch-shape tolerance.
- The schema accepts a B>1 distributional classification only under the policy
  that the external runner compared that lane with its matching batched-B1
  reference; this offline enum schema does not compute or bind that match.
- Legacy single-lane divergence is recorded without blocking qualification.
- Cache inequality always fails, regardless of token-distribution band.

## Verification

- Included in the 51-test focused pure-Python battery; all tests passed.
- Changed Python files pass Rapid's existing Ruff binary and formatting check.
- No MLX import, model, service, GPU workload, or hardware digest run.
- Pending before submission: per-PR full suite, mutation spot-check,
  PR-number `pr_validate`, human review, and exact-candidate hardware artifacts.

## Behaviour delta

Before: an undifferentiated `DISTRIBUTIONAL` digest could satisfy qualification
even at B=1, obscuring whether the engine itself diverged.

After: the qualification policy requires batched-B1 to be an exact blocking
oracle; only an externally produced B>1-vs-own-B1 classification may use the
documented batch-shape band, and cache equality remains independently exact.
The schema does not itself calculate the band or prove run identity.

## AI assistance disclosure

Codex assisted in porting source commit `0995cbc` into
`vllm_mlx/spec_decode/mtp/continuous_telemetry.py` and
`tests/test_continuous_mtp_telemetry.py`, and in writing the qualification
regressions. Human mathematical and real-hardware review remain pending;
update this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed within the 51-test focused battery**.
- Lint/format: **passed** for the changed Python files.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **updated in the final documentation PR**.
- Existing API compatibility: **schema is pre-production and not live-wired**.
- Hardware/model acceptance: **pending**.
- Human mathematical review: **pending**.

## Author

X handle (optional, external contributor):
