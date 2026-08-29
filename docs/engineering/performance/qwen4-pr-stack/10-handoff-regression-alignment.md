# PR 10 — `test(mtp): align exact batch handoff regressions`

Local commit: `6f9d4d706c61a2b79c99331c07350888661cc158`

Depends on: PR 9 / `7d4a9370215d9e35d11bf7642aaf1c7ee44df56a`

Publication branch: `fix/mtp-handoff-regression-tests`

Status: publication branch not pushed and nothing submitted upstream; this
commit is mirrored only on the private Forgejo integration branch.

## Why

PR 1 changed the correct B=1 to B>1 transition: the scheduler must first stage
the exact not-yet-emitted target token, and batch extension must retain that
token for the existing UID. Older CLI-wiring regressions still modeled direct
batch growth and therefore tested a transition the scheduler now forbids.

## Scope

- Update two existing vendored-MTP regression scenarios to call the exact
  batch-expansion preparation barrier.
- Model `BatchGenerator.extend` retaining the staged token for the incumbent
  UID while adding the waiting lane.
- Keep the assertions that the B>1 step falls through to the ordinary path and
  continues yielding tokens.

Files:

- `tests/test_mtp_cli_wiring.py`

## Non-goals

- No production-code change, new feature, scheduler policy, backend behavior,
  APC behavior, telemetry, or performance claim.

## Acceptance

- Regression fixtures use the state-exact handoff contract from PR 1.
- The incumbent UID retains the staged token through modeled batch extension.
- The ordinary B>1 path still returns output and remains usable on subsequent
  steps.

## Verification

- This commit changes tests only: 17 insertions and 8 deletions in
  `tests/test_mtp_cli_wiring.py`.
- The affected file was run once during earlier stack validation; it imported
  `mlx.core` and constructed tiny arrays, so it was not rerun after the strict
  no-GPU boundary was applied. No model or inference workload ran.
- Stack-tip lint/format passed for all changed Python files.
- Pending before submission: an authorized affected-test rerun, full unit
  suite, PR-number `pr_validate`, mutation spot-check, and human review.

## AI assistance disclosure

Claude/Codex assisted in aligning the two regression scenarios in
`tests/test_mtp_cli_wiring.py` under Pierre Lamy's direction. Human review and
independent focused/full-suite reruns remain pending; update this disclosure
before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **requires an authorized rerun because this file imports MLX**.
- Lint/format: **passed on the stack tip**; rerun on the final PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Documentation: **not required for a test-only alignment**.
- Existing API compatibility: **runtime code is unchanged**.
- Hardware/model acceptance: **not claimed by this test-only PR**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
