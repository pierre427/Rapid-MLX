# PR 13 — `feat(mtp): implement dynamic membership in the continuous wrapper`

Source integration commit: `be0b3e8c863c3364876c6ac738aedad856650d8f`

Depends on: PR 12A / `9393de76fd5e684c971b50bf631d2ce969b52b9c`

Publication branch: `feat/mtp-dynamic-membership-wrapper`

Status: publication branch not pushed and nothing submitted upstream; the
source integration commit is mirrored only on the private Forgejo branch.

## Why

The engine already supported a boundary-safe merge and per-lane extraction,
but the generation wrapper still rejected every join and tore down the whole
cohort when any lane terminated. Dynamic service membership requires the
wrapper to expose those existing transactions without weakening family
attestation or allowing mutation during an open proposal.

## Scope

- Add a non-raising dynamic-membership capability query.
- Attach canonical lane packages only at closed transaction boundaries.
- Deliver each joining lane's prepared first token before it participates in a
  proposal.
- Detach terminal lanes individually while survivor lanes continue.
- Preserve fixed-cohort teardown when the capability or opt-in is absent.
- Reject duplicate UIDs and Flash joins without Flash attestation.

Files:

- `vllm_mlx/spec_decode/mtp/continuous_engine.py`
- `vllm_mlx/spec_decode/mtp/continuous_batch.py`
- `vllm_mlx/spec_decode/mtp/continuous_routing.py`
- `tests/test_continuous_mtp_generation_batch.py`

## Non-goals

- No scheduler queue admission or live `BatchGenerator.next()` delivery.
- No Flash dynamic-membership attestation.
- No APC restore, sampled live route, memory-controller rewrite, default-on
  behavior, or performance claim.

## Acceptance

- Join is refused while a proposal is open or capability is absent.
- A joined lane emits its prepared first token exactly once before proposing.
- One terminal lane can detach without ending its companions.
- Fixed-cohort behavior remains unchanged when dynamic mode is disabled.
- Duplicate UIDs and unattested Flash joins fail before state mutation.

## Verification

- Five focused dynamic-wrapper regressions are included.
- The current 261-test model-free battery passes.
- All 34 Python files changed from the Rapid base pass Ruff lint and format.
- No model, service, GPU, or hardware qualification was run in this sweep.
- Pending: per-PR-head full suite, mutation checks, `pr_validate`, and human
  review.

## Behaviour delta

Before: every wrapper cohort was fixed and any terminal lane ended it. After:
attested, explicitly enabled runtimes may attach and detach lanes between
transactions; all other runtimes retain fixed-cohort behavior.

## AI assistance disclosure

Claude/Codex assisted with the four files listed above under Pierre Lamy's
direction. Model-free tests and static checks were rerun; human line-by-line
review and real-model qualification remain pending.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Targeted tests: **passed in the 261-test current-tip battery**.
- Lint/format: **passed on the current tip**; rerun on the split PR head.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **pending**.
- Hardware/model acceptance: **pending; not claimed**.
- Human line-by-line review: **pending**.

## Author

X handle (optional, external contributor):
