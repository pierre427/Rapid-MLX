# PR 17 — `docs(perf): document continuous self-MTP batching`

Local commit: `codex/rapid-mtp-13-docs-paper` branch tip; its SHA changes when
this self-describing packet is amended.

Publication branch: `docs/continuous-self-mtp-batching`

Depends on: PR 16 / publication dynamic-turnover split and future exact Rapid
artifacts.

Status: documentation source is mirrored on the private Forgejo integration
branch; the final publication branch is not pushed or submitted. Qualification
evidence and human review are pending, so do not submit yet.

## Why

The architecture, operating controls, supported boundaries, and evaluation
need a durable explanation tied to an exact Rapid candidate. Source-prototype
session rates are useful motivation but cannot be presented as Rapid results.

## Scope

- Document the transaction, wrapper, backend, residual verifier, planner, APC
  composition, telemetry, refusal, and rollback boundaries.
- State explicitly that PR 11 is planning-only at its own head and that PRs
  13--16 later add wrapper membership, runtime assembly, live delivery, and
  Qwen3.5 turnover behind default-off controls.
- Publish an engineering report and paper source whose Rapid tables point to
  raw artifacts and an exact candidate SHA.
- Distinguish source-prototype raw-log/transcript evidence from Rapid
  measurements and from artifact-complete qualification.
- State that Rapid Qwen3.5 turnover is wired but incompletely qualified, while
  Rapid Flash remains capability-refused for dynamic joins.
- Include reproducible commands, environment, model revisions, prompt/seed
  manifest, thermal order, aggregate/per-lane metrics, and limitations.

Final documentation, paper, bibliography, and artifact paths must be inserted
once the commit and evidence exist.

## Non-goals

- No code/default change, paper submission, or extrapolation from source
  results.
- No claim that 120.3 or 82.0 token/s are Rapid results unless independently
  reproduced on the exact Rapid candidate with durable raw artifacts.

## Acceptance

- Every Rapid number points to raw evidence and an exact candidate SHA.
- Historical planner-only and later live-delivery states are never conflated.
- Commands/environment/model identity reproduce the reported battery.
- Aggregate/per-lane outcomes, negative controls, and unsupported boundaries
  appear together.
- Documentation renders and local links resolve.

## Verification

The amended TeX source passes a Pandoc parse, local Markdown links resolve, and
TeX environments balance. The later Rapid tip has only a
real-target/random-head, zero-acceptance smoke with no checked-in raw artifact;
it is not a production-head or performance qualification.

## AI assistance disclosure

Codex assisted in writing and reviewing
`docs/engineering/performance/continuous-self-mtp-batching.tex`,
`docs/engineering/performance/rapid-qwen4-performance-pr-stack.md`, the
Markdown packet under `docs/engineering/performance/qwen4-pr-stack/`, and the
generated figure candidates under `docs/engineering/performance/figures/`.
The figure CSVs retain per-row evidence grades; underlying lab-root logs are not
packaged in this Rapid draft. Human factual/technical review and independent
evidence reruns remain pending; update this disclosure before submission.

> By submitting this PR I confirm I can explain the intent, risk, and behavior
> of every non-generated change in this PR. For any generated / boilerplate /
> scaffolded sections, I've identified them above and can describe how I
> verified them.

## Checklist

- Documentation structure: **passed** (Pandoc parse, link resolution, balanced TeX environments).
- Lint/format: **not applicable to the documentation-only diff**.
- PR-number self-validation: **pending until the PR exists**.
- Critical-path mutation spot-check: **not applicable; runtime code is unchanged**.
- Documentation completeness: **pending exact Rapid hardware artifacts and candidate SHA**.
- Existing API compatibility: **runtime code is unchanged**.
- Hardware/model acceptance: **pending and not claimed by this documentation PR**.
- Human factual/technical review: **pending**.

## Author

X handle (optional, external contributor):
