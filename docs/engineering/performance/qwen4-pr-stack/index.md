# Qwen4 continuous self-MTP PR submission packet

Status: **private Forgejo integration branch pushed; publication branches are
not pushed and nothing is submitted upstream**.

Base: Rapid-MLX `origin/main` at
`746522837c2cde5deca3784786ce06d10b45e66c`.

This directory contains one proposed PR body per focused layer in the local
Qwen4 continuous self-MTP stack. PRs 1--12A map to existing focused commits.
PR 13 maps to `be0b3e8c`. PRs 14--16 are the required publication split of the
combined 2,075-line integration commit `d7cf5bbb`; corrective PR 15A maps to
`b1be6d00` and `77eda833`; observability PR 15B maps to `001b1919`; terminal
drain PR 15C maps to `91af1da5`; PR 17 keeps documentation last. No body is
submission-ready while a blocking or pending line remains.

The original PRs 1--13 staging pass was model-free. A later real
Qwen3.8-27B target smoke used an explicitly test-only randomly initialized MTP
head and observed zero acceptance. Its 18.4 to 87.3 aggregate token/s ladder
is a diagnostic batching floor, not trained-head self-MTP uplift or production
throughput, and it has no checked-in raw artifact. This assertion sweep ran
only model-free tests and static checks and did not use a model or GPU.

## Rapid contribution format

Each body follows Rapid's required ordering: Why, Scope, Non-goals,
Acceptance, Verification, optional Behaviour delta, AI assistance disclosure,
Checklist, and optional Author. Behaviour delta is included where a runtime
object, policy seam, live transition, or qualification policy has a concrete
before-to-after description.

The checklists are deliberately expressed as evidence/status lines while the
drafts remain incomplete. Rapid's validator rejects every unchecked Markdown
task item anywhere in a PR body, so the literal checkbox block must be added
only after each item has evidence. A draft must not be submitted while any
status line says `pending`.

## Stack order

| Order | PR body | Publication branch | Local commit | Depends on | Submission state |
| ---: | --- | --- | --- | --- | --- |
| 1 | [State-exact B=1 to B>1 handoff](01-state-exact-handoff.md) | `fix/mtp-state-exact-handoff` | `de2c5c9a` | Rapid main `74652283` | Local commit; review/full validation pending |
| 2 | [Batched transaction contracts](02-batched-transaction-contracts.md) | `feat/mtp-batched-transaction-contracts` | `203cb5be` | PR 1 / `de2c5c9a` | Local commit; review/full validation pending |
| 3 | [Ragged cache rollback adapter](03-ragged-cache-rollback.md) | `feat/mtp-ragged-cache-rollback` | `f061e090` | PR 2 / `203cb5be` | Local commit; review/full validation pending |
| 4 | [APC/MTP prepared-state validation](04-apc-prepared-state.md) | `feat/mtp-apc-prepared-state` | `07f8beea` | PR 3 / `f061e090` | Local commit; review/full validation pending |
| 5 | [Continuous transaction core](05-continuous-engine-core.md) | `feat/mtp-continuous-transaction-core` | `0a3be175` | PR 4 / `07f8beea` | Local commit; review/full validation pending |
| 6 | [Family capability descriptors](06-family-capabilities.md) | `feat/mtp-family-capabilities` | `57df9e28` | PR 5 / `0a3be175` | Local commit; review/full validation pending |
| 7 | [Continuous GenerationBatch wrapper](07-generation-batch.md) | `feat/mtp-continuous-generation-batch` | `e7a20e73` | PR 6 / `57df9e28` | Local commit; review/full validation pending |
| 8 | [Continuous MLX backend](08-mlx-backend.md) | `feat/mtp-continuous-mlx-backend` | `f48fadb2` | PR 7 / `e7a20e73` | Local commit; review/full validation pending |
| 8A | [QSA selection sharing](08a-qsa-selection-sharing.md) | `feat/mtp-qsa-selection-sharing` | `f4d8aef9` | PR 8 / `f48fadb2` | Source `393eddf` port; model-free checks pass, hardware parity pending |
| 9 | [Transformed-distribution verifier](09-transformed-verifier.md) | `feat/mtp-transformed-verifier` | `7d4a9370` | PR 8A / `f4d8aef9` | Local commit; review/full validation pending |
| 10 | [Exact-handoff regression alignment](10-handoff-regression-alignment.md) | `fix/mtp-handoff-regression-tests` | `6f9d4d70` | PR 9 / `7d4a9370` | Local test commit; affected/full reruns pending |
| 11 | [Routing and APC planner](11-routing-apc-planner.md) | `feat/mtp-continuous-routing-planner` | `7ed6326c` | PR 10 / `6f9d4d70` | Local commit; **planning only, no live continuous token delivery** |
| 12 | [Bounded qualification telemetry](12-telemetry-qualification.md) | `feat/mtp-qualification-telemetry` | `32904806` | PR 11 / `7ed6326c` | Local commit; model-free checks recorded, hardware qualification pending |
| 12A | [Digest-gate reframing](12a-digest-gate-reframing.md) | `fix/mtp-digest-gate-oracles` | `9393de76` | PR 12 / `32904806` | Source `0995cbc` port; B1 exact hierarchy staged, hardware evidence pending |
| 13 | [Dynamic wrapper membership](13-dynamic-membership-wrapper.md) | `feat/mtp-dynamic-membership-wrapper` | `be0b3e8c` | PR 12A / `9393de76` | Integration commit mirrored privately; publication branch/review pending |
| 14 | [Runtime assembly and GDN rollback](14-runtime-rollback.md) | `feat/mtp-continuous-runtime-rollback` | extract from `d7cf5bbb` | PR 13 / `be0b3e8c` | Publication split, trained-head gates, and review pending |
| 15 | [Live fixed-cohort delivery](15-live-fixed-cohort-delivery.md) | `feat/mtp-live-fixed-cohort-delivery` | extract from `d7cf5bbb` | PR 14 publication split | Queue-atomicity fix/test and review pending |
| 15A | [Native Qwen3.5 head and continuous B1](15a-native-qwen35-continuous-b1.md) | `fix/mtp-native-qwen35-continuous-b1` | `b1be6d00`, `77eda833` | PR 15 publication split | CPU-pinned tests pass; dense live benchmark in progress, serial B1 hardware gate pending |
| 15B | [Continuous engine metrics](15b-continuous-observability.md) | `feat/mtp-continuous-observability` | `001b1919` | PR 15A | CPU-pinned metrics/render tests pass; live metrics observation pending |
| 15C | [Target-only terminal drain](15c-target-only-terminal-drain.md) | `fix/mtp-target-only-terminal-drain` | `91af1da5` | PR 15B | CPU-pinned boundary tests pass; exact dense live rerun and serial B1 hardware gate pending |
| 16 | [Dynamic Qwen3.5 scheduler turnover](16-dynamic-scheduler-turnover.md) | `feat/mtp-dynamic-scheduler-turnover` | extract from `d7cf5bbb` | PR 15C | Memory-cost calibration and trained-head gates pending |
| 17 | [Documentation and paper](17-docs-paper.md) | `docs/continuous-self-mtp-batching` | rebase `580042d2` docs onto final splits | PR 16; exact Rapid artifacts remain open | Evidence/human review pending — do not submit |

The commits are intentionally linear. A body may be rebased onto a merged
predecessor, but its dependency must not be hidden by combining unrelated
layers. APC itself is not replaced by this stack. The state/transaction/cache,
wrapper, backend, transformed-verifier, planner, and telemetry layers remain
separately reviewable.

## Source lineage reconciliation

- `92576ce` maps to PRs 2, 3, 5, 7, and 8: transaction, ragged cache,
  coordinator, generation ownership, and K=2 MLX data plane.
- `93d7aa9` maps to PR 6 and the join/family gates: Qwen3.5/27B capability is
  distinct from Flash dynamic-join refusal.
- `393eddf` maps to PR 8A: per-row QSA selection survives only the two draft
  steps inside one proposal and is cleared at every geometry boundary.
- `0995cbc` maps to PR 12A: batched-B1 is exact, B>1 is compared with its own
  B1 within the shape band, legacy cross-engine comparison is informative,
  and cache/transaction equality stays exact.
- `1a0a2474` is later source-only admission work: a compute-saturation ceiling
  and configurable per-lane transient estimate. It is not yet ported here.
- `be0b3e8c` maps to PR 13; `d7cf5bbb` must be decomposed across PRs 14--16.
- `b1be6d00` and `77eda833` map to corrective PR 15A: use the compatible
  native embedded Qwen3.5 MTP contract and route an explicitly enabled serial
  cohort through the same continuous engine at batched-B1.
- `001b1919` maps to PR 15B: expose continuous-engine proposal, acceptance,
  commit, rollback, failure, and draft/verify timing without aliasing legacy
  single-request MTP counters.
- `91af1da5` maps to PR 15C: drain a final one-token budget through a
  width-one target-only transaction while true zero-budget entry remains
  fail-closed.

## Verification recorded in this packet

PR bodies 1–6 retain their focused historical results. On the amended current
tip, 261 model-free tests pass and all 34 Python files changed from the Rapid
base pass Ruff lint and formatting. Two vendored-handoff tests initially failed
because their test doubles widened `uids` without widening `tokens`; the
fixtures now model `GenerationBatch.extend()` and the full 261-test battery
passes. These cumulative results are not substitutes for per-PR-head receipts.

This sweep ran model-free tests that import the installed MLX Python package
but perform no model load or benchmark; it did not intentionally execute a GPU
workload, start a service, push, or submit anything.

The existing evidence does not replace Ruff/formatting, the full unit suite,
`pr_validate`, mutation spot-checks for production lines, scheduler stress,
real MLX cache/digest gates, or serialized Apple-Silicon qualification on an
exact candidate.

## Required truth boundary

Every submitted body must preserve these statements until new evidence changes
them:

- the consolidated integration tip is mirrored on private Forgejo, but no
  publication branch is pushed and nothing is submitted upstream;
- no Rapid candidate has reproduced the source prototype's throughput;
- no Rapid performance claim is made;
- PR 11 is planner-only at its own head; later commits add a default-off live
  coordinator and Qwen3.5 boundary joins/leaves;
- Rapid Flash dynamic membership remains capability-refused; source Flash
  evidence does not transfer to Rapid's distinct cache/runtime path;
- the real-target Rapid smoke used a random head with zero acceptance and
  cannot qualify trained-head acceptance, accepted-draft commits, or speedup;
- live APC restore and transformed-distribution sampling are not wired;
- live admission currently sets incremental draft-token memory cost to zero;
- queue removal around driver creation/join is not failure-atomic and is a
  blocking correction for the publication split;
- AI assistance must name the touched files and human review actually completed
  before submission.

## Scope exclusions

NAX QSA, PLE offload, fused-expert experiments, GDN normalization fusion,
quantized/windowed batched caches, Flash dynamic joins, live APC restore,
sampled live delivery, and source-prototype benchmark claims remain outside the
qualified Rapid surface. They require independent mechanisms and evidence.
