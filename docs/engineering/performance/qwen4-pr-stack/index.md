# Qwen4 continuous self-MTP PR submission packet

Status: **local drafts only — nothing pushed or submitted**.

Base: Rapid-MLX `origin/main` at
`746522837c2cde5deca3784786ce06d10b45e66c`.

This directory contains one proposed PR body per focused layer in the local
Qwen4 continuous self-MTP stack. The original thirteen exist as local
commits/branch tips; two final source-lineage corrections are staged for
insertion as PR 8A and PR 12A. PR 13 must not be submitted as a performance
claim until exact Rapid qualification artifacts exist.

No model, service, or inference benchmark was run while preparing this packet.
One broader legacy CLI test accidentally constructed tiny `mlx.core` arrays;
Metal initialization therefore cannot be ruled out, although no model was
loaded. All subsequent checks were pure-Python, NumPy/mock, AST, lint, or
compile checks. The stack has **no Rapid-MLX performance qualification**.
Source-prototype throughput is motivation only and must not be presented as a
Rapid result.

## Rapid contribution format

Each body follows Rapid's required ordering: Why, Scope, Non-goals,
Acceptance, Verification, optional Behaviour delta, AI assistance disclosure,
Checklist, and optional Author. Behaviour delta is present only for PRs 1, 6,
11, and 12A, where a runtime object, policy seam, live transition, or
qualification policy has a concrete before-to-after description.

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
| 13 | [Documentation and paper](13-docs-paper.md) | `docs/continuous-self-mtp-batching` | `codex/rapid-mtp-13-docs-paper` tip | PR 12A / `9393de76`; exact Rapid artifacts remain open | Local commit; evidence/human review pending — do not submit |

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

## Verification recorded in this packet

PR bodies 1–6 retain their focused model-free results. The latest cumulative
stack-tip battery passed 163 model-free tests, and all 28 changed Python files
passed Ruff lint and formatting checks. PR 12 separately records 28 focused
telemetry tests and 40 combined core-plus-telemetry tests, plus compile and
import-isolation checks. These cumulative results are not substitutes for
per-PR-head receipts.

This formatting update performs documentation-only static checks. It does not
run tests, import MLX, execute a model, use the GPU, start a service, benchmark
a candidate, push, or submit anything.

The existing evidence does not replace Ruff/formatting, the full unit suite,
`pr_validate`, mutation spot-checks for production lines, scheduler stress,
real MLX cache/digest gates, or serialized Apple-Silicon qualification on an
exact candidate.

## Required truth boundary

Every submitted body must preserve these statements until new evidence changes
them:

- no commit in this packet has been pushed or submitted;
- no Rapid candidate has reproduced the source prototype's throughput;
- no Rapid performance claim is made;
- PR 11 installs a metadata planner only; it does not route live token delivery
  through `ContinuousMTPGenerationBatch`;
- Rapid Flash/27B dynamic membership is not wired or hardware-qualified; the
  source prototype's join passes do not transfer automatically;
- fixed-membership batching remains non-operational in live serving until a
  later coordinator and candidate-specific cache, digest, memory, throughput,
  abort, and EOS gates land;
- AI assistance must name the touched files and human review actually completed
  before submission.

## Scope exclusions

NAX QSA, PLE offload, fused-expert experiments, GDN normalization fusion,
quantized/windowed batched caches, live continuous scheduler token delivery,
and source-prototype benchmark prose are not implemented by PRs 1–12. They
require independent mechanisms and qualification.
