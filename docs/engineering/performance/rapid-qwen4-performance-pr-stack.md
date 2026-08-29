# Rapid-MLX Qwen performance PR stack

Status: locally staged integration plan on `codex/qwen4-performance-stack` at
Rapid-MLX base `746522837c2cde5deca3784786ce06d10b45e66c`. Nothing in this
stack has been pushed or submitted. No model or Metal/GPU validation was run
during staging.

## What current Rapid already contains

The stack does not re-port the Qwen4 architecture or optimizations already on
`origin/main`:

- vectorized QSA selection-mask construction (`#2533`);
- native one-layer Flash-Next MTP (`#2572`);
- batched QSA compressed-key prefill (`#2574`);
- MTP-state preservation across prefix-cache hits (`#2588`);
- materialization before batched QSA index-key commit (`#2596`);
- atomic Qwen recurrent-state rollback, adaptive prefix caching, radix
  indexing, continuous batching, fused sampling, and seeded sampling.

The abandoned local `codex/qwen4-exp-port` branch predates these integrations
and is archaeology only. Rebase mechanisms onto current abstractions; do not
cherry-pick that branch wholesale.

## Source lineage

The stack explicitly captures all four final source commits:

- `92576ce`: batched self-MTP engine and Flash-Next lifecycle;
- `93d7aa9`: Qwen3.5/27B qualification and join-gate tests;
- `393eddf`: cycle-local QSA selection preservation across recursive drafts;
- `0995cbc`: batched-B1 exact digest oracle plus bounded B>1 shape comparison.

The first two map across the transaction, wrapper, backend, capability, and
routing PRs. The latter two remain separately reviewable as PR 8A and PR 12A.

## Dependency sequence

The staged submission plan contains fourteen code/test PRs plus the final
documentation PR. Each publication branch follows Rapid's required prefix
convention; local `codex/` branches remain private staging refs.

| Order | Local commit / branch | Proposed PR | Default / hardware boundary |
| ---: | --- | --- | --- |
| 1 | `de2c5c9a` / `codex/rapid-mtp-01-state-exact-handoff` | `fix(mtp): make batch handoff state-exact` | Correctness fix; later live B=1 to B>1 gate |
| 2 | `203cb5be` / `codex/rapid-mtp-02-transaction-contracts` | `feat(mtp): add batched transaction contracts` | Pure, fail-closed lifecycle contracts |
| 3 | `f061e090` / `codex/rapid-mtp-03-ragged-cache-rollback` | `feat(mtp): add ragged cache rollback adapter` | mlx-lm 0.31.x only; quantized/windowed refused |
| 4 | `07f8beea` / `codex/rapid-mtp-04-apc-prepared-state` | `feat(mtp): validate APC prepared-state sidecars` | Exact target N / MTP N-1 / seed-hidden boundary |
| 5 | `0a3be175` / `codex/rapid-mtp-05-continuous-core` | `feat(mtp): add continuous transaction core` | Default-off; fixed cohort first |
| 6 | `57df9e28` / `codex/rapid-mtp-06-family-capabilities` | `feat(mtp): attest batched family capabilities` | Flash dynamic join false; 27B capability separable |
| 7 | `e7a20e73` / `codex/rapid-mtp-07-generation-batch` | `feat(mtp): add continuous generation batch` | Stop/length/abort ledger; fixed-cohort teardown |
| 8 | `f48fadb2` / `codex/rapid-mtp-08-mlx-backend` | `feat(mtp): add continuous MLX backend` | K=2; no real-MLX qualification yet |
| 8A | `f4d8aef9` / `codex/rapid-mtp-08a-qsa-selection-sharing` | `feat(mtp): preserve QSA selection across draft steps` | Source `393eddf`; default-off; real QSA parity pending |
| 9 | `7d4a9370` / `codex/rapid-mtp-09-transformed-verifier` | `feat(mtp): verify transformed distributions` | top-p/min-p/top-k; XTC fail-closed |
| 10 | `6f9d4d70` / `codex/rapid-mtp-10-handoff-regressions` | `test(mtp): align exact batch handoff regressions` | Replaces unsafe stale-token fallback expectations |
| 11 | `7ed6326c` / `codex/rapid-mtp-11-routing-apc` | `feat(mtp): plan continuous routing and APC restore` | Opt-in planner only; legacy `_step` remains live |
| 12 | `32904806` / `codex/rapid-mtp-12-qualification-telemetry` | `feat(mtp): add bounded qualification telemetry` | Synthetic evidence can never qualify performance |
| 12A | `9393de76` / `codex/rapid-mtp-12a-digest-gate-oracles` | `fix(mtp): separate digest qualification oracles` | Source `0995cbc`; B1 exact, B>1 band, cache exact |
| 13 | current documentation tip | `docs(perf): document continuous self-MTP batching` | Paper, sequence, and PR packet; no Rapid perf claim |

PR 11 deliberately stops before live continuous token delivery. Rapid's
current scheduler and its external mlx-lm `GenerationBatch` do not expose the
source prototype's prepared cohort at a safe ownership boundary. The opt-in
therefore reaches a non-mutating admission/APC planner, while the incumbent
single-lane vendored MTP `_step` remains authoritative. A later live-coordinator
PR must bind PRs 7--9 and 8A to scheduler request delivery and pass the hardware matrix
before the feature can be described as enabled. Model-specific QSA kernels
remain separate side work and are not prerequisites.

## Side work: QSA kernels

The Rapid worktree historically named
`Rapid-MLX-worktrees/qwen4-nax-side-stack-20260829` is not the lab's MPP/NAX
kernel. It rebases Rapid PR #2547's opt-in Metal block-sparse prefill kernel on
the same Rapid base: `5e265f18` (kernel), `bbfb327b` (production gates), and
`726e1ced` (unsupported-layout fallback). Its upstream M3 Ultra evidence reports
a 10.7% prefill gain at 32K with unchanged decode; this local rebase did not
rerun that measurement. Keep it behind `RAPID_MLX_QSA_BLOCK_SPARSE`.

The actual lab MPP/NAX kernel is separate in `mlx-lm-unified`; committed JSON
records median prefill gains of 16.0%, 31.6%, and 37.9% at 16K, 32K, and 64K
with flat decode. A later default-on restart served one request and then exited
without a fresh traceback. That unrooted incident keeps NAX default-off pending
isolated, memory-observed service qualification. Do not mix either kernel into
scheduler/MTP reviews.

Likewise, do not duplicate Rapid's fused gate/up path. A sorted-gather threshold
change from 64 to the lab's measured 20 is a later micro-PR only after a
Rapid-target paired benchmark.

## Excluded work

- Rapid-native incremental joins: the initial stack is fixed-cohort-only;
  scheduler turnover and Rapid-native real-BF16 qualification are follow-ups.
- NVMe PLE offload (swap abort gate failed).
- `moe_shared_in_gather` (logprob and digest divergence).
- GDN q/k normalization fusion (decode regression).
- the untracked fused-expert kernel (default-off candidate without real-model
  validation).
- the old thread-crossing RNG wiring; per-lane keys must remain owned by the
  generation thread.
- quantized/windowed batched MTP caches until their ragged rollback contracts
  have independent tests.

## Required verification before submission

The no-GPU staging gate can cover Ruff, formatting, AST/import checks, mocked
state-machine tests, fail-closed capability tests, and the repository's
`no-mlx` suite. It cannot qualify inference behavior.

Before any inference PR is called merge-ready, run on the exact candidate
commit and immutable checkpoints:

1. Existing full unit suite and changed-line coverage.
2. Mutation spot-checks proving the new production-line assertions are covered.
3. Single-lane incumbent parity for greedy and supported sampled transforms.
4. Fixed-membership `N=1,2,4` cache equality, exact batched-B1 digest,
   B>1-vs-own-B1 shape-band classification, informative legacy comparison,
   acceptance, memory, aggregate throughput, and per-lane latency.
5. EOS, max-token, abort, disconnect, UID reuse, and B=1↔B>1 handoff tests.
6. APC miss, trivial hit, exact-boundary hit, radix reuse, eviction, and
   persistence/restart tests.
7. Dynamic leave/join tests after Rapid scheduler turnover is wired. The source
   Flash/27B join gates pass, but that does not qualify the unwired Rapid path.
8. Interleaved cold/hot controls to bound Apple-Silicon thermal drift.
9. Raw JSON/JSONL artifacts checked into the benchmark ledger with commands,
   environment, checkpoint revisions, and candidate SHA.

## PR-description contract

Each PR should use Rapid's current six-field template: Why, Scope, Non-goals,
Acceptance, Verification, and Behaviour delta when applicable. Include a
truthful AI-assistance disclosure naming the AI-touched files and the human
review/verification performed. Do not claim the source prototype's 120.3 or
82.0 token/s as Rapid results until the candidate Rapid commit reproduces them.
