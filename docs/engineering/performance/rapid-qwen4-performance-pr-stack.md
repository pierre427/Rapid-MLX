# Rapid-MLX Qwen performance PR stack

Status: locally staged integration plan on `codex/qwen4-performance-stack` at
Rapid-MLX base `746522837c2cde5deca3784786ce06d10b45e66c`. The consolidated
tip `d7cf5bbb` is mirrored to private Forgejo; no publication branch is pushed
and nothing is submitted upstream. This assertion sweep used model-free tests
and static checks only. A separate real-target/random-head smoke is diagnostic,
not a production-head or performance qualification.

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
Source commit `1a0a2474` later added a compute-saturation ceiling and
configurable per-lane transient estimate; that admission change is not yet
ported into this Rapid stack. Rapid integration commits `be0b3e8c` and
`d7cf5bbb` map to PRs 13--16 below.

## Dependency sequence

The staged submission plan contains eighteen code/test PR entries (counting 8A
and 12A) plus the final documentation PR. Each publication branch follows
Rapid's required prefix convention; private `codex/` refs are staging only.

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
| 13 | `be0b3e8c` | `feat(mtp): implement dynamic membership in the continuous wrapper` | Wrapper boundary joins/leaves; still no scheduler delivery |
| 14 | split from `d7cf5bbb` | `feat(mtp): assemble the continuous runtime and exact GDN rollback` | Default-off; trained-head rollback unqualified |
| 15 | split from `d7cf5bbb` | `feat(mtp): deliver continuous fixed-cohort responses live` | Queue-atomicity fix/test blocks publication |
| 16 | split from `d7cf5bbb` | `feat(mtp): admit and retire Qwen3.5 lanes at cycle boundaries` | Qwen3.5 only; memory cost and trained-head gates pending |
| 17 | rebase `580042d2` docs last | `docs(perf): document continuous self-MTP batching` | Paper, figures, sequence, and PR packet; no Rapid perf claim |

PR 11 deliberately stops before live continuous token delivery at its own
head. PR 13 then exposes wrapper membership, and the combined integration
commit `d7cf5bbb` adds default-off runtime assembly, scheduler delivery, and
Qwen3.5 turnover. Publication must split that broad commit across PRs 14--16.
The tip is greedy-only, does not consume APC prepared state, passes zero for
incremental draft-token memory cost, and removes requests before driver
creation/join is known to succeed. Those boundaries and the queue-atomicity fix
must remain explicit. Qwen4/Flash dynamic membership is capability-refused.
Model-specific QSA kernels remain separate side work and are not prerequisites.

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
with flat decode. A session note records that a later default-on production
restart returned one request and the service then disappeared without a new
traceback. No process-level artifact captured that attempt, so it does not
establish a kernel defect. NAX remains default-off pending an isolated,
memory-observed rerun. Do not mix either kernel into scheduler/MTP reviews.

Likewise, do not duplicate Rapid's fused gate/up path. A sorted-gather threshold
change from 64 to the lab's measured 20 is a later micro-PR only after a
Rapid-target paired benchmark.

## Excluded work

- Flash dynamic joins: source Flash evidence does not attest Rapid's separate
  runtime, and the Rapid capability remains false pending real-BF16 gates.
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

Current receipt: 261 model-free tests pass on Python 3.12.13, and all 34 Python
files changed from base pass Ruff lint and formatting. The tests are primarily
pure-Python, mocked cache/runtime, and AST wiring coverage; they are not a live
service integration result. No model or GPU workload was used for this sweep.

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
7. Dynamic leave/join tests on the publication scheduler splits. Qwen3.5 is
   wired with only a random-head, zero-acceptance smoke; Flash remains refused.
8. Interleaved cold/hot controls to bound Apple-Silicon thermal drift.
9. Raw JSON/JSONL artifacts checked into the benchmark ledger with commands,
   environment, checkpoint revisions, and candidate SHA.

## PR-description contract

Each PR should use Rapid's current six-field template: Why, Scope, Non-goals,
Acceptance, Verification, and Behaviour delta when applicable. Include a
truthful AI-assistance disclosure naming the AI-touched files and the human
review/verification performed. Do not claim the source prototype's 120.3 or
82.0 token/s as Rapid results until the exact Rapid candidate reproduces them.
Likewise, the Rapid random-head 18.4-to-87.3 ladder is a zero-acceptance
batching diagnostic, not self-MTP uplift.
