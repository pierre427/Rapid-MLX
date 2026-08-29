# Continuous self-MTP product-stack benchmark protocol

Status: **prepared, not executed**. This document and the companion artifacts
were validated without importing MLX, starting a server, loading a model, or
submitting inference work.

## Question and comparison arms

The primary product comparison asks what an operator gets from Rapid current
versus the complete staged stack, not which individual commit caused each
change:

1. `rapid_current`: Rapid `746522837c2cde5deca3784786ce06d10b45e66c`, its
   current MLX 0.32.0 environment, speculative decoding explicitly disabled,
   and NAX explicitly disabled.
2. `full_stack`: Rapid `b47c32981cf1cc2e909cb37c001acb90078ec053`, the
   clean merged mlx-lm dependency at
   `5f581a4c07091dac9734f3b5c83522612325f641`, patched MLX
   `0.32.2.dev20260829+334084ce9`, continuous K=2 self-MTP, and NAX on the
   eligible Flash path.

That product comparison deliberately includes the MLX-core delta. The
`attribution` profile adds `rapid_current_common_runtime` and `candidate_plain`
controls so dormant source drift and runtime/core effects can be separated from
the enabled MTP/NAX path. All arms use the exact same local model artifact
within a model pair. These are therefore engine-stack A/Bs, not measurements
of Rapid's different published model aliases.

The combined Rapid candidate contains the full staged Rapid performance series
through `9ba3b0d8` plus the three NAX commits replayed as `1ecabe4f`,
`648f72c3`, and `b47c3298`. Its manifest records required ancestors and refuses
to validate if a checkout moves, becomes dirty, or loses one of those commits.

## Context ladder

Both Qwen3.8 27B and Qwen3.8 Flash-Next use tokenizer-counted cold prompts at
1K, 4K, 16K, 32K, and 64K, with 256 requested output tokens and three
repetitions per cell. Each concurrent cohort is released behind a barrier and
each lane has a unique nonce. Prefix caching is disabled at the server and
request; nonzero cached-token accounting is a failed cell.

Every baseline/candidate cell has the same prompt, decode budget, KV dtype,
prefill step, and concurrency. We report per-lane TTFT and usage alongside
aggregate completion throughput; a single baseline stream is never compared
with a multi-lane candidate. The service operating points are predeclared from
the completed lane sweeps:

| Context | Dense 27B N | Flash-Next N |
| ---: | ---: | ---: |
| 1K | 16 | 9 |
| 4K | 16 | 5 |
| 16K | 8 | 2 |
| 32K | 4 | 1 |
| 64K | 2 | 1 |

Flash's 32K and 64K rows are single-lane sentinels. They cannot be described as
continuous-MTP performance because the coordinator requires at least two live
lanes. A separate N=1 pass is retained for both models as a fallback/control,
not as the headline comparison.

Run separate server phases in A-B-B-A order: baseline ascending, stack
ascending, stack descending, baseline descending. Model processes remain
serialized. Before each phase, use the lab's predeclared thermal-settle band and
label the phase cold/hot; do not rely on an arbitrary sleep. Capture exact
launch argv/environment, `/v1/models`, `/metrics` before and after, process and
system memory, server log, and an atomic result checkpoint after every cell.

The stack log must prove that the continuous coordinator installed for every
N>=2 cell and contain no refusal/fallback. Metrics are supporting evidence only
until continuous-path acceptance counters are live-wired. Flash NAX is expected
only when the sparse route's query and physical-KV thresholds are crossed;
dense 27B is architecture-ineligible.

## Ten-prompt semantic regression screen

The quality bank reuses nine vetted cross-domain prompts and adds a 16K grounded
incident-synthesis sentinel. It covers advanced async code and repair,
distributed leases, hybrid scheduling, science, history, Bayes, constraint
reasoning, false-premise resistance, and long-context factual synthesis.

The primary quality profile is greedy. This is a methodological requirement,
not a preference: the current continuous route rejects non-greedy sampling, so
a production-sampling A/B would silently test a fallback path. Submit identical
4+4+2 cohorts to both arms, preserve visible content, reasoning, raw response,
usage, finish reason, cache accounting, lane mapping, and request body. An empty
visible answer fails even if reasoning text exists.

After capture, create an A/B packet with labels, timing, reasoning, and route
telemetry removed. Randomize X/Y independently per model/prompt and show every
pair again in reversed order. A human or independent strong judge scores
correctness, coverage, instruction adherence, and relevance/clarity from 0-4;
verbosity earns no credit. The local 4B judge may triage but is not the sole
promotion authority.

The predeclared screen requires each model to pass independently: no new
critical error, at most two blinded candidate losses per model and three
overall, no high-confidence correctness loss, mean paired total delta at least
-0.5/16, median delta at least zero, candidate mean correctness at least 3/4,
candidate mean total at least 12/16, and at least 90% position consistency.
Bayes, the logic grid, response formats, length limits, incident facts, and both
async implementations also get deterministic sidecars. Generated code must be
compiled and exercised in an isolated sandbox; fluent prose is not sufficient.

## Prepared commands

The default operations are CPU-only:

```bash
python bench/continuous_self_mtp_campaign.py validate
python bench/continuous_self_mtp_campaign.py plan --profile product \
  --output /tmp/qwen38-product-plan.json
python bench/continuous_self_mtp_campaign.py launch-command \
  --arm full_stack --model-key qwen38_27b
```

`launch-command` only prints the exact command. The live context and quality
clients require both `--execute` and
`RAPID_MLX_BENCHMARK_EXECUTE=YES_LOAD_MODELS`. This two-part interlock is the
boundary between preparation and authorized model/GPU work.

## Qualification boundaries

- APC restore is inactive in the primary cold ladder by design. Add a separate
  warm-prefix profile; do not claim APC benefit from the cold results.
- Rapid's combined candidate has no live PLE-NVMe activation seam. The 32 GB
  Flash PLE file is inventoried but not claimed as an enabled Rapid lever.
- Rapid's current continuous route still refuses quantized/windowed caches.
  The pinned mlx-lm dependency contains the new quantized self-MTP/APC work,
  but that is not a Rapid reached-path claim until the Rapid gate/adapter is
  ported and its own tests pass. The primary comparison therefore holds KV at
  BF16.
- Dense 27B cannot exercise Flash-only QSA/NAX/PLE mechanisms. “Full stack”
  means every applicable, reached-path lever, with ineligible and inactive
  levers recorded explicitly.
- Add a staggered-arrival turnover smoke for dense 27B to qualify dynamic
  membership; synchronized ladder cohorts do not test joins.

The source of truth for pins, artifact hashes, concurrency, commands, and gates
is `bench/continuous_self_mtp_campaign.json`; the quality prompts and rubrics
are in `bench/continuous_self_mtp_quality.json`.
