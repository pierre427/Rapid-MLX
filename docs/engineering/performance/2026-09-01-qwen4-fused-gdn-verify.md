# Qwen4 fused GDN speculative verify

## Scope

The [single-token fused GDN decode kernel](2026-09-01-qwen4-fused-gdn-decode.md)
refuses every forward that records speculative rollback, so it never runs on
the MTP verify forward that dominates a self-speculative decode cycle. This
report adds a sibling kernel for the B=1 verify block of width `k + 1`
(`2 <= k + 1 <= 8`). One Metal dispatch per GDN layer performs the causal
convolution, SiLU, q/k L2 normalization, decay and beta gates, the fp32 delta
recurrence and the sigmoid-gated RMSNorm for every token of the block, and
additionally writes the per-position restore points the stock path publishes
through `Qwen4ExpStateCache.record_slot_snapshots`: the recurrent state after
each of the first `k` tokens and the convolution window after each of them.

The path is opt-in through `RAPID_MLX_QWEN4_FUSED_GDN_VERIFY=1` or the
resident `set_qwen4_fused_gdn_verify_mode(model, "stock" | "fused")` selector,
independent of the decode selector. Batching, masks, ragged caches, training,
sharding, uninitialized caches, single-token decode and widths above 8 keep
their existing paths, and every refusal is counted with its reason in
`qwen4_fused_gdn_stats` (`verify_calls`, `verify_fallbacks`,
`verify_last_fallbacks`).

## A one-value rounding defect, found and fixed in both kernels

The first full-model gate matched for two verify rounds and then diverged by
one bf16 ULP in one GDN layer. Replaying that layer's captured inputs in
isolation was bit-exact, so the divergence entered one layer earlier as an
output-only flip. A random-input search on one real layer reproduced it at
larger activation scale (25 of 400 blocks at scale 4.0, none at scale 1.0),
and a debug-instrumented copy of the kernel bisected it to the beta gate:
`mlx_sigmoid_fast<bf16>` returned `0.0010681152` where MLX's `mx.sigmoid`
returns `0.0010604858`.

An exhaustive sweep over all 65,280 finite bf16 values, on MLX 0.32.1 and on
0.32.2.dev, settled every sigmoid boundary the kernels use:

| Boundary | Stock op | Exact form | Rejected form |
|---|---|---|---|
| beta gate (bf16) | `mx.sigmoid` | `mlx_sigmoid_precise<T>` (0 mismatches) | `mlx_sigmoid_fast<T>` (1 mismatch, x ~ -6.85) |
| SiLU (bf16) | `nn.silu` | `x * mlx_sigmoid_fast(x)` (0 mismatches) | `x * mlx_sigmoid_precise(x)` (1 mismatch) |
| output gate (float32 of bf16 z) | `mx.sigmoid(z.astype(float32))` | `mlx_sigmoid_precise<float>` (0 mismatches) | `mlx_sigmoid_fast<float>` (628 mismatches) |

The same sweep confirmed the decay formula (`mlx_softplus_fast` plus
`metal::precise::exp`) against the compiled `_compute_g_beta` decay for every
bf16 `alpha + dt_bias` at six `A_log` values, every rsqrt form against
`mx.rsqrt`, and the per-lane bf16 partial-sum q/k reduction against `mx.sum`
at input scales 0.05 through 16.

The single-token decode kernel shared the two rejected forms (beta gate and
output gate). Both kernels now use the exact forms, and
`tests/test_qwen4_fused_gdn_verify.py` pins them with the same exhaustive
sweep in the Apple lane (it also asserts that the rejected beta form differs
on exactly one value, so the sweep is not vacuous). A 32-step layer trajectory
or a token-only end-to-end comparison cannot see a one-value boundary; an
exhaustive bf16 sweep per unary boundary can.

## Verification

Environment: Apple M5 Max, 128 GB, Python 3.12, MLX 0.32.1, PyPI mlx-lm
0.31.3 (the versions this repository pins), Rapid's vendored `qwen4_exp`
pinned explicitly, checkpoint `Qwen3.8-Flash-Next-MLX-4bit-MTP` converted by
the native mlx-lm family (see the stack note for the two load-time
translations that requires), `MLX_ENABLE_TF32=0`. Every phase records free
memory, swap, `:8282` listeners, engine processes and any process holding
8 GiB or more, before and after the phase; the run fails on any violation
and refuses to start on a shared GPU or an unreadable scan. No violation was
recorded in the run reported below.

Synthetic Metal contracts (`tests/test_qwen4_fused_gdn_verify.py`): widths 2,
3, 5 and 8 reproduce the stock verify path (`gated_delta_verify_with_states`
plus the stock convolution, normalization and gated RMSNorm) bit for bit for
output, both cache slots and every restore point, through alternating commit
and restore cycles; the layer-level test drives `GatedDeltaNet` itself on 2-slot
and 4-slot (PLE-style, slots 3 and 2 pre-staged) caches through
`restore_rollback` at every boundary.

Real-weight gate (`scripts/bench_qwen4_fused_gdn_verify.py --execute-metal`),
one resident process per run. Two complete runs are on record, receipts
`qwen4-fused-gdn-verify-rapid-final2-20260901.json` and, with the final
revision of the script (whose methodology hash the receipt carries),
`qwen4-fused-gdn-verify-rapid-final3-20260901.json`. Correctness was
identical in both; the timing tables below give both.

- Layer phase (one production GDN layer, width 3): 32 verify blocks exact for
  output, both cache slots and all restore points through cycling restores
  (`n_to_drop` 0, 1 and 2 each covered), 32 fused calls, zero fallbacks. Eight
  interleaved 64-block observations: stock 7.53 ms / fused 6.53 ms medians
  (**+15.4%**) in the first complete run and 6.95 ms / 5.53 ms (**+25.6%**)
  in the second, input and output projections included. Earlier partial runs
  of the same phase measured +9.3%, +10.1% and +24.6%, and +1.2% during a run
  where both arms were running at half speed; the layer-level number is
  sensitive to host state and is not a whole-model claim.
- Rounds phase (full model, width 3, engine acceptance and rollback rules,
  drafts scripted from a width-3 oracle on a cloned cache so that acceptance
  0, 1 and 2 each occur eight times, zero schedule misses): 24 of 24 rounds
  exact for full logits, every GDN slot, every restore point and every
  attention KV and QSA state array; 864 of 864 fused calls, zero fallbacks.
- End-to-end phase (vendored `mtp_generate_step` on the inner text model,
  greedy, prompt lookup off, draft depth clamped to the head's advertised
  cap of 1, so verify width 2; 256 tokens; four repeats per mode with
  alternating order): all eight token streams share one SHA-256; every fused
  run recorded 4,716 fused calls (36 layers x 131 verify rounds) and zero
  fallbacks; 125 of 131 rounds accepted the draft.

| Run | Mode | Decode tok/s (4 runs) | Median |
|---|---|---|---:|
| final2 | stock | 62.14, 62.91, 62.73, 62.55 | 62.64 |
| final2 | fused | 66.04, 66.00, 65.54, 65.61 | 65.80 |
| final3 | stock | 63.01, 63.17, 60.41, 62.69 | 62.85 |
| final3 | fused | 66.29, 66.09, 62.85, 62.19 | 64.47 |

That is **+5.1%** and **+2.6%** median decode throughput on the real
speculative loop at the engine's own verify width. Per interleaved pair the
fused arm was ahead in seven of eight pairs (+4.1% to +6.3%) and behind once
(-0.8%, the last pair of the second run, where both arms had slowed).
The whole-cycle gain is what removing one GDN dispatch chain per verify
forward on 36 layers buys at width 2; it is an incremental, composable win,
not a headline.

After the sigmoid fix, a random-input search on one real layer was also exact
for 400 blocks at each of input scales 1.0, 4.0 and 8.0 (1,200 blocks,
including restore points).

## Stack note

- Rapid's `_register_vendored_archs` prefers a native `mlx_lm.models.qwen4_exp`
  when the installed mlx-lm ships one. Deployments that pair Rapid with such
  an mlx-lm run the native GDN layers, and neither fused kernel engages there.
  The gate pins the vendored module (`--force-vendored-arch`, default on).
- A checkpoint converted by the native mlx-lm family needs two translations
  to load into the vendored modules, both applied by the gate: its PLE
  n-gram tables are named `shard_N` there and `shards.N` here, so the
  per-path quantization overrides (group 32 for the 160-wide tables) are
  re-keyed; and it stores zero-centered RMSNorm gains as full gains, while
  `ZeroCenteredRMSNorm` adds one to the stored delta, so those gains are
  recentered (`--norm-convention auto`, detected from the conv1d layout as
  the native loader does; 148 backbone and 9 MTP-head gains). Without the
  second translation the model loads but emits gibberish with zero draft
  acceptance.
- Rapid's Qwen4 MTP injection advertises `mtp_max_speculative_tokens = 1`, so
  the engine verifies width 2; the gate's layer and rounds phases exercise
  width 3 as well.

Reproduce the layer gate on an idle GPU:

```bash
MLX_ENABLE_TF32=0 PYTHONPATH=. python scripts/bench_qwen4_fused_gdn_verify.py \
  --execute-metal --layer-only \
  --model /path/to/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --output /tmp/qwen4-fused-gdn-verify-layer.json
```

Reproduce the full gate (layer, engine-rule rounds, end-to-end):

```bash
MLX_ENABLE_TF32=0 PYTHONPATH=. python scripts/bench_qwen4_fused_gdn_verify.py \
  --execute-metal --min-free-percent 10 --max-swap-growth-mib 8192 \
  --model /path/to/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --output /tmp/qwen4-fused-gdn-verify.json
```

The memory floor of 10% reflects a 97 GiB resident load on a 128 GB host;
keep the default 20% on larger hosts.

## Failure lifecycle

Admission, probe and dispatch failures leave the request cache untouched and
use the stock path, exactly like the decode kernel. The verify path publishes
its restore points before it assigns the live slots; if the cache's snapshot
contract raises (a PLE slot that was never staged), the staging area is
unwound before the error propagates, so the cache is left as it was before
the call. A later Metal command-buffer failure is handled at the generation
boundary as for decode.
