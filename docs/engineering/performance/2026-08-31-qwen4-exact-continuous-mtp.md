# Qwen4 exact continuous-MTP qualification

Date: 2026-08-31

Status: B2 exact but slower than matched plain batching; B4 blocked

Route policy: explicit and default-off

## Artifact and environment

The gate used the released Qwen4 experimental checkpoint with:

- config SHA-256 `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`;
- index SHA-256 `f643b2bd4f768a6a68fcdec89870f080f71420e8adbd084801291835df0cca5c`;
- 48 layers in a repeating GDN/GDN/GDN/QSA topology;
- one native MTP layer, PLE beginning at layer 2, QSA budget 2048 and
  compression ratio 4;
- affine 4-bit model weights and BF16 transactional cache state.

The serialized runs used Apple Silicon with MLX 0.32.0. The harness records the
exact Rapid commit, Python, OS, MLX/mlx-lm versions, artifact hashes, prompt
lengths, token streams, acceptance counts, turnover count, and timing in each
JSON receipt.

## Correctness fixes required by the artifact

The first real transaction exposed three defects that model-free tests did not:

1. separately loaded MTP-sidecar norm gains did not pass through the target
   model's one-centered to zero-centered conversion;
2. a K=2 rejection crossing a QSA compression boundary needed a bounded raw-ring
   journal and cache-owned transaction checkpoint; and
3. block-shaped target verification changed greedy Qwen4 output. Tokenwise
   target forwards preserve the recurrent state boundary exactly.

The adapter therefore advertises a tokenwise-exact verify mode, fixed membership
only, BF16 cache state, and an exact maximum width of two lanes.

## Results

Four fixed prompts generated 24 tokens per lane. Each speculative cell ran
twice around independently repeated single-lane controls. Ordinary batching at
the same width is the product comparator.

| Route | Decode throughput | Correctness |
| --- | ---: | --- |
| sequential ordinary B1 bracket | 28.90 / 29.44 tok/s | reference |
| continuous MTP B2 repeats | 49.43 / 49.56 tok/s | exact to B1 and matched plain B2; one turnover exact |
| ordinary plain B2 | 69.35 tok/s | matched comparator |
| continuous MTP B4 repeats | 67.77 / 68.63 tok/s | deterministic, but final companion token diverged after turnover |
| ordinary plain B4 | 104.01 tok/s | matched comparator |

Continuous B2 is 1.68--1.71x the throughput of sequential B1 streams but only
0.713x ordinary B2. Continuous B4 is about 2.34x sequential B1 and 0.652x
ordinary B4. Sequential B1 is therefore a useful scaling diagnostic, not a
shipping comparator for a multi-request route.

## Decision

- Keep ordinary batching as the multi-request default.
- Keep continuous Qwen4 MTP explicit and default-off.
- Admit at most two fixed lanes; B4 remains blocked by turnover exactness.
- Do not claim parity across arbitrary quantized batch shapes. Require exactness
  against ordinary batching at the same shape.
- Preserve the separately qualified single-user MTP policy; this gate does not
  weaken it.

## Reproduction

```bash
python scripts/qwen4_exp_continuous_mtp_gate.py \
  --checkpoint /path/to/released-qwen4-checkpoint \
  --batch-size 2 \
  --max-tokens 24 \
  --output /tmp/qwen4-continuous-b2.json

python scripts/qwen4_exp_continuous_mtp_gate.py \
  --checkpoint /path/to/released-qwen4-checkpoint \
  --batch-size 4 \
  --max-tokens 24 \
  --output /tmp/qwen4-continuous-b4.json
```

GPU cells must run serially. Preserve both JSON receipts, and compare MTP with
the same-width ordinary batch rather than only with sequential B1.
