#!/usr/bin/env python3
"""Run the released Qwen4 artifact through Rapid's continuous self-MTP gate.

The gate loads the checkpoint once, injects its native MTP sidecar, and records
plain-target, single-lane MTP, and fixed two-lane MTP token streams.  Greedy
single-lane streams are required to match the plain target exactly; the two
lane stream is measured twice so determinism and batch-shape effects remain
explicit instead of being hidden behind a throughput number.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from qwen4_exp_real_parity import _load

PROMPTS = (
    "In one paragraph of at least 80 words, explain why transactional cache "
    "rollback matters during speculative decoding.",
    "In one paragraph of at least 80 words, explain how continuous batching "
    "improves inference throughput.",
    "In one paragraph of at least 80 words, explain why exact artifact hashes "
    "matter when validating a model port.",
    "In one paragraph of at least 80 words, explain how fail-closed admission "
    "protects an inference service.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _write(output: Path, result: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _release_temporary_state() -> None:
    gc.collect()
    mx.clear_cache()


def _tokens(tokenizer, prompt: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _plain_greedy(language, prompt: list[int], max_tokens: int) -> dict[str, Any]:
    from mlx_lm.models.cache import make_prompt_cache

    started = time.monotonic()
    cache = make_prompt_cache(language)
    remaining = mx.array(prompt, dtype=mx.uint32)
    while int(remaining.shape[0]) > 1:
        width = min(512, int(remaining.shape[0]) - 1)
        output = language(remaining[:width][None], cache=cache)
        mx.eval(output, [item.state for item in cache])
        remaining = remaining[width:]
    output = language(remaining[None], cache=cache)
    mx.eval(output)
    prefill_s = time.monotonic() - started

    generated: list[int] = []
    decode_started = time.monotonic()
    for index in range(max_tokens):
        token = int(mx.argmax(output[:, -1, :], axis=-1).item())
        generated.append(token)
        if index + 1 < max_tokens:
            output = language(mx.array([[token]], dtype=mx.uint32), cache=cache)
    mx.synchronize()
    decode_s = time.monotonic() - decode_started
    del output, cache
    _release_temporary_state()
    return {
        "tokens": generated,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": max_tokens / decode_s,
    }


def _plain_batched(
    language,
    prompts: list[list[int]],
    max_tokens: int,
    cache_merge,
) -> dict[str, Any]:
    from mlx_lm.models.cache import make_prompt_cache

    caches = []
    current = []
    prefill_started = time.monotonic()
    for prompt in prompts:
        cache = make_prompt_cache(language)
        remaining = mx.array(prompt, dtype=mx.uint32)
        while int(remaining.shape[0]) > 1:
            width = min(512, int(remaining.shape[0]) - 1)
            output = language(remaining[:width][None], cache=cache)
            mx.eval(output, [item.state for item in cache])
            remaining = remaining[width:]
        output = language(remaining[None], cache=cache)
        current.append(int(mx.argmax(output[:, -1, :], axis=-1).item()))
        caches.append(cache)
    merged = cache_merge(caches, "plain target")
    mx.synchronize()
    prefill_s = time.monotonic() - prefill_started

    generated = [[token] for token in current]
    decode_started = time.monotonic()
    for _ in range(max_tokens - 1):
        output = language(mx.array(current, dtype=mx.uint32)[:, None], cache=merged)
        current = mx.argmax(output[:, -1, :], axis=-1).tolist()
        for row, token in enumerate(current):
            generated[row].append(int(token))
    mx.synchronize()
    decode_s = time.monotonic() - decode_started
    del output, merged, caches
    _release_temporary_state()
    return {
        "tokens": generated,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": (len(prompts) * max_tokens) / decode_s,
    }


def _continuous(
    runtime,
    prompts: list[list[int]],
    max_tokens: int,
) -> dict[str, Any]:
    from vllm_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from vllm_mlx.spec_decode.mtp.continuous_engine import SelfMTPLaneSpec

    specs = [
        SelfMTPLaneSpec(
            uid=index + 1,
            prompt=prompt,
            max_tokens=max_tokens,
            num_draft=2,
        )
        for index, prompt in enumerate(prompts)
    ]
    prefill_started = time.monotonic()
    driver = ContinuousMTPDriver.create(specs, runtime)
    mx.synchronize()
    prefill_s = time.monotonic() - prefill_started

    generated = {spec.uid: [] for spec in specs}
    from_draft = {spec.uid: 0 for spec in specs}
    finish_reasons: dict[int, str | None] = {spec.uid: None for spec in specs}
    response_calls = 0
    turnovers = 0
    decode_started = time.monotonic()
    while True:
        while driver.has_work:
            response_calls += 1
            for response in driver.next():
                generated[response.uid].append(response.token)
                from_draft[response.uid] += int(response.from_draft)
                if response.finish_reason is not None:
                    finish_reasons[response.uid] = response.finish_reason
        resumed = driver.resume_turnover()
        if not resumed:
            break
        turnovers += 1
    mx.synchronize()
    decode_s = time.monotonic() - decode_started

    if any(len(tokens) != max_tokens for tokens in generated.values()):
        raise AssertionError(
            f"continuous MTP delivered unexpected lengths: "
            f"{ {uid: len(tokens) for uid, tokens in generated.items()} }"
        )
    if any(reason != "length" for reason in finish_reasons.values()):
        raise AssertionError(
            f"continuous MTP did not finish by length: {finish_reasons}"
        )

    result = {
        "tokens": {str(uid): tokens for uid, tokens in generated.items()},
        "from_draft": {str(uid): count for uid, count in from_draft.items()},
        "finish_reasons": {str(uid): reason for uid, reason in finish_reasons.items()},
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": (len(prompts) * max_tokens) / decode_s,
        "response_calls": response_calls,
        "turnovers": turnovers,
    }
    del driver
    _release_temporary_state()
    return result


def _sequential_arm(runtime, prompts: list[list[int]], max_tokens: int):
    lanes = [_continuous(runtime, [prompt], max_tokens) for prompt in prompts]
    return {
        "lanes": lanes,
        "prefill_s": sum(lane["prefill_s"] for lane in lanes),
        "decode_s": sum(lane["decode_s"] for lane in lanes),
        "decode_tps": (len(prompts) * max_tokens)
        / sum(lane["decode_s"] for lane in lanes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2, choices=(2, 4))
    args = parser.parse_args()
    if args.max_tokens < 3:
        parser.error("--max-tokens must be at least 3")

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    prompts = PROMPTS[: args.batch_size]
    result: dict[str, Any] = {
        "status": "loading",
        "checkpoint": str(checkpoint),
        "artifact": {
            "config_sha256": _sha256(checkpoint / "config.json"),
            "index_sha256": _sha256(checkpoint / "model.safetensors.index.json"),
        },
        "environment": {
            "rapid_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "prompts": list(prompts),
        "arms": {},
    }
    _write(output, result)

    load_started = time.monotonic()
    holder, _layers, language = _load(checkpoint, "rapid")
    mx.eval(holder.parameters())
    result["load_s"] = time.monotonic() - load_started
    result["status"] = "loaded"
    _write(output, result)

    from mlx_lm.utils import load_tokenizer

    from vllm_mlx.spec_decode.mtp import dispatch_mtp_inject, dispatch_mtp_validate
    from vllm_mlx.spec_decode.mtp.continuous_runtime import (
        assemble_continuous_self_mtp_runtime,
    )

    tokenizer = load_tokenizer(checkpoint)
    prompt_tokens = [_tokens(tokenizer, prompt) for prompt in prompts]
    result["prompt_tokens"] = [len(tokens) for tokens in prompt_tokens]

    plain = [
        _plain_greedy(language, tokens, args.max_tokens) for tokens in prompt_tokens
    ]
    for lane in plain:
        lane["text"] = tokenizer.decode(lane["tokens"])
    result["arms"]["plain_target"] = plain
    result["status"] = "plain-complete"
    _write(output, result)

    inject_started = time.monotonic()
    if not dispatch_mtp_inject(
        holder, "qwen4_exp", mtp_sidecar=checkpoint, allow_random_init=False
    ):
        raise RuntimeError("released Qwen4 MTP sidecar injection failed")
    if not dispatch_mtp_validate(holder, "qwen4_exp"):
        raise RuntimeError("injected Qwen4 MTP ABI validation failed")
    runtime = assemble_continuous_self_mtp_runtime(holder)
    result["inject_s"] = time.monotonic() - inject_started
    result["status"] = "injected"
    _write(output, result)

    plain_batch = _plain_batched(
        language,
        prompt_tokens,
        args.max_tokens,
        runtime.caches._merge,
    )
    result["arms"][f"plain_b{args.batch_size}"] = plain_batch
    result["status"] = f"plain-b{args.batch_size}-complete"
    _write(output, result)

    seq_a = _sequential_arm(runtime, prompt_tokens, args.max_tokens)
    result["arms"]["mtp_b1_sequential_a"] = seq_a
    result["status"] = "b1-a-complete"
    _write(output, result)

    batch_a = _continuous(runtime, prompt_tokens, args.max_tokens)
    batch_a_name = f"mtp_b{args.batch_size}_a"
    result["arms"][batch_a_name] = batch_a
    result["status"] = f"b{args.batch_size}-a-complete"
    _write(output, result)

    seq_b = _sequential_arm(runtime, prompt_tokens, args.max_tokens)
    result["arms"]["mtp_b1_sequential_b"] = seq_b
    result["status"] = "b1-b-complete"
    _write(output, result)

    batch_b = _continuous(runtime, prompt_tokens, args.max_tokens)
    batch_b_name = f"mtp_b{args.batch_size}_b"
    result["arms"][batch_b_name] = batch_b

    plain_tokens = [lane["tokens"] for lane in plain]
    b1_a_tokens = [lane["tokens"]["1"] for lane in seq_a["lanes"]]
    b1_b_tokens = [lane["tokens"]["1"] for lane in seq_b["lanes"]]
    batch_a_tokens = [
        batch_a["tokens"][str(uid)] for uid in range(1, args.batch_size + 1)
    ]
    batch_b_tokens = [
        batch_b["tokens"][str(uid)] for uid in range(1, args.batch_size + 1)
    ]
    checks = {
        "b1_a_equals_plain": b1_a_tokens == plain_tokens,
        "b1_b_equals_plain": b1_b_tokens == plain_tokens,
        "batch_repeat_exact": batch_a_tokens == batch_b_tokens,
        "batch_equals_plain_matched_shape": batch_a_tokens == plain_batch["tokens"],
    }
    result["checks"] = checks
    result["comparisons"] = {
        "plain_batch_equals_plain_b1": plain_batch["tokens"] == plain_tokens,
        "mtp_batch_equals_plain_b1": batch_a_tokens == plain_tokens,
    }
    result["performance"] = {
        "b1_bracket_decode_tps": [seq_a["decode_tps"], seq_b["decode_tps"]],
        "batch_repeat_decode_tps": [
            batch_a["decode_tps"],
            batch_b["decode_tps"],
        ],
        "batch_over_b1_a": batch_a["decode_tps"] / seq_a["decode_tps"],
        "batch_over_b1_b": batch_b["decode_tps"] / seq_b["decode_tps"],
        "plain_batch_decode_tps": plain_batch["decode_tps"],
        "batch_over_plain_matched_shape": batch_a["decode_tps"]
        / plain_batch["decode_tps"],
    }
    result["decoded"] = {
        "plain": [tokenizer.decode(tokens) for tokens in plain_tokens],
        "plain_batch": [tokenizer.decode(tokens) for tokens in plain_batch["tokens"]],
        "batch": [tokenizer.decode(tokens) for tokens in batch_a_tokens],
    }
    result["status"] = "passed" if all(checks.values()) else "failed"
    _write(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
