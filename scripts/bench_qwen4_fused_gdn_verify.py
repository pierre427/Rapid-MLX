#!/usr/bin/env python3
"""Real-weight gate for Rapid's default-off Qwen4 fused GDN speculative verify.

Three resident phases on one loaded checkpoint, each comparing the stock
verify path against the fused verify kernel bit for bit:

1. ``layer``: one production GDN layer runs ``--layer-blocks`` verify blocks
   of width ``k + 1`` through engine-style restores, comparing output, both
   cache slots and every published restore point; then interleaved timing.
2. ``rounds``: the full model runs greedy verify rounds (committed token plus
   ``k`` deterministic drafts) with the engine's acceptance and rollback
   rules, comparing logits, every GDN cache slot, every restore point and the
   attention cache offsets after each round.
3. ``e2e``: the vendored MTP generator produces the same request in both
   modes with reached-path receipts and interleaved decode timing.

The script records memory and concurrency receipts before and after each
phase and refuses to load when the host is not idle enough.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json

PLAN = {
    "scope": "Qwen4 B=1 speculative verify (width k+1) with resident real weights",
    "correctness": (
        "layer: exact output, cache slots and restore points through restores; "
        "rounds: exact logits, GDN slots, restore points and attention offsets "
        "through engine-rule rollbacks; e2e: identical token streams with exact "
        "fused-call receipts"
    ),
    "timing": "interleaved stock/fused observations without model reload",
    "excluded": ["prefill", "batch", "mask", "ragged caches", "single-token decode"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--execute-metal", action="store_true")
    parser.add_argument("--draft-k", type=int, default=2)
    parser.add_argument("--layer-blocks", type=int, default=32)
    parser.add_argument("--layer-timing-blocks", type=int, default=64)
    parser.add_argument("--layer-repeats", type=int, default=8)
    parser.add_argument("--verify-rounds", type=int, default=24)
    parser.add_argument(
        "--prompt",
        default="Write the integers 1 through 200, one per line.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="wrap --prompt in the tokenizer's chat template (user turn)",
    )
    parser.add_argument("--e2e-repeats", type=int, default=4)
    parser.add_argument(
        "--layer-only",
        action="store_true",
        help="lazy-load and materialize one GDN layer; run only the layer phase",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="on a rounds mismatch, save the first differing layer's inputs here",
    )
    parser.add_argument("--skip-rounds", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument(
        "--force-vendored-arch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pin Rapid's vendored qwen4_exp even when mlx-lm ships a native one",
    )
    parser.add_argument(
        "--norm-convention",
        choices=("auto", "raw", "converted"),
        default="auto",
        help=(
            "how the checkpoint stores zero-centered RMSNorm gains: 'raw' deltas "
            "(what Rapid's vendored modules expect) or 'converted' full gains "
            "(what the native mlx-lm converter writes); 'auto' detects from the "
            "conv1d weight layout like the native loader does"
        ),
    )
    parser.add_argument("--min-free-percent", type=int, default=20)
    parser.add_argument("--max-swap-growth-mib", type=float, default=2048.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without MLX or a checkpoint)
# ---------------------------------------------------------------------------


def expected_verify_calls(mode: str, fused_layers: int, verify_rounds: int):
    """Return the exact ``(verify_calls, verify_fallbacks)`` receipt."""
    if mode == "stock":
        return 0, 0
    if mode != "fused":
        raise ValueError(f"unknown mode {mode!r}")
    return fused_layers * verify_rounds, 0


def greedy_accept_count(verify_argmax, drafts) -> int:
    """Leading drafts confirmed by the greedy verify, as the engine counts."""
    accepted = 0
    for predicted, draft in zip(verify_argmax, drafts, strict=False):
        if int(predicted) != int(draft):
            break
        accepted += 1
    return accepted


def next_round_tokens(verify_argmax, drafts, filler: int):
    """Derive the next committed token and deterministic drafts.

    ``verify_argmax`` holds ``k + 1`` greedy predictions for ``[y, d_1..d_k]``.
    The bonus token at the first unconfirmed position becomes the next ``y``;
    later predictions (made under rejected context) serve as the next drafts,
    padded with ``filler``. Identical logits therefore yield identical rounds.
    """
    k = len(drafts)
    accepted = greedy_accept_count(verify_argmax, drafts)
    bonus = int(verify_argmax[accepted])
    guesses = [int(token) for token in verify_argmax[accepted + 1 : accepted + 1 + k]]
    while len(guesses) < k:
        guesses.append(int(filler))
    return bonus, guesses, accepted


def parse_free_percent(text: str) -> float | None:
    match = re.search(r"free percentage:\s*(\d+)%", text)
    return float(match.group(1)) if match else None


def parse_swap_used_mib(text: str) -> float | None:
    match = re.search(r"used = ([0-9.]+)M", text)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Host receipts
# ---------------------------------------------------------------------------


LARGE_PROCESS_RSS_KIB = 8 * 1024 * 1024  # 8 GiB resident: a model-sized tenant

# Engine signatures worth refusing a timing gate for. Only the executable and
# its first arguments are inspected: a shell or an agent whose *prompt text*
# mentions python and a model is not a GPU tenant, and a static file server
# or an MCP helper written in Python is not one either. Processes holding a
# model-sized resident set are caught separately by ``large_processes``.
_MODEL_MODULE_PATTERN = re.compile(
    r"vllm_mlx|mlx_lm|mlx-lm|mlx_vlm|bench_|serve\.sh"
    r"|(^|/)(rapid-mlx|rmlx|vllm-mlx)(\s|$)",
    re.IGNORECASE,
)
_MODEL_EXECUTABLES = (
    "lmstudio",
    "ollama",
    "llama-server",
    # this project's console scripts (pyproject [project.scripts])
    "rapid-mlx",
    "rmlx",
    "vllm-mlx",
)


def is_model_process(ps_line: str, own_pid: int) -> bool:
    """True for a ``ps -axo pid,rss,command`` line that looks like an engine."""
    parts = ps_line.split(maxsplit=2)
    if len(parts) < 3 or not parts[0].isdigit() or int(parts[0]) == own_pid:
        return False
    argv = parts[2].split()
    executable = os.path.basename(argv[0]).lower()
    if executable.startswith("python"):
        return _MODEL_MODULE_PATTERN.search(" ".join(argv[1:4])) is not None
    return any(name in executable for name in _MODEL_EXECUTABLES)


def _run(command, *, ok_codes=(0,)):
    """Return a command's stdout, or ``None`` when it could not be trusted."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode not in ok_codes:
        return None
    return completed.stdout


def host_receipt(mx=None):
    memory = _run(["memory_pressure"])
    swap = _run(["sysctl", "vm.swapusage"])
    # lsof exits 1 when nothing matches, which is the idle case.
    listeners = _run(
        ["/usr/sbin/lsof", "-nP", "-iTCP:8282", "-sTCP:LISTEN"], ok_codes=(0, 1)
    )
    processes = _run(["ps", "-axo", "pid,rss,command"])
    process_lines = [] if processes is None else processes.splitlines()[1:]
    receipt = {
        "scan_ok": None not in (memory, swap, listeners, processes)
        and bool(process_lines),
        "free_percent": parse_free_percent(memory or ""),
        "swap_used_mib": parse_swap_used_mib(swap or ""),
        "port_8282_listeners": len(
            [line for line in (listeners or "").splitlines()[1:] if line.strip()]
        ),
        "model_processes": [
            line.strip()[:160]
            for line in process_lines
            if is_model_process(line, os.getpid())
        ],
        # Any foreign process holding a model-sized resident set counts as a
        # GPU tenant whatever its command line says.
        "large_processes": [
            line.strip()[:160]
            for line in process_lines
            if line.strip()
            and int(line.split()[0]) != os.getpid()
            and int(line.split()[1]) >= LARGE_PROCESS_RSS_KIB
        ],
        "timestamp": time.time(),
    }
    if mx is not None:
        receipt["mlx_active_mib"] = mx.get_active_memory() / 2**20
        receipt["mlx_peak_mib"] = mx.get_peak_memory() / 2**20
    return receipt


class HostAbortError(RuntimeError):
    pass


def host_violation(receipt, baseline, args, phase) -> str | None:
    """Return why this receipt violates the host budget, failing closed.

    A receipt whose scans could not run, one that shows another engine, a
    model-sized resident process or a ``:8282`` listener, and one outside the
    memory budget all count; the check runs on every receipt, not only the
    first, so a tenant that appears mid-run fails the run.
    """
    if not receipt.get("scan_ok"):
        return f"{phase}: host scans unreadable"
    tenants = {
        key: receipt.get(key)
        for key in ("port_8282_listeners", "model_processes", "large_processes")
        if receipt.get(key)
    }
    if tenants:
        return f"{phase}: GPU tenant present {json.dumps(tenants)[:400]}"
    free = receipt.get("free_percent")
    before = baseline.get("swap_used_mib")
    now = receipt.get("swap_used_mib")
    if free is None or now is None or before is None:
        return f"{phase}: host memory receipt unreadable"
    if free < args.min_free_percent:
        return f"{phase}: free memory {free}% below {args.min_free_percent}%"
    if now - before > args.max_swap_growth_mib:
        return f"{phase}: swap grew {now - before:.0f} MiB"
    return None


def check_host(receipt, baseline, args, phase):
    violation = host_violation(receipt, baseline, args, phase)
    if violation is not None:
        raise HostAbortError(violation)


LOCK_PATH = Path("/tmp/rapid-qwen4-fused-gdn-verify.lock")


def acquire_gate_lock():
    """Hold an exclusive lock so two gates never share the GPU."""
    handle = open(LOCK_PATH, "w")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise HostAbortError(f"another gate holds {LOCK_PATH}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


# ---------------------------------------------------------------------------
# MLX helpers
# ---------------------------------------------------------------------------


def clone_state_cache(cache, cache_type, mx):
    clone = cache_type(len(cache.cache))
    clone.cache = [None if v is None else mx.array(v) for v in cache.cache]
    return clone


def arrays_equal(a, b, mx) -> bool:
    return bool(mx.array_equal(a, b).item())


def max_abs(a, b, mx) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def restore_points_equal(stock_cache, fused_cache, mx):
    stock = stock_cache.rollback_state or []
    fused = fused_cache.rollback_state or []
    if len(stock) != len(fused):
        return False, f"restore point count {len(stock)} vs {len(fused)}"
    for position, (left, right) in enumerate(zip(stock, fused, strict=True)):
        if len(left) != len(right):
            return False, f"slot count at position {position}"
        for slot, (a, b) in enumerate(zip(left, right, strict=True)):
            if a is None or b is None:
                if a is not b:
                    return False, f"missing slot {slot} at position {position}"
                continue
            mx.eval(a, b)
            if not arrays_equal(a, b, mx):
                return False, (
                    f"restore point {position} slot {slot} max_abs {max_abs(a, b, mx)}"
                )
    return True, "equal"


def clear_rollback(cache_list):
    def _clear(cache):
        children = getattr(cache, "caches", None)
        if children is not None:
            for child in children:
                _clear(child)
        elif hasattr(cache, "rollback_state"):
            cache.rollback_state = None

    for cache in cache_list:
        _clear(cache)


def rollback_like_engine(cache_list, n_to_drop: int, verify_size: int, trim_all):
    """Mirror ``mtp_generate_step._rollback_draft`` for a K>1 verify."""
    if n_to_drop == 0:
        clear_rollback(cache_list)
        return
    trim_caches = []
    restores = []
    for cache in cache_list:
        restore = getattr(cache, "restore_rollback", None)
        if callable(restore) and getattr(cache, "rollback_state", None) is not None:
            restores.append(restore)
        else:
            trim_caches.append(cache)
    if trim_caches and not trim_all(trim_caches, n_to_drop):
        raise RuntimeError("attention caches refused the speculative rollback")
    for restore in restores:
        restore(n_to_drop, verify_size)


def gdn_caches(cache_list, cache_type):
    return [cache for cache in cache_list if isinstance(cache, cache_type)]


def attention_offsets(cache_list, cache_type):
    offsets = []
    for cache in cache_list:
        if isinstance(cache, cache_type):
            continue
        children = getattr(cache, "caches", None) or [cache]
        offsets.append(tuple(int(getattr(child, "offset", -1)) for child in children))
    return offsets


def leaf_state_arrays(cache):
    """Every array a (possibly composite) cache exposes through ``state``."""
    children = getattr(cache, "caches", None)
    if children is not None:
        arrays = []
        for child in children:
            arrays.extend(leaf_state_arrays(child))
        return arrays
    state = getattr(cache, "state", None)
    items = state if isinstance(state, list | tuple) else [state]
    return [item for item in items if item is not None and hasattr(item, "shape")]


def attention_states_equal(stock_list, fused_list, cache_type, mx):
    """Bitwise lockstep of every non-GDN cache (KV plus QSA index state)."""
    for index, (a, b) in enumerate(zip(stock_list, fused_list, strict=True)):
        if isinstance(a, cache_type):
            continue
        left, right = leaf_state_arrays(a), leaf_state_arrays(b)
        if len(left) != len(right):
            return False, f"attention cache {index}: state arity"
        for position, (x, y) in enumerate(zip(left, right, strict=True)):
            if x.shape != y.shape or x.dtype != y.dtype or not arrays_equal(x, y, mx):
                return False, f"attention cache {index}: state array {position}"
    return True, "equal"


ZERO_CENTERED_GAIN_SUFFIXES = (
    ".hc_norm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def checkpoint_norm_convention(weights) -> str:
    """'raw' when conv1d weights still have the HF (C, 1, K) layout."""
    raw = any(
        "conv1d.weight" in key
        and getattr(value, "ndim", 0) == 3
        and value.shape[-1] != 1
        for key, value in weights.items()
    )
    return "raw" if raw else "converted"


def recenter_gains(weights, convention: str) -> int:
    """Subtract the one a converted checkpoint folded into zero-centered gains."""
    if convention != "converted":
        return 0
    touched = 0
    for key in list(weights):
        if key.endswith(ZERO_CENTERED_GAIN_SUFFIXES):
            weights[key] = weights[key] - 1.0
            touched += 1
    return touched


_NORM_PATCH_STATE: dict | None = None


def install_norm_convention_patch(requested: str) -> dict:
    """Wrap the vendored sanitizers so converted gains load correctly.

    Installing twice would wrap the wrapper and recenter twice, so the patch
    is a process-wide singleton; a second call returns the live state.
    """
    global _NORM_PATCH_STATE
    if _NORM_PATCH_STATE is not None:
        if _NORM_PATCH_STATE["requested"] != requested:
            raise RuntimeError("norm convention patch already installed differently")
        return _NORM_PATCH_STATE
    import vllm_mlx.models.qwen4_exp as vendored
    from vllm_mlx.spec_decode.mtp import qwen4_exp_inject

    state = {
        "requested": requested,
        "detected": None,
        "backbone_gains": 0,
        "mtp_gains": 0,
    }
    _NORM_PATCH_STATE = state
    original_sanitize = vendored.Model.sanitize
    original_mtp = qwen4_exp_inject._sanitize_mtp_weights

    def resolve(weights):
        if state["detected"] is None:
            state["detected"] = (
                checkpoint_norm_convention(weights)
                if requested == "auto"
                else requested
            )
        return state["detected"]

    def sanitize(self, weights):
        convention = resolve(weights)
        mapped = original_sanitize(self, weights)
        state["backbone_gains"] += recenter_gains(mapped, convention)
        return mapped

    def sanitize_mtp(raw):
        mapped = original_mtp(raw)
        state["mtp_gains"] += recenter_gains(mapped, resolve(raw))
        return mapped

    vendored.Model.sanitize = sanitize
    qwen4_exp_inject._sanitize_mtp_weights = sanitize_mtp
    return state


def clone_cache_list(cache_list, cache_type, mx):
    """Deep-copy a B=1 prompt cache so an oracle can look ahead on it."""
    from mlx_lm.models.cache import CacheList, KVCache

    def clone(cache):
        # Clones carry live state only: the oracle runs between rounds, when
        # every restore point has been consumed and nothing is staged.
        if getattr(cache, "rollback_state", None) is not None:
            raise RuntimeError("cannot clone a cache with live restore points")
        if getattr(cache, "_rollback_slots", None):
            raise RuntimeError("cannot clone a cache with staged snapshots")
        if isinstance(cache, cache_type):
            return cache.extract(0)
        children = getattr(cache, "caches", None)
        if children is not None:
            return CacheList(*[clone(child) for child in children])
        if hasattr(cache, "compress_ratio"):  # QSAIndexCache
            new = type(cache)(cache.compress_ratio)
            new.meta_state = cache.meta_state
            new.state = tuple(None if a is None else mx.array(a) for a in cache.state)
            return new
        if isinstance(cache, KVCache):
            new = KVCache()
            if getattr(cache, "keys", None) is not None:
                new.state = tuple(mx.array(a) for a in cache.state)
            return new
        raise TypeError(f"cannot clone {type(cache).__name__}")

    return [clone(cache) for cache in cache_list]


def oracle_next_tokens(model, caches, y, k, mx, cache_type, trim_all):
    """The ``k`` greedy tokens after ``y`` under the verify-width forward.

    Each step verifies ``[t, t, ..., t]`` on a cloned cache and keeps only
    position 0, so every prediction is made by the same width-``k+1``
    rollback-recording forward the real round uses (single-token decode takes
    a different MoE path and does not reproduce it bit for bit).
    """
    clone = clone_cache_list(caches, cache_type, mx)
    tokens = [int(y)]
    for _ in range(k):
        block = mx.array([[tokens[-1]] * (k + 1)], mx.uint32)
        logits = model(block, cache=clone, n_confirmed=k)
        tokens.append(int(mx.argmax(logits[0, 0]).item()))
        rollback_like_engine(clone, k, k + 1, trim_all)
    return tokens[1:]


def scripted_drafts(next_tokens, k, round_index, vocab):
    """Drafts that force full, partial and zero acceptance in turn.

    ``next_tokens`` are the true ``k`` continuations of the committed token.
    Round ``r`` keeps the first ``k - r % (k+1)`` drafts true and corrupts the
    next one, cycling through every acceptance count from ``k`` down to ``0``.
    """
    correct = k - (round_index % (k + 1))
    drafts = [int(token) for token in next_tokens[:k]]
    if correct < k:
        drafts[correct] = (drafts[correct] + 1) % vocab
    return drafts, correct


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_layer(args, layer, mx, cache_type, stats_fn, set_verify_mode):
    steps = args.draft_k + 1
    key = mx.random.key(2105)
    hidden = mx.random.normal(
        (max(args.layer_blocks, args.layer_timing_blocks), 1, steps, layer.hidden_size),
        key=key,
    ).astype(layer.dt_bias.dtype)
    warm = cache_type(2)
    set_verify_mode(layer, "stock")
    layer.set_fused_gdn_decode_mode("stock")
    warm_output = layer(hidden[0][:, :1], cache=warm)
    mx.eval(warm_output, *warm.cache)
    stock_cache = clone_state_cache(warm, cache_type, mx)
    fused_cache = clone_state_cache(warm, cache_type, mx)

    mismatch = None
    before = stats_fn(layer)
    restores = []
    for block in range(args.layer_blocks):
        set_verify_mode(layer, "stock")
        stock = layer(hidden[block], cache=stock_cache, record_rollback=True)
        set_verify_mode(layer, "fused")
        fused = layer(hidden[block], cache=fused_cache, record_rollback=True)
        mx.eval(stock, fused, *stock_cache.cache, *fused_cache.cache)
        output_equal = arrays_equal(stock, fused, mx)
        slots_equal = all(
            arrays_equal(a, b, mx)
            for a, b in zip(stock_cache.cache, fused_cache.cache, strict=True)
        )
        points_equal, detail = restore_points_equal(stock_cache, fused_cache, mx)
        if not (output_equal and slots_equal and points_equal):
            mismatch = {
                "block": block,
                "output_equal": output_equal,
                "slots_equal": slots_equal,
                "restore_points_equal": points_equal,
                "restore_detail": detail,
                "max_output_abs": max_abs(stock, fused, mx),
            }
            break
        n_to_drop = block % steps
        restores.append(n_to_drop)
        if n_to_drop == 0:
            stock_cache.rollback_state = None
            fused_cache.rollback_state = None
        else:
            stock_cache.restore_rollback(n_to_drop, steps)
            fused_cache.restore_rollback(n_to_drop, steps)
    after = stats_fn(layer)
    verify_calls = after["verify_calls"] - before["verify_calls"]
    fallbacks = after["verify_fallbacks"] - before["verify_fallbacks"]
    correctness = {
        "passed": mismatch is None
        and args.layer_blocks > 0
        and verify_calls == args.layer_blocks
        and fallbacks == 0
        and set(restores) == set(range(steps)),
        "blocks": args.layer_blocks,
        "verify_width": steps,
        "mismatch": mismatch,
        "verify_calls": verify_calls,
        "verify_fallbacks": fallbacks,
        "restores": restores,
        "last_fallbacks": after["verify_last_fallbacks"],
    }
    if not correctness["passed"]:
        return {"correctness": correctness, "timing": None}

    timings = {"stock": [], "fused": []}
    for repeat in range(args.layer_repeats):
        order = ("stock", "fused") if repeat % 2 == 0 else ("fused", "stock")
        for mode in order:
            cache = clone_state_cache(warm, cache_type, mx)
            mx.eval(*cache.cache)
            set_verify_mode(layer, mode)
            started = time.perf_counter()
            output = None
            for block in range(args.layer_timing_blocks):
                cache.rollback_state = None
                output = layer(
                    hidden[block % len(hidden)], cache=cache, record_rollback=True
                )
            mx.eval(output, *cache.cache)
            timings[mode].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    timing = {
        "raw_seconds": timings,
        "median_seconds": medians,
        "median_speedup_percent": 100.0 * (medians["stock"] / medians["fused"] - 1.0),
        "blocks_per_observation": args.layer_timing_blocks,
    }
    return {"correctness": correctness, "timing": timing}


def phase_rounds(
    args,
    model,
    prompt,
    mx,
    cache_type,
    stats_fn,
    set_verify_mode,
    trim_all,
    layer_type,
):
    steps = args.draft_k + 1
    fused_layers = set_verify_mode(model, "stock")
    caches = {"stock": model.make_cache(), "fused": model.make_cache()}
    prompt_ids = prompt[None]
    logits = {}
    for mode, cache in caches.items():
        logits[mode] = model(prompt_ids, cache=cache)
        mx.eval(logits[mode], *(c.state for c in cache))
    if not arrays_equal(logits["stock"], logits["fused"], mx):
        raise RuntimeError("stock prefill is not reproducible across cache copies")
    # A plain greedy continuation supplies drafts that deliberately hit every
    # acceptance count; if the verify-width forward ever disagrees with the
    # single-token continuation (bf16 batch-shape numerics), the schedule
    # falls back to prediction-derived drafts and records where.
    vocab = int(logits["stock"].shape[-1])
    y = int(mx.argmax(logits["stock"][0, -1]).item())
    schedule_misses = 0
    accepted_hist = {str(n): 0 for n in range(args.draft_k + 1)}
    # Optional repro capture: remember the stock arm's exact verify inputs
    # (layer input, conv slot, recurrent slot) per GDN layer for the current
    # round, so a mismatch can be replayed on one lazily loaded layer.
    gdn_layers = [m for _, m in model.named_modules() if isinstance(m, layer_type)]
    layer_index = {id(layer): index for index, layer in enumerate(gdn_layers)}
    captures: dict[int, dict] = {}
    original_call = layer_type.__call__
    if args.dump_dir is not None:

        def capturing_call(self, inputs, mask=None, cache=None, **kwargs):
            if (
                kwargs.get("record_rollback")
                and inputs.shape[1] > 1
                and self.fused_gdn_verify_mode == "stock"
                and cache is not None
            ):
                captures[layer_index[id(self)]] = {
                    "inputs": inputs,
                    "conv_state": cache[0],
                    "recurrent_state": cache[1],
                }
            return original_call(self, inputs, mask, cache, **kwargs)

        layer_type.__call__ = capturing_call
    before = stats_fn(model)
    mismatch = None
    rounds_done = 0
    try:
        for round_index in range(args.verify_rounds):
            set_verify_mode(model, "stock")
            next_tokens = oracle_next_tokens(
                model, caches["stock"], y, args.draft_k, mx, cache_type, trim_all
            )
            drafts, expected_accept = scripted_drafts(
                next_tokens, args.draft_k, round_index, vocab
            )
            tokens = mx.array([[y, *drafts]], mx.uint32)
            outputs = {}
            for mode, cache in caches.items():
                set_verify_mode(model, mode)
                outputs[mode] = model(tokens, cache=cache, n_confirmed=args.draft_k)
            mx.eval(*outputs.values())
            mx.eval(*(c.state for cache in caches.values() for c in cache))
            logits_equal = arrays_equal(outputs["stock"], outputs["fused"], mx)
            stock_gdn = gdn_caches(caches["stock"], cache_type)
            fused_gdn = gdn_caches(caches["fused"], cache_type)
            slot_detail = None
            failing_index = None
            for index, (a, b) in enumerate(zip(stock_gdn, fused_gdn, strict=True)):
                for slot, (x, w) in enumerate(zip(a.cache, b.cache, strict=True)):
                    if x is None and w is None:
                        continue
                    if x is None or w is None or not arrays_equal(x, w, mx):
                        slot_detail = f"gdn cache {index} slot {slot}"
                        break
                if slot_detail is None:
                    ok, detail = restore_points_equal(a, b, mx)
                    if not ok:
                        slot_detail = f"gdn cache {index}: {detail}"
                if slot_detail is not None:
                    failing_index = index
                    break
            offsets_equal = attention_offsets(caches["stock"], cache_type) == (
                attention_offsets(caches["fused"], cache_type)
            )
            states_equal, state_detail = attention_states_equal(
                caches["stock"], caches["fused"], cache_type, mx
            )
            if (
                not logits_equal
                or slot_detail is not None
                or not offsets_equal
                or not states_equal
            ):
                mismatch = {
                    "round": round_index,
                    "logits_equal": logits_equal,
                    "max_logit_abs": max_abs(outputs["stock"], outputs["fused"], mx),
                    "gdn_detail": slot_detail,
                    "attention_offsets_equal": offsets_equal,
                    "attention_states_equal": states_equal,
                    "attention_detail": state_detail,
                }
                if args.dump_dir is not None and captures:
                    failing = (
                        failing_index if failing_index is not None else max(captures)
                    )
                    args.dump_dir.mkdir(parents=True, exist_ok=True)
                    dump = (
                        args.dump_dir / f"round{round_index}-gdn{failing}.safetensors"
                    )
                    mx.save_safetensors(
                        str(dump),
                        {k: mx.array(v) for k, v in captures[failing].items()},
                    )
                    mismatch["dump"] = str(dump)
                    mismatch["dump_layer_index"] = failing
                break
            verify_argmax = mx.argmax(outputs["stock"][0], axis=-1).tolist()
            accepted = greedy_accept_count(verify_argmax, drafts)
            bonus = int(verify_argmax[accepted])
            accepted_hist[str(accepted)] += 1
            n_to_drop = args.draft_k - accepted
            for cache in caches.values():
                rollback_like_engine(cache, n_to_drop, steps, trim_all)
            rounds_done += 1
            schedule_misses += int(accepted != expected_accept)
            y = bonus
    finally:
        layer_type.__call__ = original_call
    coverage = all(accepted_hist[str(n)] > 0 for n in range(args.draft_k + 1))
    after = stats_fn(model)
    verify_calls = after["verify_calls"] - before["verify_calls"]
    fallbacks = after["verify_fallbacks"] - before["verify_fallbacks"]
    expected_calls, expected_fallbacks = expected_verify_calls(
        "fused", fused_layers, rounds_done + (1 if mismatch is not None else 0)
    )
    return {
        "correctness": {
            "passed": mismatch is None
            and rounds_done == args.verify_rounds
            and verify_calls == expected_calls
            and fallbacks == expected_fallbacks
            and coverage,
            "rounds": rounds_done,
            "verify_width": steps,
            "prompt_tokens": int(prompt.size),
            "mismatch": mismatch,
            "accepted_histogram": accepted_hist,
            "acceptance_coverage": coverage,
            "schedule_misses": schedule_misses,
            "verify_calls": verify_calls,
            "expected_verify_calls": expected_calls,
            "verify_fallbacks": fallbacks,
            "fused_layers": fused_layers,
            "last_fallbacks": after["verify_last_fallbacks"],
        }
    }


def phase_e2e(args, model, tokenizer, prompt, mx, stats_fn, set_verify_mode, generate):
    eos = getattr(tokenizer, "eos_token_id", None)
    # The vendored generator clamps the draft depth to what the injected head
    # advertises (the scheduler does the same), so the end-to-end cell runs at
    # the engine's real width rather than the requested one.
    # Rapid's Qwen4 injection installs the MTP surfaces on the inner text
    # model and the scheduler hands that inner model to the generator.
    model = getattr(model, "language_model", model)
    model_cap = getattr(model, "mtp_max_speculative_tokens", None)
    draft_k = args.draft_k if model_cap is None else min(args.draft_k, int(model_cap))
    if args.e2e_repeats % 2:
        raise ValueError("--e2e-repeats must be even so both arm orders balance")
    observations = []
    orders = (("stock", "fused"), ("fused", "stock"))
    for repeat in range(args.e2e_repeats):
        for mode in orders[repeat % len(orders)]:
            fused_layers = set_verify_mode(model, mode)
            before = stats_fn(model)
            timing_stats: dict[str, float] = {}
            tokens = []
            drafted = 0
            started = time.perf_counter()
            first_token_at = None
            for token, _, from_draft in generate(
                prompt,
                model,
                max_tokens=args.max_tokens,
                temp=0.0,
                max_k=draft_k,
                disable_auto_k=True,
                prompt_lookup_enabled=False,
                timing_stats=timing_stats,
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                tokens.append(int(token))
                drafted += int(bool(from_draft))
                if eos is not None and int(token) == int(eos):
                    break
            ended = time.perf_counter()
            after = stats_fn(model)
            if len(tokens) < 2 or first_token_at is None:
                raise RuntimeError(f"generation produced {len(tokens)} tokens")
            decode_seconds = ended - first_token_at
            rounds = int(timing_stats.get("verify_calls", 0))
            verify_calls = after["verify_calls"] - before["verify_calls"]
            fallbacks = after["verify_fallbacks"] - before["verify_fallbacks"]
            expected_calls, expected_fallbacks = expected_verify_calls(
                mode, fused_layers, rounds
            )
            observations.append(
                {
                    "mode": mode,
                    "repeat": repeat + 1,
                    "tokens": len(tokens),
                    "draft_accepted_tokens": drafted,
                    "verify_rounds": rounds,
                    "token_sha256": hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "ttft_seconds": first_token_at - started,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds,
                    "verify_calls": verify_calls,
                    "expected_verify_calls": expected_calls,
                    "verify_fallbacks": fallbacks,
                    "expected_verify_fallbacks": expected_fallbacks,
                    "path_counts_exact": verify_calls == expected_calls
                    and fallbacks == expected_fallbacks
                    and rounds > 0,
                    "last_fallbacks": after["verify_last_fallbacks"],
                }
            )
            mx.clear_cache()
    hashes = {item["token_sha256"] for item in observations}
    path_counts_exact = all(item["path_counts_exact"] for item in observations)
    medians = {
        mode: statistics.median(
            item["decode_tokens_per_second"]
            for item in observations
            if item["mode"] == mode
        )
        for mode in ("stock", "fused")
    }
    return {
        "correctness": {
            "passed": len(hashes) == 1 and path_counts_exact,
            "token_exact": len(hashes) == 1,
            "path_counts_exact": path_counts_exact,
            "hashes": sorted(hashes),
        },
        "draft_k_requested": args.draft_k,
        "draft_k_used": draft_k,
        "model_draft_k_cap": model_cap,
        "verify_width": draft_k + 1,
        "observations": observations,
        "median_decode_tokens_per_second": medians,
        "median_speedup_percent": 100.0 * (medians["fused"] / medians["stock"] - 1.0),
    }


def run(args, partial: dict | None = None):
    """Run the gate; ``partial`` receives phase results as they complete."""
    if partial is None:
        partial = {}
    if args.model is None or not args.model.is_dir():
        raise SystemExit("--model must name an existing local checkpoint")
    if args.draft_k < 1:
        raise SystemExit("--draft-k must be at least 1")

    for name in (
        "layer_blocks",
        "layer_timing_blocks",
        "layer_repeats",
        "verify_rounds",
        "e2e_repeats",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    if args.layer_blocks < args.draft_k + 1:
        raise SystemExit("--layer-blocks must cover every restore boundary")
    lock = acquire_gate_lock()
    receipts = {"start": host_receipt()}
    # The start receipt is its own baseline; a tenant or an unreadable scan
    # refuses the run here, and the same check repeats after every phase.
    check_host(receipts["start"], receipts["start"], args, "start")
    violations = []

    def take_receipt(name, mx_module=None):
        receipts[name] = host_receipt(mx_module)
        violation = host_violation(receipts[name], receipts["start"], args, name)
        if violation is not None:
            violations.append(violation)
        return violation

    import mlx.core as mx
    from mlx_lm.utils import load, load_config

    from vllm_mlx.cache_rollback import trim_all
    from vllm_mlx.models.qwen4_exp import (
        GatedDeltaNet,
        Qwen4ExpStateCache,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
        set_qwen4_fused_gdn_verify_mode,
    )
    from vllm_mlx.spec_decode.mtp.dispatch import dispatch_mtp_inject
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    mx.set_default_device(mx.gpu)
    _register_vendored_archs()
    if args.force_vendored_arch:
        # The registry prefers a native ``mlx_lm.models.qwen4_exp`` when the
        # installed mlx-lm ships one; this gate measures Rapid's vendored
        # layers, so pin the vendored module regardless of what is installed.
        import sys

        import vllm_mlx.models.qwen4_exp as vendored_qwen4_exp

        sys.modules["mlx_lm.models.qwen4_exp"] = vendored_qwen4_exp
    # Rapid's ZeroCenteredRMSNorm adds one to the stored gain, i.e. it expects
    # the raw checkpoint deltas. Checkpoints written by the native mlx-lm
    # converter already store the full gains ("converted"), so loading them
    # into the vendored modules doubles the +1 and the model emits gibberish.
    # Detect the convention the way the native loader does (a converted
    # checkpoint has its conv1d weights already moved to (C, K, 1)) and
    # subtract the one back out for every zero-centered gain, backbone and
    # MTP head alike.
    norm_state = install_norm_convention_patch(args.norm_convention)
    # The vendored module names PLE n-gram tables ``ngram_embedding.shards.N``
    # while checkpoints converted by the native mlx-lm family name them
    # ``ngram_embedding.shard_N``. ``sanitize`` maps the tensors, but the
    # per-path quantization overrides (group 32 for the 160-wide tables) are
    # looked up by module path, so translate those keys too.
    quantization = load_config(args.model.resolve()).get("quantization")
    model_config = {}
    translated_keys = 0
    if isinstance(quantization, dict):
        translated = {}
        for key, value in quantization.items():
            new_key = re.sub(
                r"ngram_embedding\.shard_(\d+)$", r"ngram_embedding.shards.\1", key
            )
            translated_keys += new_key != key
            translated[new_key] = value
        model_config["quantization"] = translated
    model, tokenizer = load(
        str(args.model.resolve()), model_config=model_config, lazy=args.layer_only
    )
    model.eval()
    set_qwen4_fused_gdn_mode(model, "stock")
    fused_layers = set_qwen4_fused_gdn_verify_mode(model, "stock")
    if not fused_layers:
        raise SystemExit(
            "checkpoint has no Rapid Qwen4 GDN layers (a native mlx_lm "
            "qwen4_exp may have been resolved; pass --force-vendored-arch)"
        )
    layer = next(m for _, m in model.named_modules() if isinstance(m, GatedDeltaNet))
    if args.layer_only:
        # Materialize one production GDN layer only; the layer gate needs
        # nothing else resident. The exclusive-host checks above still apply.
        mx.eval(layer.parameters())
    violation = take_receipt("after_load", mx)
    if violation is not None:
        raise HostAbortError(violation)

    import mlx_lm

    result = {
        "plan": PLAN,
        "fused_layers": fused_layers,
        "draft_k": args.draft_k,
        "model_module": type(model).__module__,
        "quantization_keys_translated": translated_keys,
        "norm_convention": norm_state,
        "checkpoint": checkpoint_fingerprint(args.model.resolve()),
        "arguments": {
            name: (str(value) if isinstance(value, Path) else value)
            for name, value in vars(args).items()
        },
        "layer_only": bool(args.layer_only),
        "mlx_version": mx.__version__,
        "mlx_lm_version": getattr(mlx_lm, "__version__", None),
        "mlx_lm_path": str(Path(mlx_lm.__file__).parent),
        "phases": {},
        "host": receipts,
    }
    partial.update(result)
    result["phases"]["layer"] = phase_layer(
        args,
        layer,
        mx,
        Qwen4ExpStateCache,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_verify_mode,
    )
    take_receipt("after_layer", mx)
    passed = result["phases"]["layer"]["correctness"]["passed"]

    if args.chat_template:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array(list(prompt_ids), mx.uint32)
    result["prompt_tokens"] = int(prompt.size)
    result["chat_template"] = bool(args.chat_template)
    if args.layer_only:
        args.skip_rounds = True
        args.skip_e2e = True
    if passed and not args.skip_rounds:
        check_host(receipts["after_layer"], receipts["start"], args, "rounds")
        result["phases"]["rounds"] = phase_rounds(
            args,
            model,
            prompt,
            mx,
            Qwen4ExpStateCache,
            qwen4_fused_gdn_stats,
            set_qwen4_fused_gdn_verify_mode,
            trim_all,
            GatedDeltaNet,
        )
        take_receipt("after_rounds", mx)
        passed = result["phases"]["rounds"]["correctness"]["passed"]
        mx.clear_cache()

    if passed and not args.skip_e2e:
        check_host(take_latest(receipts), receipts["start"], args, "e2e")
        if not dispatch_mtp_inject(
            model, "qwen4_exp", mtp_sidecar=args.model.resolve()
        ):
            result["phases"]["e2e"] = {
                "correctness": {"passed": False, "reason": "MTP inject refused"}
            }
            passed = False
        else:
            result["phases"]["e2e"] = phase_e2e(
                args,
                model,
                tokenizer,
                prompt,
                mx,
                qwen4_fused_gdn_stats,
                set_qwen4_fused_gdn_verify_mode,
                mtp_generate_step,
            )
            take_receipt("after_e2e", mx)
            passed = result["phases"]["e2e"]["correctness"]["passed"]

    take_receipt("end", mx)
    result["host_violations"] = violations
    result.update(completion_status(result["phases"], passed and not violations))
    lock.close()
    return result


REQUIRED_PHASES = ("layer", "rounds", "e2e")


def completion_status(phases: dict, executed_passed: bool) -> dict:
    """``passed`` is reserved for a complete run of every required phase.

    A ``--layer-only`` or ``--skip-*`` run reports ``partial_passed`` and
    ``complete=False`` instead, so a receipt can never be mistaken for the
    full qualification.
    """
    executed = [name for name in REQUIRED_PHASES if name in phases]
    complete = len(executed) == len(REQUIRED_PHASES)
    return {
        "phases_executed": executed,
        "complete": complete,
        "partial_passed": bool(executed_passed),
        "passed": bool(executed_passed and complete),
    }


def checkpoint_fingerprint(model_dir: Path) -> dict:
    """Identify the checkpoint the receipt was produced from."""
    fingerprint = {"path": str(model_dir)}
    for name in ("config.json", "model.safetensors.index.json"):
        target = model_dir / name
        if target.is_file():
            fingerprint[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return fingerprint


def take_latest(receipts):
    return receipts[max(receipts, key=lambda name: receipts[name]["timestamp"])]


def main() -> int:
    args = parse_args()
    if not args.execute_metal:
        print(format_bench_json({"plan_only": True, "plan": PLAN}, __file__))
        return 0
    partial: dict = {}
    try:
        result = run(args, partial)
    except HostAbortError as exc:
        result = {**partial, "plan": PLAN, "passed": False, "aborted": str(exc)}
    except Exception as exc:  # noqa: BLE001 - keep the receipts of a crashed run
        import traceback

        result = {
            **partial,
            "plan": PLAN,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
    payload = format_bench_json(result, __file__, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    # A complete run must pass every required phase; an explicitly partial
    # run (--layer-only, --skip-*) exits by its own phases but its receipt
    # never carries ``passed``.
    if result.get("complete"):
        return 0 if result.get("passed") else 1
    return 0 if result.get("partial_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
