"""Model-free guard for the maskless batched-GQA hazard in MLX #4431.

The affected MLX vector kernel read batch zero's K/V for later batch rows when
all of the following were true: batched decode, GQA, and no attention mask.
Rapid's Qwen4 route is safe only while every batched KV cache supplies a mask
and the model forwards that mask to ``scaled_dot_product_attention``.

These tests inspect Python ASTs instead of constructing MLX arrays, so the
contract runs on CPU-only/headless CI and remains independent of a Metal
device.  It is intentionally a dispatch-contract test, not a numerical kernel
test; MLX's own regression covers the kernel once the dependency is rebased.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QWEN4_SOURCE = ROOT / "vllm_mlx/models/qwen4_exp.py"
RAPID_QUANTIZED_CACHE_SOURCE = ROOT / "vllm_mlx/quantized_batch_cache.py"


def _mlx_lm_cache_source() -> Path:
    spec = importlib.util.find_spec("mlx_lm")
    assert spec is not None and spec.submodule_search_locations, (
        "mlx-lm is required to verify the pinned batch-cache mask contract"
    )
    source = Path(next(iter(spec.submodule_search_locations))) / "models/cache.py"
    assert source.is_file(), f"mlx-lm cache source not found at {source}"
    return source


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    match = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name),
        None,
    )
    assert match is not None, f"missing required cache class {name}"
    return match


def _method(owner: ast.ClassDef, name: str) -> ast.FunctionDef:
    match = next(
        (node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    assert match is not None, f"{owner.name} must define {name}"
    return match


def _assert_make_mask_never_returns_none(source: Path, class_names: tuple[str, ...]) -> None:
    tree = ast.parse(source.read_text(), filename=str(source))
    for class_name in class_names:
        method = _method(_class(tree, class_name), "make_mask")
        returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
        assert returns, f"{class_name}.make_mask must return an attention mask"
        assert all(
            node.value is not None
            and not (isinstance(node.value, ast.Constant) and node.value.value is None)
            for node in returns
        ), f"{class_name}.make_mask permits the MLX #4431 maskless batch route"


def test_all_mlx_lm_batched_kv_caches_always_return_a_mask() -> None:
    _assert_make_mask_never_returns_none(
        _mlx_lm_cache_source(),
        (
            "BatchKVCache",
            "BatchQuantizedKVCache",
            "BatchRotatingKVCache",
            "BatchRotatingQuantizedKVCache",
        ),
    )


def test_rapid_quantized_batch_cache_always_returns_a_mask() -> None:
    _assert_make_mask_never_returns_none(
        RAPID_QUANTIZED_CACHE_SOURCE,
        ("QuantizedBatchKVCache",),
    )


def test_qwen4_attention_builds_and_forwards_the_cache_mask() -> None:
    tree = ast.parse(QWEN4_SOURCE.read_text(), filename=str(QWEN4_SOURCE))
    attention = _class(tree, "QSAAttention")
    call = _method(attention, "__call__")

    creates_mask = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_attention_mask"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "kv_cache"
        for node in ast.walk(call)
    )
    assert creates_mask, "Qwen4 attention must derive its mask from the batch KV cache"

    sdpa_calls = [
        node
        for node in ast.walk(call)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "scaled_dot_product_attention"
    ]
    assert sdpa_calls, "Qwen4 attention must retain the audited SDPA wrapper route"
    for sdpa_call in sdpa_calls:
        mask = next((kw.value for kw in sdpa_call.keywords if kw.arg == "mask"), None)
        assert isinstance(mask, ast.Name) and mask.id == "additive_mask", (
            "Qwen4 SDPA must forward the cache-derived mask; a missing/None mask can "
            "re-expose MLX #4431 for B>1 GQA decode"
        )
