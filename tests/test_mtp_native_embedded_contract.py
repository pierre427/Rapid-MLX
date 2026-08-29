"""CPU-only contracts for mlx-lm's native embedded Qwen3.5 MTP head."""

from __future__ import annotations

from vllm_mlx.spec_decode.mtp.dispatch import (
    dispatch_mtp_inject,
    dispatch_mtp_validate,
)
from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
    inject_mtp_support,
    validate_mtp_support,
)


class _Args:
    mtp_num_hidden_layers = 1
    hidden_size = 8
    tie_word_embeddings = False


class _NativeMTP:
    def __init__(self, layer_count: int = 1):
        # Structural mlx-lm MTP module contract. The sentinels deliberately
        # are not arrays: these tests prove routing/ownership without loading
        # weights or touching the GPU.
        self.fc = object()
        self.pre_fc_norm_embedding = object()
        self.pre_fc_norm_hidden = object()
        self.layers = [object() for _ in range(layer_count)]
        self.norm = object()


class _NativeInner:
    args = _Args()
    model = object()

    def __init__(self, layer_count: int = 1):
        self.mtp = _NativeMTP(layer_count)
        self.calls = []

    def mtp_step(self, hidden, tokens, cache):
        self.calls.append((hidden, tokens, cache))
        return "native-logits", "native-post-norm-hidden"

    def make_mtp_cache(self):
        return ["native-cache"]


def _disable_process_global_cache_patches(monkeypatch):
    from vllm_mlx.spec_decode.mtp import cache_patch

    monkeypatch.setattr(cache_patch, "patch_arrays_cache_rollback_state", lambda: None)
    monkeypatch.setattr(cache_patch, "patch_gated_delta_net_for_mtp", lambda: None)


def test_adopts_native_embedded_mtp_without_rebuilding(monkeypatch):
    _disable_process_global_cache_patches(monkeypatch)
    model = _NativeInner()
    loaded_native_mtp = model.mtp

    assert inject_mtp_support(model) is True
    assert validate_mtp_support(model) is True
    assert model.mtp is loaded_native_mtp

    result = model.mtp_forward("hidden", "tokens", "cache", return_hidden=True)
    assert result == ("native-logits", "native-post-norm-hidden")
    assert model.calls == [("hidden", "tokens", "cache")]


def test_native_embedded_mtp_logits_only_adapter(monkeypatch):
    _disable_process_global_cache_patches(monkeypatch)
    model = _NativeInner()

    assert inject_mtp_support(model) is True
    assert model.mtp_forward("hidden", "tokens", "cache") == "native-logits"
    # The fused-greedy shortcut must not bypass native mlx-lm's mtp_step.
    assert model.mtp_greedy("hidden", "tokens", "cache") is None


def test_dispatch_reaches_native_embedded_contract_without_sidecar(monkeypatch):
    _disable_process_global_cache_patches(monkeypatch)
    model = _NativeInner()

    assert dispatch_mtp_inject(model, "qwen3_5") is True
    assert dispatch_mtp_validate(model, "qwen3_5") is True


def test_rejects_embedded_mtp_without_native_step():
    model = _NativeInner()
    model.mtp_step = None
    original_class = type(model)

    assert inject_mtp_support(model) is False
    assert type(model) is original_class
    assert validate_mtp_support(model) is False


def test_rejects_embedded_mtp_layer_count_mismatch():
    model = _NativeInner(layer_count=2)
    original_class = type(model)

    assert inject_mtp_support(model) is False
    assert type(model) is original_class
    assert validate_mtp_support(model) is False


def test_rejects_declared_mtp_when_embedded_module_is_missing():
    model = _NativeInner()
    del model.mtp
    original_class = type(model)

    assert inject_mtp_support(model) is False
    assert type(model) is original_class
    assert validate_mtp_support(model) is False
