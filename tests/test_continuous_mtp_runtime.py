"""Pure-Python contracts for continuous self-MTP runtime assembly."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.spec_decode.mtp import continuous_runtime as runtime_module
from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPUnsupportedError,
)
from vllm_mlx.spec_decode.mtp.ragged_cache import (
    preflight_ragged_cache,
    trim_ragged_cache,
)


def _descriptor(family: str = "qwen3_5", **changes):
    values = {
        "protocol_version": 1,
        "model_family": family,
        "batch_forward": "mtp_batch_forward",
        "recursive_draft_depth": 2,
        "share_qsa_indices": family == "qwen4_exp",
        "fixed_membership": True,
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
        "dynamic_join": family == "qwen3_5",
        "flash_dynamic_membership_attested": False,
        "quantized_cache": True,
        "windowed_cache": False,
        "xtc": False,
    }
    values.update(changes)
    return values


class _InjectedTextModel:
    model_type = "qwen3_5_text"

    def __init__(self, descriptor=None):
        self.args = SimpleNamespace(hidden_size=8)
        self.model = SimpleNamespace(layers=[object()])
        self.batched_mtp_capability = descriptor or _descriptor()
        self.calls = []

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden=False,
        n_confirmed=0,
    ):
        self.calls.append(
            (
                "target",
                inputs,
                cache,
                input_embeddings,
                return_hidden,
                n_confirmed,
            )
        )
        return "target-logits", "target-hidden"

    def mtp_batch_forward(self, hidden, token_ids, mtp_cache):
        self.calls.append(("draft", hidden, token_ids, mtp_cache))
        return "draft-logits", "post-hidden"

    def make_mtp_cache(self):
        self.calls.append(("make-mtp-cache",))
        return ["draft-cache"]


class _OuterModel:
    def __init__(self, inner):
        self.language_model = inner
        self.batched_mtp_capability = inner.batched_mtp_capability


class _NoConfirmedTarget(_InjectedTextModel):
    def __call__(self, inputs, cache=None, return_hidden=False):
        return inputs, cache, return_hidden


class _ArrayOpsStub:
    pass


def test_prompt_cache_factory_quantizes_immediately_via_dependency(monkeypatch):
    cache_module = ModuleType("mlx_lm.models.cache")
    generate_module = ModuleType("mlx_lm.generate")
    prompt_cache = [object(), object()]
    calls = []

    cache_module.make_prompt_cache = lambda model: (
        calls.append(("make", model)) or prompt_cache
    )
    generate_module.maybe_quantize_kv_cache = lambda cache, start, group, bits: (
        calls.append(("quantize", cache, start, group, bits))
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)

    assert runtime_module._make_prompt_cache("target", (32, 8)) is prompt_cache
    assert calls == [
        ("make", "target"),
        ("quantize", prompt_cache, 0, 32, 8),
    ]


@pytest.mark.requires_mlx
@pytest.mark.parametrize("bits", (4, 8))
def test_qwen4_prompt_cache_factory_quantizes_nested_qsa_kv_only(bits):
    from mlx_lm.models.cache import CacheList, KVCache, QuantizedKVCache

    from vllm_mlx.models.qwen4_exp_cache import QSAIndexCache

    class _Qwen4CacheOwner:
        def make_cache(self):
            return [CacheList(KVCache(), QSAIndexCache(compress_ratio=4))]

    prompt_cache = runtime_module._make_prompt_cache(
        _Qwen4CacheOwner(), (32, bits)
    )

    assert isinstance(prompt_cache[0], CacheList)
    assert isinstance(prompt_cache[0].caches[0], QuantizedKVCache)
    assert prompt_cache[0].caches[0].group_size == 32
    assert prompt_cache[0].caches[0].bits == bits
    # The QSA selection ledger is persistent model state, not attention K/V;
    # it keeps its concrete owner and native precision.
    assert isinstance(prompt_cache[0].caches[1], QSAIndexCache)


def test_assembler_resolves_inner_model_and_wires_forward_and_cache_seams(
    monkeypatch,
):
    inner = _InjectedTextModel()
    outer = _OuterModel(inner)
    target_cache_calls = []
    monkeypatch.setattr(
        runtime_module,
        "_make_prompt_cache",
        lambda model, quantization=None: (
            target_cache_calls.append((model, quantization)) or ["target-cache"]
        ),
    )

    runtime = runtime_module.assemble_continuous_self_mtp_runtime(
        outer,
        array_ops=_ArrayOpsStub(),
        prefill_step_size=17,
    )

    assert runtime.config.enabled is True
    assert runtime.config.architecture == "qwen3_5"
    assert runtime.config.allow_dynamic_membership is False
    assert runtime.capabilities.missing_fixed_core() == ()
    assert runtime.capabilities.dynamic_membership is False
    assert runtime.compute.prefill_step_size == 17
    assert runtime.compute.share_qsa_indices is False

    assert runtime.forwards.target("ids", "target-kv", n_confirmed=2) == (
        "target-logits",
        "target-hidden",
    )
    assert inner.calls[-1] == (
        "target",
        "ids",
        "target-kv",
        None,
        True,
        2,
    )
    assert runtime.forwards.draft("hidden", "tokens", "draft-kv") == (
        "draft-logits",
        "post-hidden",
    )
    assert inner.calls[-1] == ("draft", "hidden", "tokens", "draft-kv")

    assert runtime.compute.target_cache_factory() == ["target-cache"]
    assert target_cache_calls == [(inner, None)]
    assert runtime.compute.draft_cache_factory() == ["draft-cache"]
    assert runtime.caches._preflight is preflight_ragged_cache
    assert runtime.caches._trim is trim_ragged_cache
    # The opt-in speculation-rollback protocol defaults off.
    assert runtime.compute.speculation_rollback is False


def test_speculation_rollback_is_opt_in_and_reaches_the_backend(monkeypatch):
    inner = _InjectedTextModel()
    outer = _OuterModel(inner)
    monkeypatch.setattr(
        runtime_module,
        "_make_prompt_cache",
        lambda model, quantization=None: ["target-cache"],
    )
    default = runtime_module.assemble_continuous_self_mtp_runtime(outer)
    assert default.compute.speculation_rollback is False
    enabled = runtime_module.assemble_continuous_self_mtp_runtime(
        outer, speculation_rollback=True
    )
    assert enabled.compute.speculation_rollback is True


def test_effective_quantized_kv_settings_reach_target_cache_factory(monkeypatch):
    inner = _InjectedTextModel()
    calls = []
    abi_checks = []
    monkeypatch.setattr(
        runtime_module,
        "_require_quantized_cache_abi",
        lambda: abi_checks.append(True),
    )
    monkeypatch.setattr(
        runtime_module,
        "_make_prompt_cache",
        lambda model, quantization=None: (
            calls.append((model, quantization)) or ["quantized-target-cache"]
        ),
    )

    runtime = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        kv_quantization=(32, 8),
        array_ops=_ArrayOpsStub(),
    )

    assert runtime.compute.target_cache_factory() == ["quantized-target-cache"]
    assert calls == [(inner, (32, 8))]
    assert abi_checks == [True]


def test_dynamic_membership_requires_both_policy_and_descriptor_attestation():
    inner = _InjectedTextModel()
    enabled = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )
    policy_off = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        allow_dynamic_membership=False,
        array_ops=_ArrayOpsStub(),
    )
    unattested_inner = _InjectedTextModel(_descriptor(dynamic_join=False))
    unattested = runtime_module.assemble_continuous_self_mtp_runtime(
        unattested_inner,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )
    flash_inner = _InjectedTextModel(
        _descriptor(flash_dynamic_membership_attested=True)
    )
    flash_attested = runtime_module.assemble_continuous_self_mtp_runtime(
        flash_inner,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )
    flash_policy_off = runtime_module.assemble_continuous_self_mtp_runtime(
        flash_inner,
        allow_dynamic_membership=False,
        array_ops=_ArrayOpsStub(),
    )

    assert enabled.capabilities.dynamic_membership is True
    assert policy_off.capabilities.dynamic_membership is False
    assert unattested.config.allow_dynamic_membership is True
    assert unattested.capabilities.dynamic_membership is False
    assert enabled.capabilities.flash_dynamic_membership_attested is False
    assert flash_attested.capabilities.flash_dynamic_membership_attested is True
    assert flash_policy_off.capabilities.flash_dynamic_membership_attested is False


def test_qwen4_uses_its_injector_resolver_and_qsa_policy():
    descriptor = _descriptor("qwen4_exp")
    inner = _InjectedTextModel(descriptor)
    inner.model_type = "qwen4_exp_text"
    outer = _OuterModel(inner)

    runtime = runtime_module.assemble_continuous_self_mtp_runtime(
        outer,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )

    assert runtime.config.architecture == "qwen4_exp"
    assert runtime.compute.share_qsa_indices is True
    assert runtime.capabilities.dynamic_membership is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"protocol_version": 2}, "protocol_version"),
        ({"recursive_draft_depth": 1}, "recursive_draft_depth"),
        ({"fixed_membership": False}, "fixed_membership"),
        ({"target_return_hidden": False}, "target_return_hidden"),
        ({"mtp_return_hidden": False}, "mtp_return_hidden"),
        ({"confirmed_target_forward": False}, "confirmed_target_forward"),
        ({"ragged_rollback": False}, "ragged_rollback"),
        ({"atomic_cache_commit": False}, "atomic_cache_commit"),
        ({"quantized_cache": False}, "quantized_cache"),
        ({"windowed_cache": True}, "windowed_cache"),
        ({"xtc": True}, "xtc"),
        ({"batch_forward": None}, "batch_forward"),
        ({"share_qsa_indices": None}, "share_qsa_indices"),
        ({"model_family": "unknown"}, "unsupported model family"),
    ],
)
def test_descriptor_mismatches_fail_closed(changes, message):
    inner = _InjectedTextModel(_descriptor(**changes))

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match=message):
        runtime_module.assemble_continuous_self_mtp_runtime(
            inner,
            array_ops=_ArrayOpsStub(),
        )


def test_missing_injected_surfaces_and_target_abi_fail_closed():
    no_confirmed = _NoConfirmedTarget()
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="n_confirmed"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            no_confirmed,
            array_ops=_ArrayOpsStub(),
        )

    no_draft = _InjectedTextModel()
    no_draft.mtp_batch_forward = None
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="not callable"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            no_draft,
            array_ops=_ArrayOpsStub(),
        )

    no_cache = _InjectedTextModel()
    no_cache.make_mtp_cache = None
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="make_mtp_cache"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            no_cache,
            array_ops=_ArrayOpsStub(),
        )


def test_outer_and_resolved_inner_descriptors_must_match():
    inner = _InjectedTextModel()
    outer = _OuterModel(inner)
    outer.batched_mtp_capability = _descriptor(dynamic_join=False)

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="descriptors disagree"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            outer,
            allow_dynamic_membership=True,
            array_ops=_ArrayOpsStub(),
        )
