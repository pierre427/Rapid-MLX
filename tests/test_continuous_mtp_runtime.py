"""Pure-Python contracts for continuous self-MTP runtime assembly."""

from __future__ import annotations

from types import SimpleNamespace

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
        "fixed_membership": True,
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
        "dynamic_join": True,
        "flash_dynamic_membership_attested": False,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    if family == "qwen4_exp":
        values["target_verify_mode"] = "tokenwise_exact"
        values["max_exact_fixed_lanes"] = 2
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


def test_assembler_resolves_inner_model_and_wires_forward_and_cache_seams(
    monkeypatch,
):
    inner = _InjectedTextModel()
    outer = _OuterModel(inner)
    target_cache_calls = []
    monkeypatch.setattr(
        runtime_module,
        "_make_prompt_cache",
        lambda model: target_cache_calls.append(model) or ["target-cache"],
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
    assert target_cache_calls == [inner]
    assert runtime.compute.draft_cache_factory() == ["draft-cache"]
    assert runtime.caches._preflight is preflight_ragged_cache
    assert runtime.caches._trim is trim_ragged_cache


def test_dynamic_membership_requires_policy_and_dense_attestation():
    inner = _InjectedTextModel()
    enabled = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )
    policy_off = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        array_ops=_ArrayOpsStub(),
    )
    assert enabled.capabilities.dynamic_membership is True
    assert policy_off.capabilities.dynamic_membership is False


def test_qwen4_assembles_fixed_membership_with_qwen4_cache_install(monkeypatch):
    descriptor = _descriptor("qwen4_exp", dynamic_join=False)
    inner = _InjectedTextModel(descriptor)
    inner.model_type = "qwen4_exp_text"
    outer = _OuterModel(inner)
    installs = []
    monkeypatch.setattr(
        "vllm_mlx.spec_decode.mtp.ragged_cache.install_ragged_cache_rollback",
        lambda **kwargs: installs.append(kwargs),
    )

    assembled = runtime_module.assemble_continuous_self_mtp_runtime(
        outer,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )

    assert assembled.config.architecture == "qwen4_exp"
    assert assembled.capabilities.dynamic_membership is False
    assert installs == [{}]


def test_qwen4_exact_target_verify_records_tokenwise_state_boundaries():
    import mlx.core as mx

    from vllm_mlx.models.qwen4_exp_cache import Qwen4ExpStateCache

    state = Qwen4ExpStateCache(1)
    state.cache[0] = mx.zeros((1, 1), dtype=mx.float32)
    calls = []

    class Inner:
        def __call__(
            self,
            inputs,
            *,
            cache,
            return_hidden,
            n_confirmed,
        ):
            calls.append((inputs.tolist(), return_hidden, n_confirmed))
            state.cache[0] = state.cache[0] + inputs[:, :1].astype(mx.float32)
            return inputs[..., None].astype(mx.float32), inputs[..., None].astype(
                mx.float32
            )

    logits, hidden = runtime_module._qwen4_exact_target_forward(
        Inner(),
        mx.array([[1, 2, 3]], dtype=mx.int32),
        cache=[state],
        return_hidden=True,
        n_confirmed=2,
    )
    mx.eval(logits, hidden, state.rollback_state)

    assert calls == [([[token]], True, 0) for token in (1, 2, 3)]
    assert logits.reshape(-1).tolist() == [1.0, 2.0, 3.0]
    assert hidden.reshape(-1).tolist() == [1.0, 2.0, 3.0]
    assert [snapshot[0].item() for snapshot in state.rollback_state] == [1, 3, 6]


def test_qwen4_dynamic_attestation_cannot_be_manufactured_by_caller():
    descriptor = _descriptor("qwen4_exp", dynamic_join=False)
    inner = _InjectedTextModel(descriptor)
    inner.model_type = "qwen4_exp_text"

    assembled = runtime_module.assemble_continuous_self_mtp_runtime(
        inner,
        allow_dynamic_membership=True,
        array_ops=_ArrayOpsStub(),
    )
    assert assembled.capabilities.dynamic_membership is False


def test_qwen4_refuses_nonexact_target_verify_mode():
    descriptor = _descriptor("qwen4_exp", target_verify_mode="block_approximate")
    inner = _InjectedTextModel(descriptor)
    inner.model_type = "qwen4_exp_text"

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="target_verify_mode"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            inner,
            array_ops=_ArrayOpsStub(),
        )


def test_qwen4_refuses_unattested_fixed_lane_width():
    descriptor = _descriptor("qwen4_exp", max_exact_fixed_lanes=4)
    inner = _InjectedTextModel(descriptor)
    inner.model_type = "qwen4_exp_text"

    with pytest.raises(
        ContinuousSelfMTPUnsupportedError, match="max_exact_fixed_lanes"
    ):
        runtime_module.assemble_continuous_self_mtp_runtime(
            inner,
            array_ops=_ArrayOpsStub(),
        )


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
        ({"quantized_cache": True}, "quantized_cache"),
        ({"windowed_cache": True}, "windowed_cache"),
        ({"xtc": True}, "xtc"),
        ({"batch_forward": None}, "batch_forward"),
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
    outer.batched_mtp_capability = _descriptor(protocol_version=2)

    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="descriptors disagree"):
        runtime_module.assemble_continuous_self_mtp_runtime(
            outer,
            array_ops=_ArrayOpsStub(),
        )
