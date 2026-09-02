# SPDX-License-Identifier: Apache-2.0
"""Contracts for the default-off fused Qwen4 GDN speculative-verify path."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.kernels import qwen4_fused_gdn_decode as fused_gdn
from vllm_mlx.kernels import qwen4_fused_gdn_verify as fused_verify
from vllm_mlx.kernels.qwen4_gdn_verify import gated_delta_verify_with_states
from vllm_mlx.models import qwen4_exp


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeCache:
    """Minimal stand-in for ``Qwen4ExpStateCache`` slot and snapshot calls."""

    def __init__(self, conv_state=None, recurrent_state=None):
        self.cache = [conv_state, recurrent_state]
        self.lengths = None
        self.advanced = 0
        self.snapshot_calls = []
        # One ordered log of every mutation, so tests can assert the
        # snapshot -> live slot -> advance ordering the engine relies on.
        self.events = []
        self._rollback_slots = None

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value
        self.events.append(("set", index))

    def advance(self, amount):
        self.advanced += amount
        self.events.append(("advance", amount))

    def record_slot_snapshots(self, slot, snapshots, *, finalize=False):
        self.snapshot_calls.append((slot, list(snapshots), finalize))
        self.events.append(("snapshot", slot, finalize))
        if self._rollback_slots is None:
            self._rollback_slots = {}
        self._rollback_slots[slot] = list(snapshots)


class FinalizeFailsCache(FakeCache):
    """Stages slot 0 like the real cache, then refuses slot-1 finalization."""

    def record_slot_snapshots(self, slot, snapshots, *, finalize=False):
        super().record_slot_snapshots(slot, snapshots, finalize=finalize)
        if finalize:
            raise AssertionError("snapshots do not cover every state slot")


class SnapshotlessCache(FakeCache):
    record_slot_snapshots = None


def production_values(steps=3, dtype=mx.bfloat16):
    return {
        "qkv": FakeArray((1, steps, 10240), dtype),
        "z": FakeArray((1, steps, 6144), dtype),
        "beta": FakeArray((1, steps, 48), dtype),
        "alpha": FakeArray((1, steps, 48), dtype),
        "conv_state": FakeArray((1, 3, 10240), dtype),
        "recurrent_state": FakeArray((1, 48, 128, 128), mx.float32),
        "conv_weight": FakeArray((10240, 4, 1), dtype),
        "a_log": FakeArray((48,), mx.float32),
        "dt_bias": FakeArray((48,), dtype),
        "norm_weight": FakeArray((128,), dtype),
    }


GEOMETRY = {
    "training": False,
    "sharded": False,
    "num_key_heads": 16,
    "num_value_heads": 48,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "conv_kernel": 4,
    "gate_activation": "sigmoid",
}


def admission(
    steps=3, *, mask=None, cache_lengths=None, record_rollback=True, **overrides
):
    values = production_values(steps)
    values.update(overrides)
    geometry = dict(GEOMETRY)
    for key in list(overrides):
        if key in geometry:
            geometry[key] = values.pop(key)
    return fused_verify.admit_qwen4_fused_gdn_verify(
        **values,
        mask=mask,
        cache_lengths=cache_lengths,
        record_rollback=record_rollback,
        **geometry,
    )


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )


class Identity:
    def __call__(self, value):
        return value


def test_production_verify_widths_are_admitted():
    for steps in range(2, fused_verify.MAX_VERIFY_STEPS + 1):
        result = admission(steps)
        assert result.accepted, (steps, result.reason)


def test_single_token_batch_mask_ragged_and_plain_decode_fall_back():
    assert admission(1).reason == "verify width 1 below 2"
    wide = fused_verify.MAX_VERIFY_STEPS + 1
    assert admission(wide).reason == f"verify width {wide} above 8"
    result = admission(qkv=FakeArray((2, 3, 10240), mx.bfloat16))
    assert "qkv shape" in result.reason
    assert admission(mask=object()).reason == "masked verify"
    assert admission(cache_lengths=object()).reason == "ragged cache lengths"
    assert admission(record_rollback=False).reason == "not a speculative verify"
    assert admission(training=True).reason == "training"
    assert admission(sharded=True).reason == "distributed sharding"


def test_dtype_and_geometry_are_strict():
    result = admission(a_log=FakeArray((48,), mx.float16))
    assert "A_log" in result.reason
    assert admission(a_log=FakeArray((48,), mx.bfloat16)).accepted
    assert "unsupported geometry" in admission(num_key_heads=24).reason
    assert "z shape" in admission(z=FakeArray((1, 2, 6144), mx.bfloat16)).reason
    assert (
        "recurrent_state must be float32"
        in admission(recurrent_state=FakeArray((1, 48, 128, 128), mx.bfloat16)).reason
    )
    assert admission(qkv=FakeArray((1, 3, 10240), mx.float16)).reason == (
        "unsupported activation dtype mlx.core.float16"
    )


def test_kernel_dispatch_emits_restore_points_for_every_earlier_position():
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [
            FakeArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        ]

    values = production_values(steps=3)
    with patch.object(fused_verify, "_kernel", return_value=fake_kernel):
        outputs = fused_verify.qwen4_fused_gdn_verify(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            values["conv_state"],
            values["conv_weight"],
            values["a_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=16,
        )
    assert calls[0]["grid"] == (32, 16, 48)
    assert calls[0]["threadgroup"] == (32, 16, 1)
    assert ("S", 3) in calls[0]["template"]
    assert ("RATIO", 3) in calls[0]["template"]
    assert [item.shape for item in outputs] == [
        (1, 3, 6144),
        (1, 3, 10240),
        (1, 48, 128, 128),
        (1, 2, 48, 128, 128),
        (1, 2, 3, 10240),
    ]
    assert outputs[3].dtype == mx.float32
    assert outputs[4].dtype == mx.bfloat16


def test_kernel_dispatch_rejects_unsupported_widths_and_threadgroups():
    values = production_values(steps=1)
    with pytest.raises(ValueError, match="verify width"):
        fused_verify.qwen4_fused_gdn_verify(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            values["conv_state"],
            values["conv_weight"],
            values["a_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=32,
        )
    values = production_values(steps=3)
    with pytest.raises(ValueError, match="threadgroup_y"):
        fused_verify.qwen4_fused_gdn_verify(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            values["conv_state"],
            values["conv_weight"],
            values["a_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=3,
        )


def test_probe_caches_per_width_and_reuses_decode_geometry():
    with (
        patch.dict(fused_verify._PROBED_STEPS, {}, clear=True),
        patch.object(fused_verify, "fused_gdn_runtime_supported", return_value=True),
        patch.object(fused_verify, "probe_qwen4_fused_gdn_decode", return_value=16),
        patch.object(
            fused_verify,
            "qwen4_fused_gdn_verify",
            side_effect=[
                RuntimeError("threadgroup resources"),
                (object(),) * 5,
                (object(),) * 5,
            ],
        ) as execute,
        patch.object(fused_verify.mx, "eval"),
    ):
        # Width 3: the decode geometry (16) is rejected, the next smaller
        # candidate (8) works and is cached; width 4 succeeds at 16 directly;
        # width 1 is refused without touching Metal.
        assert fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 3) == 8
        assert fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 3) == 8
        assert fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 4) == 16
        assert fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 1) is None
    assert [call.kwargs["threadgroup_y"] for call in execute.call_args_list] == [
        16,
        8,
        16,
    ]
    assert execute.call_args_list[2].args[0].shape == (1, 4, fused_gdn.CONV_DIM)


def test_probe_exhausts_every_candidate_before_declining():
    with (
        patch.dict(fused_verify._PROBED_STEPS, {}, clear=True),
        patch.object(fused_verify, "fused_gdn_runtime_supported", return_value=True),
        patch.object(fused_verify, "probe_qwen4_fused_gdn_decode", return_value=32),
        patch.object(
            fused_verify,
            "qwen4_fused_gdn_verify",
            side_effect=RuntimeError("threadgroup resources"),
        ) as execute,
        patch.object(fused_verify.mx, "eval"),
    ):
        assert fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 3) is None
    assert [call.kwargs["threadgroup_y"] for call in execute.call_args_list] == [
        32,
        16,
        8,
        4,
    ]


def test_probe_declines_when_decode_probe_declined():
    with (
        patch.dict(fused_verify._PROBED_STEPS, {}, clear=True),
        patch.object(fused_verify, "fused_gdn_runtime_supported", return_value=True),
        patch.object(fused_verify, "probe_qwen4_fused_gdn_decode", return_value=None),
        patch.object(fused_verify, "qwen4_fused_gdn_verify") as execute,
    ):
        assert fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 3) is None
    execute.assert_not_called()


def test_resident_verify_switch_is_independent_of_decode_and_defaults_stock():
    with (
        patch.object(qwen4_exp, "_FUSED_GDN_DEFAULT", False),
        patch.object(qwen4_exp, "_FUSED_GDN_VERIFY_DEFAULT", False),
    ):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
    weight = layer.conv1d.weight
    assert qwen4_exp.qwen4_fused_gdn_verify_mode_counts(layer) == {
        "stock": 1,
        "fused": 0,
    }
    assert qwen4_exp.set_qwen4_fused_gdn_verify_mode(layer, "fused") == 1
    assert layer.fused_gdn_verify_mode == "fused"
    assert layer.fused_gdn_decode_mode == "stock"
    assert layer.conv1d.weight is weight
    assert qwen4_exp.set_qwen4_fused_gdn_verify_mode(layer, "stock") == 1
    with pytest.raises(ValueError, match="unknown fused GDN verify mode"):
        qwen4_exp.set_qwen4_fused_gdn_verify_mode(layer, "other")
    stats = qwen4_exp.qwen4_fused_gdn_stats(layer)
    assert stats["verify_calls"] == 0
    assert stats["verify_fallbacks"] == 0
    assert stats["verify_last_fallbacks"] == {}


def test_forward_routes_verify_blocks_and_single_tokens_separately():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    sentinel = object()
    with (
        patch.object(layer, "_try_fused_verify", return_value=sentinel) as verify,
        patch.object(layer, "_try_fused_decode", return_value=sentinel) as decode,
    ):
        result = layer(mx.zeros((1, 3, 16)), None, None, record_rollback=True)
        assert result is sentinel
        verify.assert_called_once()
        decode.assert_not_called()
    with (
        patch.object(layer, "_try_fused_verify", return_value=sentinel) as verify,
        patch.object(layer, "_try_fused_decode", return_value=sentinel) as decode,
    ):
        result = layer(mx.zeros((1, 1, 16)), None, None, record_rollback=True)
        assert result is sentinel
        verify.assert_not_called()
        assert decode.call_args.kwargs["record_rollback"] is True
    with (
        patch.object(layer, "_try_fused_verify", return_value=sentinel) as verify,
        patch.object(layer, "_try_fused_decode", return_value=sentinel) as decode,
    ):
        result = layer(mx.zeros((1, 3, 16)), None, None, record_rollback=False)
        assert result is sentinel
        verify.assert_not_called()
        assert decode.call_args.kwargs["record_rollback"] is False


def test_stock_mode_and_unfit_caches_do_not_probe_metal():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    values = production_values(steps=3)
    args = (values["qkv"], values["z"], values["beta"], values["alpha"], None)
    with patch.object(qwen4_exp, "fused_gdn_runtime_supported") as runtime:
        layer.set_fused_gdn_verify_mode("stock")
        assert layer._try_fused_verify(*args, FakeCache()) is None
        assert layer.fused_gdn_verify_fallbacks == 0

        layer.set_fused_gdn_verify_mode("fused")
        assert layer._try_fused_verify(*args, FakeCache()) is None
        assert layer.fused_gdn_verify_last_fallback == "uninitialized cache"

        cache = SnapshotlessCache(values["conv_state"], values["recurrent_state"])
        assert layer._try_fused_verify(*args, cache) is None
        assert layer.fused_gdn_verify_last_fallback == "cache lacks slot snapshots"

        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        cache.lengths = object()
        assert layer._try_fused_verify(*args, cache) is None
        assert layer.fused_gdn_verify_last_fallback == "ragged cache lengths"
    runtime.assert_not_called()
    assert layer.fused_gdn_verify_fallbacks == 3
    assert layer.fused_gdn_verify_calls == 0


def test_admitted_verify_publishes_restore_points_before_live_slots():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    layer.out_proj = Identity()
    values = production_values(steps=3)
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    fused_output = FakeArray((1, 3, 6144), mx.bfloat16)
    next_conv = object()
    next_state = object()
    state_snapshots = mx.zeros((1, 2, 2, 2, 2))
    conv_snapshots = mx.zeros((1, 2, 3, 4))
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_verify", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(
            qwen4_exp, "probe_qwen4_fused_gdn_verify", return_value=8
        ) as probe,
        patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_verify",
            return_value=(
                fused_output,
                next_conv,
                next_state,
                state_snapshots,
                conv_snapshots,
            ),
        ) as execute,
    ):
        result = layer._try_fused_verify(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
        )
    assert result is fused_output
    assert probe.call_args.args == (mx.bfloat16, 3)
    assert execute.call_args.kwargs["threadgroup_y"] == 8
    assert cache[0] is next_conv
    assert cache[1] is next_state
    assert cache.advanced == 3
    assert layer.fused_gdn_verify_calls == 1
    assert layer.fused_gdn_verify_last_fallback is None
    # Stock order: the convolution slot first, then the recurrent slot with
    # ``finalize=True``; one restore point per earlier position.
    assert [
        (slot, len(items), finalize) for slot, items, finalize in cache.snapshot_calls
    ] == [
        (0, 2, False),
        (1, 2, True),
    ]
    for position, item in enumerate(cache.snapshot_calls[0][1]):
        assert item.shape == (1, 3, 4)
        assert mx.array_equal(item, conv_snapshots[:, position]).item()
    for position, item in enumerate(cache.snapshot_calls[1][1]):
        assert item.shape == (1, 2, 2, 2)
        assert mx.array_equal(item, state_snapshots[:, position]).item()
    # The full mutation order: both restore-point slots are published before
    # either live slot changes, and the cursor advances last.
    assert cache.events == [
        ("snapshot", 0, False),
        ("snapshot", 1, True),
        ("set", 0),
        ("set", 1),
        ("advance", 3),
    ]


def test_snapshot_contract_failure_unwinds_staging_and_leaves_cache_untouched():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    values = production_values(steps=3)
    cache = FinalizeFailsCache(values["conv_state"], values["recurrent_state"])
    # A PLE layer would have staged its own slots before the GDN call.
    staged_before = {3: ["ple-tail"], 2: ["ple-history"]}
    cache._rollback_slots = dict(staged_before)
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_verify", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(qwen4_exp, "probe_qwen4_fused_gdn_verify", return_value=8),
        patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_verify",
            return_value=(
                FakeArray((1, 3, 6144), mx.bfloat16),
                object(),
                object(),
                mx.zeros((1, 2, 2, 2, 2)),
                mx.zeros((1, 2, 3, 4)),
            ),
        ),
        pytest.raises(AssertionError, match="every state slot"),
    ):
        layer._try_fused_verify(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
        )
    assert cache[0] is values["conv_state"]
    assert cache[1] is values["recurrent_state"]
    assert cache.advanced == 0
    assert cache._rollback_slots == staged_before
    assert [event for event in cache.events if event[0] != "snapshot"] == []
    assert layer.fused_gdn_verify_calls == 0


def test_dispatch_failure_and_probe_failure_preserve_cache():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    values = production_values(steps=3)
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    for failure, expected in (
        (
            {"qwen4_fused_gdn_verify": RuntimeError("dispatch rejected")},
            "Metal kernel dispatch failed: RuntimeError",
        ),
        (
            {"probe_qwen4_fused_gdn_verify": ValueError("probe rejected")},
            "Metal kernel dispatch failed: ValueError",
        ),
        ({"probe_qwen4_fused_gdn_verify": None}, "Metal kernel probe declined"),
    ):
        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        patches = {
            "admit_qwen4_fused_gdn_verify": patch.object(
                qwen4_exp, "admit_qwen4_fused_gdn_verify", return_value=accepted
            ),
            "fused_gdn_runtime_supported": patch.object(
                qwen4_exp, "fused_gdn_runtime_supported", return_value=True
            ),
            "probe_qwen4_fused_gdn_verify": patch.object(
                qwen4_exp, "probe_qwen4_fused_gdn_verify", return_value=8
            ),
            "qwen4_fused_gdn_verify": patch.object(qwen4_exp, "qwen4_fused_gdn_verify"),
        }
        for name, effect in failure.items():
            patches[name] = patch.object(
                qwen4_exp,
                name,
                **(
                    {"side_effect": effect}
                    if isinstance(effect, Exception)
                    else {"return_value": effect}
                ),
            )
        with (
            patches["admit_qwen4_fused_gdn_verify"],
            patches["fused_gdn_runtime_supported"],
            patches["probe_qwen4_fused_gdn_verify"],
            patches["qwen4_fused_gdn_verify"],
        ):
            result = layer._try_fused_verify(
                values["qkv"],
                values["z"],
                values["beta"],
                values["alpha"],
                None,
                cache,
            )
        assert result is None
        assert cache[0] is values["conv_state"]
        assert cache[1] is values["recurrent_state"]
        assert cache.advanced == 0
        assert cache.snapshot_calls == []
        assert layer.fused_gdn_verify_last_fallback == expected
    assert layer.fused_gdn_verify_calls == 0
    assert layer.fused_gdn_verify_fallbacks == 3


def _stock_verify_block(
    qkv, z, beta, alpha, conv_state, conv_weight, a_log, dt_bias, norm_weight, state
):
    """Reproduce ``GatedDeltaNet.__call__``'s stock verify path (B=1, S>1)."""
    steps = qkv.shape[1]
    keep = fused_gdn.CONV_KERNEL - 1
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv = mx.contiguous(conv_input[:, -keep:, :])
    conv_snapshots = [
        mx.contiguous(conv_input[:, position : position + keep, :])
        for position in range(1, steps)
    ]
    convolved = nn.silu(mx.conv1d(conv_input, conv_weight, groups=fused_gdn.CONV_DIM))
    query, key, value = [
        item.reshape(1, steps, heads, dim)
        for item, heads, dim in zip(
            mx.split(convolved, [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM], axis=-1),
            [
                fused_gdn.NUM_KEY_HEADS,
                fused_gdn.NUM_KEY_HEADS,
                fused_gdn.NUM_VALUE_HEADS,
            ],
            [fused_gdn.KEY_HEAD_DIM, fused_gdn.KEY_HEAD_DIM, fused_gdn.VALUE_HEAD_DIM],
        )
    ]
    query = query * mx.rsqrt(mx.sum(mx.square(query), axis=-1, keepdims=True) + 1e-6)
    key = key * mx.rsqrt(mx.sum(mx.square(key), axis=-1, keepdims=True) + 1e-6)
    query = query * (fused_gdn.KEY_HEAD_DIM**-0.5)
    output, next_state, state_snapshots = gated_delta_verify_with_states(
        query, key, value, alpha, beta, a_log, dt_bias, state, None, use_kernel=True
    )
    output = (
        mx.fast.rms_norm(output, norm_weight, 1e-6).astype(mx.float32)
        * mx.sigmoid(
            z.reshape(
                1, steps, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM
            ).astype(mx.float32)
        )
    ).astype(qkv.dtype)
    return (
        output.reshape(1, steps, fused_gdn.VALUE_DIM),
        next_conv,
        next_state,
        [state_snapshots[:, position] for position in range(steps - 1)],
        conv_snapshots,
    )


def _assert_real_metal_verify_matches_stock(steps, threadgroup_y, blocks=6):
    """Bit-exact output, caches and every restore point, through rollbacks."""
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    dtype = mx.bfloat16
    conv_weight = (
        mx.random.normal(
            (fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1), key=mx.random.key(1)
        )
        * 0.02
    ).astype(dtype)
    a_log = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(2)) * 0.2
    ).astype(mx.float32)
    dt_bias = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(3)) * 0.2
    ).astype(dtype)
    norm_weight = (
        mx.random.normal((fused_gdn.VALUE_HEAD_DIM,), key=mx.random.key(4)) * 0.05 + 1
    ).astype(dtype)
    stock_conv = mx.zeros(
        (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=dtype
    )
    fused_conv = mx.array(stock_conv)
    stock_state = mx.zeros(
        (
            1,
            fused_gdn.NUM_VALUE_HEADS,
            fused_gdn.VALUE_HEAD_DIM,
            fused_gdn.KEY_HEAD_DIM,
        ),
        dtype=mx.float32,
    )
    fused_state = mx.array(stock_state)

    try:
        for block in range(blocks):
            seed = 1000 * steps + 10 * block

            def draw(shape, offset, scale=0.2, seed=seed):
                return (
                    mx.random.normal(shape, key=mx.random.key(seed + offset)) * scale
                ).astype(dtype)

            qkv = draw((1, steps, fused_gdn.CONV_DIM), 1)
            z = draw((1, steps, fused_gdn.VALUE_DIM), 2)
            beta = draw((1, steps, fused_gdn.NUM_VALUE_HEADS), 3)
            alpha = draw((1, steps, fused_gdn.NUM_VALUE_HEADS), 4)

            stock = _stock_verify_block(
                qkv,
                z,
                beta,
                alpha,
                stock_conv,
                conv_weight,
                a_log,
                dt_bias,
                norm_weight,
                stock_state,
            )
            fused = fused_verify.qwen4_fused_gdn_verify(
                qkv,
                z,
                beta,
                alpha,
                fused_conv,
                conv_weight,
                a_log,
                dt_bias,
                fused_state,
                norm_weight,
                1e-6,
                threadgroup_y=threadgroup_y,
            )
            fused_state_snaps = [fused[3][:, p] for p in range(steps - 1)]
            fused_conv_snaps = [fused[4][:, p] for p in range(steps - 1)]
            mx.eval(*stock[:3], *stock[3], *stock[4], *fused[:3], fused[3], fused[4])
            assert mx.array_equal(stock[0], fused[0]).item(), ("output", block)
            assert mx.array_equal(stock[1], fused[1]).item(), ("conv", block)
            assert mx.array_equal(stock[2], fused[2]).item(), ("state", block)
            for p in range(steps - 1):
                assert mx.array_equal(stock[3][p], fused_state_snaps[p]).item(), (
                    "state snapshot",
                    block,
                    p,
                )
                assert mx.array_equal(stock[4][p], fused_conv_snaps[p]).item(), (
                    "conv snapshot",
                    block,
                    p,
                )
            # Alternate between committing the whole block and rolling both
            # arms back to an earlier restore point, so later blocks start
            # from the published snapshots exactly as the engine would.
            keep = block % steps  # 0 commits everything; k>0 keeps k tokens
            if keep == 0:
                stock_conv, stock_state = stock[1], stock[2]
                fused_conv, fused_state = fused[1], fused[2]
            else:
                stock_conv, stock_state = stock[4][keep - 1], stock[3][keep - 1]
                fused_conv = fused_conv_snaps[keep - 1]
                fused_state = fused_state_snaps[keep - 1]
    finally:
        mx.set_default_device(previous_device)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("steps", [2, 3, 5, fused_verify.MAX_VERIFY_STEPS])
def test_real_metal_verify_matches_stock_for_every_supported_threadgroup(steps):
    supported = []
    for threadgroup_y in fused_gdn._THREADGROUP_Y_CANDIDATES:
        try:
            _assert_real_metal_verify_matches_stock(steps, threadgroup_y)
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
        except RuntimeError:
            # Mirror the production probe: a Metal resource rejection means
            # this candidate is unsupported, so try the next smaller one.
            continue
        else:
            supported.append(threadgroup_y)
    assert supported, "Metal rejected every fused GDN verify threadgroup candidate"


@pytest.mark.requires_mlx
@pytest.mark.parametrize("slots", [2, 4])
def test_real_metal_layer_verify_matches_stock_layer_through_restore(slots):
    """Drive ``GatedDeltaNet`` itself: routing, snapshots, restore, counters.

    ``slots=4`` mimics a PLE layer, whose cache stages slots 3 and 2 before
    the GDN mixer publishes slots 0 and 1 with ``finalize=True``; every
    restore then has to move all four slots atomically.
    """
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        args = SimpleNamespace(
            hidden_size=256,
            linear_num_value_heads=fused_gdn.NUM_VALUE_HEADS,
            linear_num_key_heads=fused_gdn.NUM_KEY_HEADS,
            linear_key_head_dim=fused_gdn.KEY_HEAD_DIM,
            linear_value_head_dim=fused_gdn.VALUE_HEAD_DIM,
            linear_conv_kernel_dim=fused_gdn.CONV_KERNEL,
            rms_norm_eps=1.0e-6,
            output_gate_type="sigmoid",
            hidden_act="silu",
        )
        mx.random.seed(7)
        layer = qwen4_exp.GatedDeltaNet(args)
        layer.eval()
        layer.A_log = layer.A_log.astype(mx.float32)
        layer.dt_bias = layer.dt_bias.astype(mx.bfloat16)
        layer.norm.weight = layer.norm.weight.astype(mx.bfloat16)
        layer.set_dtype(mx.bfloat16)
        layer.A_log = layer.A_log.astype(mx.float32)
        mx.eval(layer.parameters())
        steps = 3
        stock_cache = qwen4_exp.Qwen4ExpStateCache(slots)
        fused_cache = qwen4_exp.Qwen4ExpStateCache(slots)
        warm = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
        layer.set_fused_gdn_verify_mode("stock")
        layer.set_fused_gdn_decode_mode("stock")
        mx.eval(layer(warm, cache=stock_cache), layer(warm, cache=fused_cache))
        for cache in (stock_cache, fused_cache):
            for slot in range(2, slots):
                cache.cache[slot] = mx.array([100 + slot])

        def stage_ple_slots(cache, block):
            # PLE stages its tail (slot 3) and history (slot 2) first.
            for slot in reversed(range(2, slots)):
                cache.record_slot_snapshots(
                    slot,
                    [
                        mx.array([block, position, slot])
                        for position in range(steps - 1)
                    ],
                )

        for block in range(6):
            hidden = (mx.random.normal((1, steps, args.hidden_size)) * 0.5).astype(
                mx.bfloat16
            )
            layer.set_fused_gdn_verify_mode("stock")
            stage_ple_slots(stock_cache, block)
            stock_out = layer(hidden, cache=stock_cache, record_rollback=True)
            layer.set_fused_gdn_verify_mode("fused")
            stage_ple_slots(fused_cache, block)
            fused_out = layer(hidden, cache=fused_cache, record_rollback=True)
            mx.eval(stock_out, fused_out, *stock_cache.cache, *fused_cache.cache)
            assert mx.array_equal(stock_out, fused_out).item(), block
            for slot in range(slots):
                assert mx.array_equal(
                    stock_cache.cache[slot], fused_cache.cache[slot]
                ).item(), (block, slot)
            assert len(stock_cache.rollback_state) == steps - 1
            assert len(fused_cache.rollback_state) == steps - 1
            for position in range(steps - 1):
                assert len(fused_cache.rollback_state[position]) == slots
                for slot in range(slots):
                    a = stock_cache.rollback_state[position][slot]
                    b = fused_cache.rollback_state[position][slot]
                    mx.eval(a, b)
                    assert mx.array_equal(a, b).item(), (block, position, slot)
            n_to_drop = 1 + block % (steps - 1)
            stock_cache.restore_rollback(n_to_drop, steps)
            fused_cache.restore_rollback(n_to_drop, steps)
        assert layer.fused_gdn_verify_calls == 6
        assert layer.fused_gdn_verify_fallbacks == 0
        assert layer.fused_gdn_decode_calls == 0
    finally:
        mx.set_default_device(previous_device)


def _every_finite_bf16():
    import numpy as np

    bits = np.arange(0, 65536, dtype=np.uint32)
    values = (bits << 16).view(np.float32)
    return mx.array(values[np.isfinite(values)]).astype(mx.bfloat16)


def _elementwise_kernel(name, body, out_dtype, x):
    kernel = mx.fast.metal_kernel(
        name=f"sweep_{name}",
        input_names=["x"],
        output_names=["out"],
        header=fused_gdn._HEADER,
        source="uint i = thread_position_in_grid.x; " + body,
    )
    (out,) = kernel(
        inputs=[x],
        template=[("T", mx.bfloat16)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[out_dtype],
    )
    return out


@pytest.mark.requires_mlx
def test_real_metal_unary_boundaries_match_mlx_on_every_finite_bf16_value():
    """Pin the sigmoid forms both fused kernels use, over the whole bf16 domain.

    The beta gate once used ``mlx_sigmoid_fast<bf16>``, which differs from
    ``mx.sigmoid`` on exactly one finite bf16 input (x ~ -6.85); real Qwen4
    activations reach it. Sampled trajectories cannot see a one-value
    boundary, an exhaustive sweep can.
    """
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    from vllm_mlx.kernels.qwen4_gdn_verify import _compute_g_beta

    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        x = _every_finite_bf16()
        assert x.size == 65280

        beta = _elementwise_kernel(
            "beta_gate",
            "out[i] = float(mlx_sigmoid_precise(x[i]));",
            mx.float32,
            x,
        )
        silu = _elementwise_kernel(
            "silu",
            "{ T v = x[i]; T s = mlx_sigmoid_fast(v); out[i] = v * s; }",
            mx.bfloat16,
            x,
        )
        output_gate = _elementwise_kernel(
            "output_gate",
            "out[i] = mlx_sigmoid_precise<float>(float(x[i]));",
            mx.float32,
            x,
        )
        a_log = mx.array([0.7], mx.float32)
        decay = _elementwise_kernel(
            "decay",
            "{ T sp = mlx_softplus_fast(x[i]); out[i] = metal::precise::exp("
            "-metal::precise::exp(0.7f) * float(sp)); }",
            mx.float32,
            x,
        )
        ref_decay, ref_beta = _compute_g_beta(a_log, x, x, mx.zeros_like(x))
        mx.eval(beta, silu, output_gate, decay, ref_decay, ref_beta)

        assert int(mx.sum(beta != ref_beta.astype(mx.float32)).item()) == 0
        assert int(mx.sum(silu != nn.silu(x)).item()) == 0
        assert int(mx.sum(output_gate != mx.sigmoid(x.astype(mx.float32))).item()) == 0
        assert int(mx.sum(decay != ref_decay).item()) == 0
        # The rejected forms really do differ, so the sweep is not vacuous.
        fast_beta = _elementwise_kernel(
            "beta_gate_fast", "out[i] = float(mlx_sigmoid_fast(x[i]));", mx.float32, x
        )
        mx.eval(fast_beta)
        assert int(mx.sum(fast_beta != ref_beta.astype(mx.float32)).item()) == 1
    finally:
        mx.set_default_device(previous_device)
