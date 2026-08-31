from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from vllm_mlx.kernels import qwen4_fused_moe
from vllm_mlx.models import qwen4_exp


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    def squeeze(self, axis):
        axis %= self.ndim
        return FakeArray(self.shape[:axis] + self.shape[axis + 1 :], self.dtype)


class FakeQuantizedDown:
    training = False
    num_experts = 512
    group_size = 64
    bits = 4
    mode = "affine"

    def __init__(self):
        self.weight = FakeArray((512, 2560, 80), mx.uint32)
        self.scales = FakeArray((512, 2560, 10), mx.bfloat16)
        self.biases = FakeArray((512, 2560, 10), mx.bfloat16)

    def __contains__(self, name):
        return False

    def __getitem__(self, name):
        return getattr(self, name)


def _production_inputs(tokens):
    prefix = (1, tokens)
    return (
        FakeArray(prefix + (10, 640), mx.bfloat16),
        FakeArray(prefix + (10,), mx.uint32),
        FakeArray(prefix + (10,), mx.bfloat16),
        FakeArray((512, 2560, 80), mx.uint32),
        FakeArray((512, 2560, 10), mx.bfloat16),
        FakeArray((512, 2560, 10), mx.bfloat16),
    )


def test_default_policy_is_auto_with_explicit_hard_off(monkeypatch):
    monkeypatch.delenv("MLX_QWEN4_FUSED_EXPERT_KERNEL", raising=False)
    assert qwen4_exp._qwen4_fused_expert_mode_from_env() == "auto"
    for value in ("0", "off", "false", "stock"):
        monkeypatch.setenv("MLX_QWEN4_FUSED_EXPERT_KERNEL", value)
        assert qwen4_exp._qwen4_fused_expert_mode_from_env() == "stock"


def test_kernel_admits_only_qualified_decode_widths():
    for tokens in (1, 3):
        admission = qwen4_fused_moe.admit_qwen4_fused_down(
            *_production_inputs(tokens)
        )
        assert admission.accepted, admission.reason
    refusal = qwen4_fused_moe.admit_qwen4_fused_down(*_production_inputs(2))
    assert not refusal.accepted
    assert "M=1 or M=3" in refusal.reason


def test_auto_selects_real_weight_winners_by_token_width():
    down = FakeQuantizedDown()
    sentinel = object()
    for tokens, expected in ((1, "tile4"), (3, "scalar")):
        hidden = FakeArray((1, tokens, 10, 1, 640), mx.bfloat16)
        indices = FakeArray((1, tokens, 10), mx.uint32)
        scores = FakeArray((1, tokens, 10), mx.bfloat16)
        admission = qwen4_fused_moe.FusedMoeAdmission(True, "eligible", tokens)
        with patch.object(
            qwen4_exp, "QuantizedSwitchLinear", FakeQuantizedDown
        ), patch.object(
            qwen4_exp, "admit_qwen4_fused_down", return_value=admission
        ), patch.object(
            qwen4_exp, "qwen4_fused_down", return_value=sentinel
        ) as execute:
            result, variant = qwen4_exp._try_qwen4_fused_expert_down(
                hidden, indices, scores, down, "auto"
            )
        assert result is sentinel
        assert variant == expected
        assert execute.call_args.kwargs["variant"] == expected


def test_unsupported_projection_falls_back_before_kernel_dispatch():
    hidden = FakeArray((1, 1, 10, 1, 640), mx.bfloat16)
    indices = FakeArray((1, 1, 10), mx.uint32)
    scores = FakeArray((1, 1, 10), mx.bfloat16)
    with patch.object(qwen4_exp, "qwen4_fused_down") as execute:
        result, reason = qwen4_exp._try_qwen4_fused_expert_down(
            hidden,
            indices,
            scores,
            SimpleNamespace(training=False),
            "auto",
        )
    assert result is None
    assert reason == "projection_layout"
    execute.assert_not_called()


def test_status_reports_configured_blocks_and_reached_path_counts(monkeypatch):
    monkeypatch.delenv("MLX_QWEN4_FUSED_EXPERT_KERNEL", raising=False)
    block = qwen4_exp.SparseMoeBlock(
        SimpleNamespace(
            hidden_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
        )
    )
    block.fused_expert_dispatches["tile4"] = 7
    block.fused_expert_dispatches["scalar"] = 3
    block.fused_expert_fallbacks = 2

    status = qwen4_exp.qwen4_fused_expert_status(block)
    assert status == {
        "mode_counts": {"stock": 0, "auto": 1, "scalar": 0, "tile4": 0},
        "dispatches": {"scalar": 3, "tile4": 7},
        "fallbacks": 2,
    }


def test_status_accepts_native_mlx_lm_observability_contract():
    native_block = SimpleNamespace(
        fused_expert_kernel_mode="auto",
        fused_expert_dispatches={"scalar": 9, "tile4": 11},
        fused_expert_fallbacks=4,
    )
    native_model = SimpleNamespace(
        named_modules=lambda: [("language_model.layers.0.mlp", native_block)]
    )

    assert qwen4_exp.qwen4_fused_expert_status(native_model) == {
        "mode_counts": {"stock": 0, "auto": 1, "scalar": 0, "tile4": 0},
        "dispatches": {"scalar": 9, "tile4": 11},
        "fallbacks": 4,
    }
