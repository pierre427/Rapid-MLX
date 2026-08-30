# SPDX-License-Identifier: Apache-2.0
"""Pure mocked/AST checks for model-family continuous-MTP capability seams."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_mlx.spec_decode.mtp import qwen3_5_inject, qwen4_exp_inject

ROOT = Path(__file__).resolve().parents[1]
FAMILY_MODULES = (qwen4_exp_inject, qwen3_5_inject)


@pytest.mark.parametrize(
    "module,family,dynamic_join",
    [
        (qwen4_exp_inject, "qwen4_exp", False),
        (qwen3_5_inject, "qwen3_5", True),
    ],
)
def test_capability_descriptor_is_explicit_immutable_and_conservative(
    module, family, dynamic_join
):
    capability = module.BATCHED_MTP_CAPABILITY
    assert dict(capability) == {
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
        "dynamic_join": dynamic_join,
        "flash_dynamic_membership_attested": False,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    with pytest.raises(TypeError):
        capability["dynamic_join"] = not dynamic_join


@pytest.mark.parametrize("module", FAMILY_MODULES)
def test_batch_forward_delegates_to_existing_recursive_hidden_path(module):
    calls = []

    class FakeInjectedModel:
        def mtp_forward(self, hidden, tokens, cache, *, return_hidden=False):
            calls.append((hidden, tokens, cache, return_hidden))
            return "logits", "next-hidden"

    result = module._mtp_batch_forward(FakeInjectedModel(), "hidden", "tokens", "cache")
    assert result == ("logits", "next-hidden")
    assert calls == [("hidden", "tokens", "cache", True)]


@pytest.mark.parametrize(
    "relative_path,injected_class",
    [
        ("vllm_mlx/spec_decode/mtp/qwen4_exp_inject.py", "_Qwen4ExpWithMTP"),
        ("vllm_mlx/spec_decode/mtp/qwen3_5_inject.py", "_Qwen3_5WithMTP"),
    ],
)
def test_injected_class_exposes_descriptor_seam_and_separate_recursive_depth(
    relative_path, injected_class
):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == injected_class
    )
    assignments = {
        target.id: node.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert isinstance(assignments["batched_mtp_capability"], ast.Name)
    assert assignments["batched_mtp_capability"].id == "BATCHED_MTP_CAPABILITY"
    assert isinstance(assignments["mtp_batch_forward"], ast.Name)
    assert assignments["mtp_batch_forward"].id == "_mtp_batch_forward"
    depth = assignments["mtp_recursive_draft_depth"]
    assert isinstance(depth, ast.Constant) and depth.value == 2
    # The new recursive batch depth is deliberately not implemented by
    # changing the legacy single-request cap on the injected class.
    assert "mtp_max_speculative_tokens" not in assignments


def test_flash_and_qwen35_join_claims_remain_intentionally_asymmetric():
    assert qwen4_exp_inject.BATCHED_MTP_CAPABILITY["dynamic_join"] is False
    assert qwen3_5_inject.BATCHED_MTP_CAPABILITY["dynamic_join"] is True


def test_qsa_selection_sharing_is_attested_only_for_qwen4():
    assert qwen4_exp_inject.BATCHED_MTP_CAPABILITY["share_qsa_indices"] is True
    assert qwen3_5_inject.BATCHED_MTP_CAPABILITY["share_qsa_indices"] is False


def test_native_qwen4_adapter_preserves_model_and_exposes_rapid_abi():
    calls = []

    class NativeQwen4:
        model_type = "qwen4_exp"
        supports_speculative_rollback = True

        def __init__(self):
            self.mtp = object()

        def __call__(self, inputs, cache=None, input_embeddings=None):
            calls.append(("plain", inputs, cache, input_embeddings))
            return "plain-logits"

        def mtp_backbone(self, inputs, cache=None):
            calls.append(("backbone", inputs, cache))
            return "logit-hidden", "mtp-hidden"

        def logits(self, hidden):
            calls.append(("logits", hidden))
            return "hidden-logits"

        def mtp_step(self, hidden, tokens, cache):
            calls.append(("mtp", hidden, tokens, cache))
            return "draft-logits", "next-hidden"

        def make_mtp_cache(self):
            return ["native-cache"]

    model = NativeQwen4()
    original_mtp = model.mtp
    assert qwen4_exp_inject._attach_native_qwen4_exp_mtp_support(model) is True

    assert model.mtp is original_mtp
    assert model("ids", cache="target-cache") == "plain-logits"
    assert model(
        "ids", cache="target-cache", return_hidden=True, n_confirmed=2
    ) == ("hidden-logits", "mtp-hidden")
    assert model.mtp_forward(
        "hidden", "tokens", "draft-cache", return_hidden=True
    ) == ("draft-logits", "next-hidden")
    assert model.mtp_batch_forward("hidden", "tokens", "draft-cache") == (
        "draft-logits",
        "next-hidden",
    )
    assert model.make_mtp_cache() == ["native-cache"]
    assert model.mtp_max_speculative_tokens == 2
    assert qwen4_exp_inject._resolve_inner(model) is model
    assert qwen4_exp_inject.validate_qwen4_exp_mtp_support(model) is True
    assert calls == [
        ("plain", "ids", "target-cache", None),
        ("backbone", "ids", "target-cache"),
        ("logits", "logit-hidden"),
        ("mtp", "hidden", "tokens", "draft-cache"),
        ("mtp", "hidden", "tokens", "draft-cache"),
    ]


@pytest.mark.parametrize("value,error", [(True, TypeError), (-1, ValueError)])
def test_native_qwen4_adapter_validates_confirmed_count(value, error):
    class NativeQwen4:
        model_type = "qwen4_exp"
        supports_speculative_rollback = True
        mtp = object()

        def __call__(self, inputs, cache=None, input_embeddings=None):
            return inputs

        def mtp_backbone(self, inputs, cache=None):
            return inputs, inputs

        def logits(self, hidden):
            return hidden

        def mtp_step(self, hidden, tokens, cache):
            return tokens, hidden

        def make_mtp_cache(self):
            return []

    model = NativeQwen4()
    assert qwen4_exp_inject._attach_native_qwen4_exp_mtp_support(model) is True
    with pytest.raises(error):
        model("ids", return_hidden=True, n_confirmed=value)
