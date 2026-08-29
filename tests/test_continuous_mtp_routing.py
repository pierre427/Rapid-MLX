"""Model-free and AST tests for PR 9 continuous-MTP routing."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType

import pytest

from vllm_mlx.spec_decode.config import (
    SpeculativeConfigError,
    parse_speculative_config,
)
from vllm_mlx.spec_decode.mtp.batched import SamplingContract
from vllm_mlx.spec_decode.mtp.continuous_routing import (
    ContinuousMTPAPCHit,
    ContinuousMTPIntegrationRoute,
    ContinuousMTPRequestMetadata,
    plan_router_install,
)
from vllm_mlx.spec_decode.mtp.prepared_state import (
    PreparedStateIdentity,
    prepare_mtp_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _descriptor(family="qwen3_5"):
    return MappingProxyType(
        {
            "protocol_version": 1,
            "model_family": family,
            "batch_forward": "mtp_batch_forward",
            "recursive_draft_depth": 2,
            "fixed_membership": True,
            "dynamic_join": family == "qwen3_5",
            "quantized_cache": False,
            "windowed_cache": False,
            "xtc": False,
        }
    )


class _Model:
    batched_mtp_capability = _descriptor()
    mtp = object()

    def __call__(self, *args, **kwargs):
        return args, kwargs

    def mtp_batch_forward(self, *args, **kwargs):
        return args, kwargs

    def mtp_forward(self, *args, **kwargs):
        return args, kwargs

    def make_mtp_cache(self):
        return object()


def _request(lane, uid, **changes):
    values = {
        "lane_id": lane,
        "uid": uid,
        "prompt_tokens": (10 + uid, 20 + uid),
        "max_tokens": 16,
    }
    values.update(changes)
    return ContinuousMTPRequestMetadata(**values)


def _router(**changes):
    values = {"enabled": True, "hard_reserve_bytes": 0}
    values.update(changes)
    decision = plan_router_install(_Model(), **values)
    assert decision.admitted is True
    assert decision.router is not None
    return decision.router


def _identity():
    return PreparedStateIdentity.from_config(
        model_id="Qwen/Qwen3.8-Flash-Next",
        model_revision="revision-a",
        speculative_config={"method": "mtp", "continuous_batching": True},
        target_cache_layout="qwen4-batch-kv:bf16",
        mtp_cache_layout="qwen4-mtp-batch-kv:bf16",
        seed_hidden_layout="bf16[1,1,2048]",
    )


def _apc_hit(prefix, *, identity=None):
    expected = _identity()
    state = prepare_mtp_state(
        identity=identity or expected,
        prefix_tokens=prefix,
        target_cache="target-cache",
        target_cache_tokens=len(prefix),
        mtp_cache="mtp-cache",
        mtp_cache_pairs=len(prefix) - 1,
        seed_hidden="seed-hidden",
        captured_at=10.0,
    )
    return ContinuousMTPAPCHit(
        state=state,
        expected_identity=expected,
        target_cache_tokens=len(prefix),
        mtp_cache_pairs=len(prefix) - 1,
        now=11.0,
        max_age_seconds=60.0,
    )


def test_speculative_config_continuous_batching_is_explicit_and_mtp_only():
    default = parse_speculative_config('{"method":"mtp"}')
    enabled = parse_speculative_config('{"method":"mtp","continuous_batching":true}')
    assert default is not None and default.continuous_batching is False
    assert enabled is not None and enabled.continuous_batching is True

    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","continuous_batching":1}')
    with pytest.raises(SpeculativeConfigError, match="unsupported speculative-config"):
        parse_speculative_config('{"method":"suffix","continuous_batching":true}')


def test_install_plan_is_default_off_and_fails_closed_without_mutation():
    model = _Model()
    before = dict(vars(model))

    disabled = plan_router_install(model, enabled=False, hard_reserve_bytes=0)
    quantized = plan_router_install(
        model,
        enabled=True,
        cache_quantized=True,
        hard_reserve_bytes=0,
    )

    assert disabled.admitted is False
    assert disabled.fallback is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert "disabled" in " ".join(disabled.reasons)
    assert quantized.admitted is False
    assert "quantized cache" in " ".join(quantized.reasons)
    assert vars(model) == before


def test_supported_requests_build_an_immutable_fixed_cohort_plan():
    router = _router()

    decision = router.plan([_request("a", 1), _request("b", 2)], free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert decision.live_token_delivery is False
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert [lane.spec.uid for lane in decision.cohort] == [1, 2]
    assert [lane.spec.num_draft for lane in decision.cohort] == [2, 2]
    assert all(lane.prepared_state is None for lane in decision.cohort)


def test_exact_apc_sidecar_is_validated_and_carried_as_a_resume_plan():
    router = _router()
    prefix = tuple(range(64))
    hit = _apc_hit(prefix)
    requests = [
        _request(
            "a",
            1,
            prompt_tokens=prefix + (999,),
            apc_hit=hit,
        ),
        _request("b", 2),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    restored = decision.cohort[0]
    assert restored.resume_at == 64
    assert restored.spec.prompt == (999,)
    assert restored.spec.prompt_cache == "target-cache"
    assert restored.spec.mtp_cache == "mtp-cache"
    assert restored.prepared_state is hit.state
    assert decision.live_token_delivery is False


def test_bad_apc_sidecar_routes_plain_while_other_lanes_form_cohort():
    router = _router()
    prefix = tuple(range(64))
    foreign = PreparedStateIdentity.from_config(
        model_id="other/model",
        model_revision="revision-a",
        speculative_config={"method": "mtp", "continuous_batching": True},
        target_cache_layout="qwen4-batch-kv:bf16",
        mtp_cache_layout="qwen4-mtp-batch-kv:bf16",
        seed_hidden_layout="bf16[1,1,2048]",
    )
    requests = [
        _request(
            "bad-apc",
            1,
            prompt_tokens=prefix + (999,),
            apc_hit=_apc_hit(prefix, identity=foreign),
        ),
        _request("a", 2),
        _request("b", 3),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert decision.plain_lane_ids == ("bad-apc",)
    assert "model_mismatch" in " ".join(decision.reasons)


def test_unsupported_sampling_falls_back_to_legacy_without_a_cohort():
    router = _router()
    transformed = SamplingContract(greedy=False)
    requests = [
        _request("a", 1, sampling=transformed, temperature=0.8),
        _request("b", 2, sampling=transformed, temperature=0.8),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert decision.cohort == ()
    assert decision.legacy_lane_ids == ("a", "b")
    assert "transformed_distribution_verify" in " ".join(decision.reasons)


def test_scheduler_wiring_is_planning_only_and_refusal_precedes_mutation():
    tree = ast.parse((ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8"))
    installer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_install_continuous_mtp_router"
    )
    assignments = [
        node
        for node in ast.walk(installer)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_step"
            for target in node.targets
        )
        for node in assignments
    )
    source = ast.get_source_segment(
        (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8"),
        installer,
    )
    assert source is not None
    assert "if not decision.admitted" in source
    assert source.index("if not decision.admitted") < source.index(
        "batch_gen._continuous_mtp_router"
    )
    assert "live token delivery " in source
    assert "is not wired in PR 9" in source


def test_cli_and_scheduler_config_carry_the_default_off_opt_in_by_ast():
    cli_source = (ROOT / "vllm_mlx" / "cli.py").read_text(encoding="utf-8")
    scheduler_source = (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8")
    assert "args.mtp_continuous_batching = config.continuous_batching" in cli_source
    assert (
        'mtp_continuous_batching=getattr(args, "mtp_continuous_batching", False)'
        in cli_source
    )
    assert "mtp_continuous_batching: bool = False" in scheduler_source
    assert (
        'if self.mtp_continuous_batching and self.spec_decode != "mtp"'
        in scheduler_source
    )
