"""Model-free and AST tests for PR 9 continuous-MTP routing."""

from __future__ import annotations

import ast
import logging
from collections import deque
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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
    assert default.allow_dynamic_membership is False
    assert enabled is not None and enabled.continuous_batching is True

    dynamic = parse_speculative_config(
        '{"method":"mtp","continuous_batching":true,"allow_dynamic_membership":true}'
    )
    assert dynamic is not None and dynamic.allow_dynamic_membership is True

    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","continuous_batching":1}')
    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","allow_dynamic_membership":1}')
    with pytest.raises(SpeculativeConfigError, match="requires continuous_batching"):
        parse_speculative_config('{"method":"mtp","allow_dynamic_membership":true}')
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


def test_install_plan_threads_dynamic_membership_descriptor_attestation():
    attested = plan_router_install(
        _Model(),
        enabled=True,
        allow_dynamic_membership=True,
        hard_reserve_bytes=0,
    )
    assert attested.admitted is True
    assert attested.router is not None
    assert attested.router.config.allow_dynamic_membership is True
    assert attested.router.runtime.capabilities.dynamic_membership is True

    qwen4_model = _Model()
    qwen4_model.batched_mtp_capability = _descriptor("qwen4_exp")
    qwen4 = plan_router_install(
        qwen4_model,
        enabled=True,
        allow_dynamic_membership=True,
        hard_reserve_bytes=0,
    )
    assert qwen4.admitted is True
    assert qwen4.router is not None
    assert qwen4.router.runtime.capabilities.dynamic_membership is False
    refused = qwen4.router.plan([_request("a", 1), _request("b", 2)], free_bytes=1)
    assert refused.route is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert "dynamic_membership" in " ".join(refused.reasons)


def test_supported_requests_build_an_immutable_fixed_cohort_plan():
    router = _router()

    decision = router.plan([_request("a", 1), _request("b", 2)], free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert decision.live_token_delivery is False
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert [lane.spec.uid for lane in decision.cohort] == [1, 2]
    assert [lane.spec.num_draft for lane in decision.cohort] == [2, 2]
    assert all(lane.prepared_state is None for lane in decision.cohort)


def test_b1_is_explicit_to_continuous_integration_and_default_stays_batch_first():
    default_router = _router()
    default = default_router.plan([_request("a", 1)], free_bytes=1)
    assert default.route is ContinuousMTPIntegrationRoute.LEGACY_MTP

    b1_router = _router(max_lanes=1, min_batch_lanes=1)
    b1 = b1_router.plan([_request("a", 1)], free_bytes=1)
    assert b1.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert [lane.spec.uid for lane in b1.cohort] == [1]
    assert b1.admission is not None
    assert b1.admission.batched_lane_ids == ("a",)

    refused = b1_router.plan(
        [_request("bad", 2, cache_ready=False)],
        free_bytes=1,
    )
    assert refused.route is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert refused.cohort == ()
    assert refused.legacy_lane_ids == ("bad",)
    assert "cache is not ready" in " ".join(refused.reasons)


def test_live_integration_executes_b1_continuous_driver(monkeypatch, caplog):
    import vllm_mlx.scheduler as scheduler
    from vllm_mlx.spec_decode.mtp import continuous_driver, continuous_runtime

    created = []

    class _Driver:
        has_work = True
        closed = False
        has_pending_responses = False
        dynamic_membership = False
        lane_uids = (1,)
        pending_join_uids = ()

        def __init__(self):
            self.next_calls = 0

        def next(self):
            self.next_calls += 1
            return ["continuous-b1"]

        def take_terminal_detaches(self):
            return ()

        def resume_turnover(self):
            raise AssertionError("B1 driver unexpectedly entered turnover")

        def discard_all(self):
            return ()

        def remove_uids(self, _uids):
            return ()

    driver = _Driver()

    def _create(specs, runtime, **kwargs):
        created.append((tuple(specs), runtime, kwargs))
        return driver

    monkeypatch.setattr(
        continuous_driver.ContinuousMTPDriver,
        "create",
        staticmethod(_create),
    )
    runtime = object()
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    class _Response:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _GenerationBatch:
        Response = _Response

    class _Cache:
        nbytes = 0
        offset = 0

    class _BatchGenerator:
        def __init__(self):
            self._generation_batch = _GenerationBatch()
            self._unprocessed_sequences = deque(
                [(1, ((11, 12),), 8, [_Cache()], [], None, [], None)]
            )
            self.fallback_calls = 0

        def next(self):
            self.fallback_calls += 1
            return [], []

        def close(self):
            return None

        def remove(self, _uids, *args, **kwargs):
            del args, kwargs
            return {}

    batch_gen = _BatchGenerator()
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=0.0, stop=None),
        has_tools=False,
    )
    config = SimpleNamespace(
        kv_cache_quantization=False,
        kv_cache_turboquant=False,
        kv_cache_dtype="bf16",
        max_num_seqs=1,
        completion_batch_size=1,
        mtp_max_lanes=1,
        mtp_continuous_batching=True,
        use_paged_cache=False,
        mtp_allow_dynamic_membership=False,
        mtp_speculation_rollback=True,
        mtp_lane_transient_gib=1.76,
    )

    with caplog.at_level(logging.INFO, logger="vllm_mlx.scheduler"):
        installed = scheduler._install_continuous_mtp_router(
            batch_gen,
            _Model(),
            config,
            requests={"req-1": request},
            uid_to_request_id={1: "req-1"},
            free_bytes_getter=lambda: 64 * 1024**3,
        )
        assert installed is True
        assert batch_gen.next() == ([], [])
        assert batch_gen.next() == ([], ["continuous-b1"])

    assert len(created) == 1
    assert [spec.uid for spec in created[0][0]] == [1]
    assert created[0][1] is runtime
    assert driver.next_calls == 1
    assert not batch_gen._unprocessed_sequences
    assert (
        "[MTP-continuous] admitted continuous cohort "
        "route=continuous_planned lanes=1 draft_depth=2"
    ) in caplog.text


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


def test_scheduler_wiring_diverts_next_and_refusal_precedes_mutation():
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
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "next"
            and isinstance(target.value, ast.Name)
            and target.value.id == "batch_gen"
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
    assert "ContinuousMTPDriver.create" in source
    assert "continuous_responses + list(fallback_responses)" in source
    assert "scheduler_owned_termination" in source
    assert "bool(params.stop) or bool(request.has_tools)" in source
    assert source.index("_remove_queued(selected)") < source.index(
        "raw = original_next()"
    )
    assert "prompt_cache=package.target_cache" in (
        ROOT / "vllm_mlx" / "spec_decode" / "mtp" / "continuous_driver.py"
    ).read_text(encoding="utf-8")

    scheduler_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    create_generator = next(
        node
        for node in scheduler_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_batch_generator"
    )
    create_source = ast.get_source_segment(
        (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8"),
        create_generator,
    )
    assert create_source is not None
    router_call = create_source.index("_install_continuous_mtp_router(")
    vendored_call = create_source.index("_install_mtp_vendored(", router_call)
    assert router_call < vendored_call


def test_cli_and_scheduler_config_carry_the_default_off_opt_in_by_ast():
    cli_source = (ROOT / "vllm_mlx" / "cli.py").read_text(encoding="utf-8")
    scheduler_source = (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8")
    assert "args.mtp_continuous_batching = config.continuous_batching" in cli_source
    assert (
        'mtp_continuous_batching=getattr(args, "mtp_continuous_batching", False)'
        in cli_source
    )
    assert "mtp_continuous_batching: bool = False" in scheduler_source
    assert "mtp_allow_dynamic_membership: bool = False" in scheduler_source
    assert (
        'if self.mtp_continuous_batching and self.spec_decode != "mtp"'
        in scheduler_source
    )
    assert "mtp_allow_dynamic_membership=getattr(" in cli_source
    assert '"--self-mtp-max-lanes"' in cli_source
    assert '"--self-mtp-lane-transient-gib"' in cli_source
    assert "mtp_max_lanes: int = 16" in scheduler_source
    assert "mtp_lane_transient_gib: float = 1.76" in scheduler_source


def test_scheduler_config_validates_continuous_mtp_admission_controls():
    from vllm_mlx.scheduler import SchedulerConfig

    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_max_lanes=16,
        mtp_lane_transient_gib=3.1,
    )
    assert config.mtp_max_lanes == 16
    assert config.mtp_lane_transient_gib == 3.1

    with pytest.raises(ValueError, match="mtp_max_lanes"):
        SchedulerConfig(mtp_max_lanes=0)
    with pytest.raises(ValueError, match="mtp_lane_transient_gib"):
        SchedulerConfig(mtp_lane_transient_gib=0)
    with pytest.raises(ValueError, match="mtp_lane_transient_gib"):
        SchedulerConfig(mtp_lane_transient_gib=float("inf"))


def test_live_router_charges_k2_transient_and_compute_cap_by_source():
    scheduler_source = (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8")
    assert 'getattr(config, "mtp_max_lanes", 16)' in scheduler_source
    assert 'getattr(config, "mtp_lane_transient_gib", 1.76)' in scheduler_source
    assert "bytes_per_draft_token=int(" in scheduler_source
