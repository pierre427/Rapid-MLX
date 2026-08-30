"""Model-free multi-turn contracts for continuous self-MTP APC restore."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vllm_mlx.scheduler import _apply_continuous_mtp_apc_restore
from vllm_mlx.spec_decode.mtp.continuous_apc import (
    ContinuousMTPAPCBridge,
    ContinuousMTPAPCNamespace,
)
from vllm_mlx.spec_decode.mtp.continuous_routing import (
    ContinuousMTPIntegrationRoute,
    ContinuousMTPRequestMetadata,
    plan_router_install,
)
from vllm_mlx.spec_decode.mtp.prepared_state import RestoreReason

ROOT = Path(__file__).resolve().parents[1]


class _Cache:
    def __init__(self, offset: int, *, dtype=np.float16, capacity: int | None = None):
        self.offset = offset
        capacity = offset if capacity is None else capacity
        self.state = (
            np.zeros((1, 2, capacity, 4), dtype=dtype),
            np.ones((1, 2, capacity, 4), dtype=dtype),
        )
        self.meta_state = (offset, 256)


class ArraysCache(_Cache):
    pass


class Qwen4ArraysCache(ArraysCache):
    def __init__(self, offset: int):
        super().__init__(offset)
        self.state = self.state + (
            np.zeros((1, 2, 4), dtype=np.float16),
            np.zeros((1, 8), dtype=np.int32),
        )
        self._ple_rollback = None


class QSAKVCache(_Cache):
    def __init__(self, offset: int):
        super().__init__(offset)
        self.index_keys = np.zeros((1, 2, max(1, offset), 4), dtype=np.float16)
        self.pooled_keys = np.zeros((1, 2, max(1, offset), 4), dtype=np.float16)
        self._mtp_shared_topk = None


class _Model:
    batched_mtp_capability = {
        "protocol_version": 1,
        "model_family": "qwen3_5",
        "batch_forward": "mtp_batch_forward",
        "recursive_draft_depth": 2,
        "fixed_membership": True,
        "dynamic_join": True,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    mtp = object()

    def __call__(self, *args, **kwargs):
        return args, kwargs

    def mtp_batch_forward(self, *args, **kwargs):
        return args, kwargs

    def mtp_forward(self, *args, **kwargs):
        return args, kwargs

    def make_mtp_cache(self):
        return []


def _namespace(**changes):
    values = {
        "model_id": "Qwen/Qwen3.8-27B",
        "model_revision": "revision-a",
        "speculative_config": {
            "method": "mtp",
            "continuous_batching": True,
            "kv_cache_dtype": "bf16",
        },
        "tokenizer_fingerprint": "tokenizer-a",
    }
    values.update(changes)
    return ContinuousMTPAPCNamespace(**values)


def _capture(bridge, tokens):
    target = [_Cache(len(tokens))]
    draft = [_Cache(len(tokens) - 1)]
    seed = np.zeros((1, 1, 8), dtype=np.float16)
    sidecar = bridge.capture(
        tokens,
        target,
        (draft, seed),
        captured_at=10.0,
    )
    assert sidecar is not None
    assert bridge.commit(tokens, sidecar)
    return target, draft, seed, sidecar


def _qwen4_namespace(**changes):
    return _namespace(model_family="qwen4_exp", **changes)


def _capture_qwen4(bridge, tokens):
    target = [
        ArraysCache(len(tokens)),
        Qwen4ArraysCache(len(tokens)),
        QSAKVCache(len(tokens)),
    ]
    draft = [_Cache(len(tokens) - 1)]
    seed = np.zeros((1, 1, 8), dtype=np.float16)
    sidecar = bridge.capture(tokens, target, (draft, seed), captured_at=10.0)
    assert sidecar is not None
    assert bridge.commit(tokens, sidecar)
    return target, draft, seed, sidecar


def test_two_turn_restore_reaches_router_with_exact_target_draft_and_seed():
    bridge = ContinuousMTPAPCBridge(_namespace(), min_useful_prefix_tokens=4)
    turn_one = tuple(range(80))
    _capture(bridge, turn_one)

    turn_two = turn_one + (901, 902, 903)
    request = SimpleNamespace(
        prompt_token_ids=list(turn_two),
        # APC compaction may change backing capacity without changing the
        # logical cursor, precision, or topology.
        prompt_cache=[_Cache(len(turn_one), capacity=96)],
        cached_tokens=len(turn_one),
        remaining_tokens=list(turn_two[len(turn_one) :]),
        cache_hit_type="prefix",
    )
    assert _apply_continuous_mtp_apc_restore(request, bridge) is True
    hit = request._continuous_mtp_apc_hit

    install = plan_router_install(
        _Model(), enabled=True, hard_reserve_bytes=0, max_lanes=2
    )
    assert install.router is not None
    decision = install.router.plan(
        [
            ContinuousMTPRequestMetadata(
                lane_id="warm-turn-two",
                uid=1,
                prompt_tokens=turn_two,
                max_tokens=16,
                apc_hit=hit,
            ),
            ContinuousMTPRequestMetadata(
                lane_id="cold-peer",
                uid=2,
                prompt_tokens=(7, 8, 9),
                max_tokens=16,
            ),
        ],
        free_bytes=1,
    )

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    restored = decision.cohort[0]
    assert restored.spec.prompt == (901, 902, 903)
    assert restored.spec.prompt_cache is hit.state.target_cache
    assert restored.spec.mtp_cache is hit.state.mtp_cache
    assert restored.spec.seed_hidden is hit.state.seed_hidden
    assert restored.resume_at == len(turn_one)


def test_missing_sidecar_preserves_target_hit_for_plain_fallback():
    bridge = ContinuousMTPAPCBridge(_namespace(), min_useful_prefix_tokens=4)
    prompt = list(range(12))
    request = SimpleNamespace(
        prompt_token_ids=prompt,
        prompt_cache=[_Cache(8)],
        cached_tokens=8,
        remaining_tokens=prompt[8:],
        cache_hit_type="prefix",
    )

    assert _apply_continuous_mtp_apc_restore(request, bridge) is False
    assert request.prompt_cache is not None
    assert request.cached_tokens == 8
    assert request.remaining_tokens == prompt[8:]
    assert request.cache_hit_type == "prefix"
    assert request._continuous_mtp_apc_hit is None
    assert request._continuous_mtp_apc_refusal == "boundary_mismatch"


def test_precision_layout_mismatch_refuses_joint_restore_but_keeps_target_hit():
    bridge = ContinuousMTPAPCBridge(_namespace(), min_useful_prefix_tokens=4)
    prefix = tuple(range(12))
    _capture(bridge, prefix)
    request = SimpleNamespace(
        prompt_token_ids=list(prefix + (99,)),
        prompt_cache=[_Cache(len(prefix), dtype=np.float32)],
        cached_tokens=len(prefix),
        remaining_tokens=[99],
        cache_hit_type="prefix",
    )

    assert _apply_continuous_mtp_apc_restore(request, bridge) is False
    assert request._continuous_mtp_apc_refusal == RestoreReason.CONFIG_MISMATCH.value
    assert request.prompt_cache is not None
    assert request.cached_tokens == len(prefix)
    assert request._continuous_mtp_apc_hit is None


def test_model_or_config_namespace_mismatch_refuses_restore():
    source = ContinuousMTPAPCBridge(_namespace(), min_useful_prefix_tokens=4)
    prefix = tuple(range(12))
    target, _draft, _seed, sidecar = _capture(source, prefix)

    foreign = ContinuousMTPAPCBridge(
        _namespace(model_revision="revision-b"), min_useful_prefix_tokens=4
    )
    assert foreign.commit(prefix, sidecar)
    restored = foreign.restore(
        prefix + (77,),
        target_cache=target,
        cached_tokens=len(prefix),
        now=11.0,
    )

    assert restored.state is None
    assert restored.eligibility.reason is RestoreReason.MODEL_MISMATCH


def test_windowed_target_or_mtp_cache_is_never_captured():
    class SinkWindowKVCache(_Cache):
        pass

    bridge = ContinuousMTPAPCBridge(_namespace(), min_useful_prefix_tokens=4)
    prefix = tuple(range(12))
    seed = np.zeros((1, 1, 8), dtype=np.float16)

    assert (
        bridge.capture(
            prefix,
            [SinkWindowKVCache(len(prefix))],
            ([_Cache(len(prefix) - 1)], seed),
        )
        is None
    )
    assert (
        bridge.capture(
            prefix,
            [_Cache(len(prefix))],
            ([SinkWindowKVCache(len(prefix) - 1)], seed),
        )
        is None
    )


def test_bridge_is_bounded_and_clear_drops_sidecars():
    bridge = ContinuousMTPAPCBridge(
        _namespace(), max_entries=2, min_useful_prefix_tokens=2
    )
    for start in range(3):
        _capture(bridge, tuple(range(start, start + 4)))
    assert len(bridge) == 2
    bridge.clear()
    assert len(bridge) == 0


def test_qwen4_capture_requires_live_gdn_ple_and_qsa_surfaces():
    bridge = ContinuousMTPAPCBridge(
        _qwen4_namespace(), min_useful_prefix_tokens=4
    )
    prefix = tuple(range(12))
    target, _draft, _seed, sidecar = _capture_qwen4(bridge, prefix)

    assert sidecar.metadata.identity.gdn_state_layout != "absent"
    assert sidecar.metadata.identity.ple_state_layout != "absent"
    assert sidecar.metadata.identity.qsa_state_layout != "absent"
    assert bridge.capture(prefix, target[:1], ([_Cache(11)], _seed)) is None
    assert bridge.capture(prefix, target[:2], ([_Cache(11)], _seed)) is None


def test_qwen4_restore_carries_live_target_surfaces_and_counts_memory():
    bridge = ContinuousMTPAPCBridge(
        _qwen4_namespace(), min_useful_prefix_tokens=4
    )
    prefix = tuple(range(12))
    target, _draft, _seed, _sidecar = _capture_qwen4(bridge, prefix)

    restored = bridge.restore(
        prefix + (99,), target_cache=target, cached_tokens=len(prefix), now=11.0
    )

    assert restored.eligibility.eligible is True
    assert restored.state is not None
    assert restored.state.gdn_state
    assert restored.state.ple_state
    assert restored.state.qsa_state
    assert restored.state.ple_state[0] is target[1]
    assert restored.state.qsa_state[0] is target[2]
    stats = bridge.stats_snapshot()
    assert stats["entries"] == 1
    assert stats["payload_nbytes"] > 0
    assert stats["restore_reasons"]["eligible"] == 1


def test_qwen4_restore_refuses_corrupt_sidecar_without_mutating_target_hit():
    bridge = ContinuousMTPAPCBridge(
        _qwen4_namespace(), min_useful_prefix_tokens=4
    )
    prefix = tuple(range(12))
    target, _draft, _seed, sidecar = _capture_qwen4(bridge, prefix)
    corrupt = replace(sidecar, seed_hidden=np.zeros((1, 2, 8), dtype=np.float16))
    assert bridge.commit(prefix, corrupt)
    request = SimpleNamespace(
        prompt_token_ids=list(prefix + (99,)),
        prompt_cache=target,
        cached_tokens=len(prefix),
        remaining_tokens=[99],
        cache_hit_type="prefix",
    )

    assert _apply_continuous_mtp_apc_restore(request, bridge) is False
    assert request.prompt_cache is target
    assert request.cached_tokens == len(prefix)
    assert request.remaining_tokens == [99]
    assert request.cache_hit_type == "prefix"
    assert request._continuous_mtp_apc_hit is None
    assert request._continuous_mtp_apc_refusal == "boundary_mismatch"


def test_request_attachment_rolls_back_if_any_attribute_write_fails():
    bridge = ContinuousMTPAPCBridge(_namespace(), min_useful_prefix_tokens=4)
    prefix = tuple(range(12))
    target, _draft, _seed, _sidecar = _capture(bridge, prefix)

    class OneShotFailure(SimpleNamespace):
        _armed = True

        def __setattr__(self, name, value):
            if name == "cache_hit_type" and self._armed:
                object.__setattr__(self, "_armed", False)
                raise RuntimeError("injected attachment failure")
            super().__setattr__(name, value)

    request = OneShotFailure()
    request.prompt_token_ids = list(prefix + (99,))
    request.prompt_cache = target
    request.cached_tokens = len(prefix)
    request.remaining_tokens = [99]
    object.__setattr__(request, "cache_hit_type", "prefix")

    assert _apply_continuous_mtp_apc_restore(request, bridge) is False
    assert request.prompt_cache is target
    assert request.cached_tokens == len(prefix)
    assert request.remaining_tokens == [99]
    assert request.cache_hit_type == "prefix"
    assert not hasattr(request, "_continuous_mtp_apc_hit")


def test_scheduler_reached_path_captures_restores_and_clears_sidecars():
    scheduler = (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8")
    driver = (
        ROOT / "vllm_mlx" / "spec_decode" / "mtp" / "continuous_driver.py"
    ).read_text(encoding="utf-8")

    assert "_apply_continuous_mtp_apc_restore(" in scheduler
    assert "mtp_apc_bridge.capture(" in scheduler
    assert "mtp_apc_bridge.commit(" in scheduler
    assert "request._continuous_mtp_cache_tokens = getattr(" in scheduler
    assert 'stats["continuous_mtp_apc"] = bridge.stats_snapshot()' in scheduler
    assert scheduler.count("bridge.clear()") >= 2
    assert "package.lane.token_prefix" in driver
