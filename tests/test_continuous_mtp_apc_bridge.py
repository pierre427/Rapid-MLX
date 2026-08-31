"""Model-free multi-turn contracts for continuous self-MTP APC restore."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
        "quantized_cache": True,
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


@pytest.mark.requires_mlx
@pytest.mark.parametrize("bits", (4, 8))
def test_real_qwen4_quantized_qsa_apc_restore_stays_packed_and_replays(bits):
    """The production Qwen4 owners survive APC without a bf16 shadow copy."""

    import copy

    import mlx.core as mx
    from mlx_lm.models.cache import CacheList, KVCache, QuantizedKVCache

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    from vllm_mlx.models.qwen4_exp_cache import (
        QSAIndexCache,
        Qwen4ExpStateCache,
    )

    prefix_length = 12
    prefix = tuple(range(prefix_length))

    def kv_values(length, *, shift=0):
        values = mx.arange(2 * length * 64, dtype=mx.float32).reshape(
            1, 2, length, 64
        )
        return ((values + shift) / 37.0 - 3.0).astype(mx.bfloat16)

    def quantized_kv(length):
        cache = QuantizedKVCache(group_size=32, bits=bits)
        values = kv_values(length)
        cache.update_and_fetch(values, -values * 1.7)
        return cache

    def plain_kv(length):
        cache = KVCache()
        values = kv_values(length)
        cache.update_and_fetch(values, -values * 1.7)
        return cache

    def qsa_index(length):
        cache = QSAIndexCache(compress_ratio=2)
        values = (
            mx.arange(length * 64, dtype=mx.float32).reshape(1, length, 64)
            / 19.0
        ).astype(mx.bfloat16)
        cache.update(values, lambda group, start: group + start)
        return cache

    def recurrent_state(slots):
        cache = Qwen4ExpStateCache(size=slots)
        cache.cache = [
            mx.full((1, 2, 4), slot + 0.25, dtype=mx.bfloat16)
            for slot in range(slots)
        ]
        return cache

    def as_numpy(value):
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float32)
        return np.array(value)

    target_qsa = qsa_index(prefix_length)
    target = [
        recurrent_state(2),
        recurrent_state(4),
        CacheList(quantized_kv(prefix_length), target_qsa),
    ]
    # This mirrors the shipped runtime: target attention K/V is quantized;
    # the one-layer MTP side cache remains its native KVCache + QSA ledger.
    draft = [CacheList(plain_kv(prefix_length - 1), qsa_index(prefix_length - 1))]
    seed = mx.arange(64, dtype=mx.float32).reshape(1, 1, 64).astype(mx.bfloat16)
    mx.eval(seed, [cache.state for cache in target], [cache.state for cache in draft])

    namespace = _qwen4_namespace(
        speculative_config={
            "method": "mtp",
            "continuous_batching": True,
            "kv_cache_dtype": f"int{bits}",
            "kv_cache_quantization": True,
            "kv_cache_quantization_bits": bits,
            "kv_cache_quantization_group_size": 32,
        }
    )
    bridge = ContinuousMTPAPCBridge(namespace, min_useful_prefix_tokens=4)
    sidecar = bridge.capture(prefix, target, (draft, seed), captured_at=10.0)
    assert sidecar is not None

    packed_before = target[2].caches[0].keys
    scales = np.array(packed_before[1].astype(mx.float32))
    assert not np.allclose(scales, 1.0), "fixture must exercise non-unit scales"

    apc = MemoryAwarePrefixCache(
        object(),
        MemoryCacheConfig(
            max_memory_mb=64,
            max_entries=8,
            kv_quantize=True,
            kv_bits=bits,
            kv_group_size=32,
            kv_min_quantize_tokens=0,
            hybrid_reuse_max_entries=8,
        ),
    )
    assert apc.store(list(prefix), target)
    assert bridge.commit(prefix, sidecar)

    request_tokens = prefix + (999,)
    restored_target, remaining = apc.fetch(list(request_tokens))
    assert remaining == [999]
    assert restored_target is not None
    restored_kv, restored_qsa = restored_target[2].caches
    assert type(restored_kv).__name__ == "QuantizedKVCache"
    assert isinstance(restored_kv.keys, (list, tuple)) and len(restored_kv.keys) == 3
    assert restored_kv.offset == prefix_length
    assert restored_qsa.offset == prefix_length
    assert restored_qsa._compressed_count == target_qsa._compressed_count
    for actual, expected in zip(restored_kv.keys, packed_before):
        np.testing.assert_array_equal(
            as_numpy(actual[..., :prefix_length, :]),
            as_numpy(expected[..., :prefix_length, :]),
        )
    np.testing.assert_array_equal(
        np.array(restored_qsa.raw_ring.astype(mx.float32)),
        np.array(target_qsa.raw_ring.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(restored_qsa.state[1].astype(mx.float32)),
        np.array(target_qsa.state[1].astype(mx.float32)),
    )

    restored = bridge.restore(
        request_tokens,
        target_cache=restored_target,
        cached_tokens=prefix_length,
        now=11.0,
    )
    assert restored.eligibility.eligible is True
    assert restored.state is not None
    assert restored.state.target_cache is restored_target
    assert restored.state.qsa_state[0] is restored_qsa
    assert type(restored.state.mtp_cache[0].caches[0]).__name__ == "KVCache"

    # A one-token continuation from the warm packed state must match an
    # independently copied reference, including the QSA ledger transition.
    reference = copy.deepcopy(restored_target)
    next_kv = kv_values(1, shift=10_000)
    next_qsa = (
        mx.arange(64, dtype=mx.float32).reshape(1, 1, 64) / 11.0
    ).astype(mx.bfloat16)
    restored_kv.update_and_fetch(next_kv, -next_kv * 1.7)
    restored_qsa.update(next_qsa, lambda group, start: group + start)
    reference[2].caches[0].update_and_fetch(next_kv, -next_kv * 1.7)
    reference[2].caches[1].update(next_qsa, lambda group, start: group + start)
    mx.eval(restored_kv.state, restored_qsa.state, reference[2].state)
    for actual, expected in zip(restored_kv.keys, reference[2].caches[0].keys):
        np.testing.assert_array_equal(
            as_numpy(actual[..., : prefix_length + 1, :]),
            as_numpy(expected[..., : prefix_length + 1, :]),
        )
    np.testing.assert_array_equal(
        np.array(restored_qsa.state[1].astype(mx.float32)),
        np.array(reference[2].caches[1].state[1].astype(mx.float32)),
    )

    bf16_bridge = ContinuousMTPAPCBridge(
        _qwen4_namespace(
            speculative_config={
                "method": "mtp",
                "continuous_batching": True,
                "kv_cache_dtype": "bf16",
                "kv_cache_quantization": False,
            }
        ),
        min_useful_prefix_tokens=4,
    )
    assert bf16_bridge.commit(prefix, sidecar)
    refused = bf16_bridge.restore(
        request_tokens,
        target_cache=copy.deepcopy(target),
        cached_tokens=prefix_length,
        now=11.0,
    )
    assert refused.state is None
    assert refused.eligibility.reason is RestoreReason.CONFIG_MISMATCH


@pytest.mark.requires_mlx
@pytest.mark.parametrize("bits", (4, 8))
def test_monolithic_qwen4_quantized_qsa_owner_survives_apc_and_exact_trim(
    bits, tmp_path
):
    """Exercise the cache class used by the released Flash-Next checkpoint."""

    import mlx.core as mx
    from mlx_lm.models.qwen4_exp import QSAKVCache, QSAQuantizedKVCache

    from vllm_mlx.memory_cache import (
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
        _load_prompt_cache_compat,
        _save_prompt_cache_compat,
        _trim_cache_offset,
        estimate_kv_cache_memory,
    )
    from vllm_mlx.models.qwen4_exp_cache import Qwen4ExpStateCache

    length = 12
    prefix = tuple(range(length))
    values = (
        mx.arange(2 * length * 64, dtype=mx.float32).reshape(1, 2, length, 64)
        / 23.0
        - 4.0
    ).astype(mx.bfloat16)
    ledger = (
        mx.arange(length * 64, dtype=mx.float32).reshape(1, length, 64)
        / 17.0
    ).astype(mx.bfloat16)
    plain = QSAKVCache()
    plain.update_and_fetch(values, -values)
    plain.update_index_keys(ledger)
    packed = plain.to_quantized(group_size=32, bits=bits)
    assert isinstance(packed, QSAQuantizedKVCache)

    def recurrent(slots):
        cache = Qwen4ExpStateCache(size=slots)
        cache.cache = [
            mx.full((1, 2, 4), i + 0.5, dtype=mx.bfloat16)
            for i in range(slots)
        ]
        return cache

    target = [recurrent(2), recurrent(4), packed]
    draft_plain = QSAKVCache()
    draft_plain.update_and_fetch(values[..., :-1, :], -values[..., :-1, :])
    draft_plain.update_index_keys(ledger[:, :-1])
    seed = mx.zeros((1, 1, 64), dtype=mx.bfloat16)
    mx.eval([cache.state for cache in target], draft_plain.state, seed)

    # Admission must charge both packed triples and the still-native ledger.
    packed_bytes = sum(x.nbytes for x in (*packed.keys, *packed.values))
    assert estimate_kv_cache_memory([packed]) == packed_bytes + ledger.nbytes

    namespace = _qwen4_namespace(
        speculative_config={
            "method": "mtp",
            "continuous_batching": True,
            "kv_cache_dtype": f"int{bits}",
            "kv_cache_quantization": True,
            "kv_cache_quantization_bits": bits,
            "kv_cache_quantization_group_size": 32,
        }
    )
    bridge = ContinuousMTPAPCBridge(namespace, min_useful_prefix_tokens=4)
    sidecar = bridge.capture(
        prefix, target, ([draft_plain], seed), captured_at=10.0
    )
    assert sidecar is not None

    apc = MemoryAwarePrefixCache(
        object(),
        MemoryCacheConfig(
            max_memory_mb=64,
            max_entries=8,
            kv_quantize=True,
            kv_bits=bits,
            kv_group_size=32,
            kv_min_quantize_tokens=0,
            hybrid_reuse_max_entries=8,
        ),
    )
    assert apc.store(list(prefix), target)
    assert bridge.commit(prefix, sidecar)
    restored_target, remaining = apc.fetch([*prefix, 999])
    assert remaining == [999]
    restored = restored_target[-1]
    assert isinstance(restored, QSAQuantizedKVCache)
    assert restored.offset == length
    assert restored.index_keys.shape[1] == length

    persisted = tmp_path / f"qsa-int{bits}.safetensors"
    _save_prompt_cache_compat(str(persisted), [restored], {})
    reloaded = _load_prompt_cache_compat(str(persisted))[0]
    assert isinstance(reloaded, QSAQuantizedKVCache)
    assert reloaded.offset == length
    assert reloaded.index_keys.shape[1] == length
    assert len(reloaded.keys) == 3
    assert len(restored.keys) == 3

    prepared = bridge.restore(
        (*prefix, 999),
        target_cache=restored_target,
        cached_tokens=length,
        now=11.0,
    )
    assert prepared.eligibility.eligible is True
    assert prepared.state.qsa_state[0] is restored

    trimmed = _trim_cache_offset([restored], 3)
    assert trimmed is not None
    assert isinstance(trimmed[0], QSAQuantizedKVCache)
    assert trimmed[0].offset == length - 3
    assert trimmed[0].index_keys.shape[1] == length - 3
    # The retained APC entry is immutable relative to the returned trim copy.
    assert restored.offset == length
    assert restored.index_keys.shape[1] == length


@pytest.mark.requires_mlx
def test_legacy_bare_model_local_cache_name_uses_constrained_resolver(
    monkeypatch, tmp_path
):
    """Load pre-qualified-name files without widening the import allowlist."""

    import mlx.core as mx
    import mlx_lm.models.cache as mlx_cache

    from vllm_mlx.memory_cache import (
        _load_prompt_cache_compat,
        _save_prompt_cache_compat,
    )

    class LegacyModelLocalCache:
        def __init__(self):
            self.payload = mx.arange(4, dtype=mx.float32)
            self.optional = None

        @property
        def state(self):
            return self.payload, self.optional

        @property
        def meta_state(self):
            return ("legacy",)

        @classmethod
        def from_state(cls, state, meta_state):
            assert tuple(meta_state) == ("legacy",)
            obj = cls.__new__(cls)
            obj.payload, obj.optional = state
            return obj

    path = tmp_path / "legacy-bare-model-local.safetensors"
    _save_prompt_cache_compat(str(path), [LegacyModelLocalCache()], {})

    original = mlx_cache._resolve_cache_class

    def resolve(name):
        if name == "LegacyModelLocalCache":
            return LegacyModelLocalCache
        return original(name)

    monkeypatch.setattr(mlx_cache, "_resolve_cache_class", resolve)
    restored = _load_prompt_cache_compat(str(path))[0]

    assert isinstance(restored, LegacyModelLocalCache)
    assert restored.optional is None
    mx.eval(restored.payload)
    assert restored.payload.tolist() == [0.0, 1.0, 2.0, 3.0]


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
