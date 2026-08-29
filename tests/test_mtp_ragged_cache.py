# SPDX-License-Identifier: Apache-2.0
"""Pure/mock coverage for the mlx-lm 0.31.x ragged rollback adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx.spec_decode.mtp.ragged_cache import (
    RaggedCacheUnsupportedError,
    install_ragged_cache_rollback,
    preflight_ragged_cache,
    trim_ragged_cache,
)


class Rows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(self.rows), 1)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Rows(self.rows[item])
        return self.rows[item]

    def __eq__(self, other):
        return isinstance(other, Rows) and self.rows == other.rows


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)

    def __sub__(self, other):
        values = (
            other.values if isinstance(other, Vector) else [other] * len(self.values)
        )
        return Vector([left - right for left, right in zip(self.values, values)])

    def __add__(self, other):
        values = (
            other.values if isinstance(other, Vector) else [other] * len(self.values)
        )
        return Vector([left + right for left, right in zip(self.values, values)])


class FakeOps:
    def __init__(self):
        self.rolls = []

    @staticmethod
    def vector(values):
        return Vector(values)

    @staticmethod
    def tolist(value):
        return value.tolist()

    @staticmethod
    def concat(rows):
        result = []
        for row in rows:
            result.extend(row.rows)
        return Rows(result)

    def roll_rows(self, value, shifts, *, axis, stop):
        self.rolls.append((value, shifts.tolist(), axis, stop))
        return ("rolled", value, tuple(shifts.tolist()), axis, stop)


class FakeArraysCache:
    rollback_state = None

    def __init__(self, cache):
        self.cache = cache
        self.rollback_state = None

    @property
    def batch_size(self):
        return self.cache[0].shape[0]

    def trim(self, count):
        return count


class FakeBatchKVCache:
    def __init__(self):
        self.keys = "keys"
        self.values = "values"
        self.offset = Vector([8, 6])
        self.left_padding = Vector([0, 2])
        self._idx = 8
        self._right_padding = None

    def trim(self, count):
        self._idx -= count
        self.offset = self.offset - count
        return count

    def _invalidate_attention_groups(self):
        self.invalidations = getattr(self, "invalidations", 0) + 1


class FakeQwen4StateCache(FakeArraysCache):
    def __init__(self, cache):
        super().__init__(cache)
        self._rollback_slots = None

    def restore_rollback(self, n_to_drop, verify_size):
        self.scalar_restore = (n_to_drop, verify_size)


class FakeQSAIndexCache(FakeArraysCache):
    def __init__(self):
        super().__init__([Rows(["raw-a", "raw-b"])])
        self._offsets = [9, 8]
        self._compressed_counts = [2, 2]
        self.compress_ratio = 4
        self._right_padding = None
        self.lengths = None

    @staticmethod
    def _can_trim_row(offset, count):
        if count < 0 or count > offset:
            return False
        if count == 0:
            return True
        remainder = offset % 4
        available = remainder if remainder else min(4, offset)
        return count <= available

    def trim(self, count):
        if not all(self._can_trim_row(offset, count) for offset in self._offsets):
            return 0
        self._offsets = [offset - count for offset in self._offsets]
        return count


def _install(ops=None, **kwargs):
    arrays = kwargs.pop("arrays", type("TestArraysCache", (FakeArraysCache,), {}))
    batch_kv = kwargs.pop("batch_kv", type("TestBatchKVCache", (FakeBatchKVCache,), {}))
    module = kwargs.pop(
        "module", SimpleNamespace(ArraysCache=arrays, BatchKVCache=batch_kv)
    )
    return install_ragged_cache_rollback(
        mlx_lm_version=kwargs.pop("version", "0.31.3"),
        cache_module=module,
        qwen4_state_cls=kwargs.pop("qwen", None),
        qsa_cls=kwargs.pop("qsa", None),
        array_ops=ops or FakeOps(),
        **kwargs,
    )


@pytest.mark.parametrize("version", ["0.31.2", "0.32.0", "main"])
def test_installer_is_strictly_version_gated(version):
    with pytest.raises(RaggedCacheUnsupportedError):
        _install(version=version)


def test_install_is_idempotent_and_preserves_scalar_methods():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    class Qwen(FakeQwen4StateCache, Arrays):
        pass

    module = SimpleNamespace(ArraysCache=Arrays, BatchKVCache=Batch)
    ops = FakeOps()
    scalar_kv = Batch.trim
    scalar_qwen = Qwen.restore_rollback
    first = _install(ops, module=module, qwen=Qwen)
    second = _install(ops, module=module, qwen=Qwen)

    assert first.patched
    assert not second.patched
    assert second.already_present
    assert Batch.trim is scalar_kv
    assert Qwen.restore_rollback is scalar_qwen


def test_installer_refuses_to_replace_an_unknown_existing_method():
    class ConflictingArrays(FakeArraysCache):
        def trim_ragged(self, values, **kwargs):
            return values

    with pytest.raises(RaggedCacheUnsupportedError, match="refusing to replace"):
        _install(arrays=ConflictingArrays, qwen=None, qsa=None)


def test_installer_rejects_qwen_cache_from_a_different_runtime():
    class Arrays(FakeArraysCache):
        pass

    class ForeignQwen:
        pass

    with pytest.raises(RaggedCacheUnsupportedError, match="not an ArraysCache"):
        _install(arrays=Arrays, qwen=ForeignQwen, qsa=None)


def test_batch_kv_splits_uniform_cursor_move_from_residual_row_roll():
    ops = FakeOps()

    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(ops, arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    cache = Batch()
    assert cache.trim_ragged([1, 3], verify_size=3) == [1, 3]

    assert cache._idx == 7
    assert cache.offset.tolist() == [7, 3]
    assert cache.left_padding.tolist() == [0, 4]
    assert len(ops.rolls) == 2
    assert ops.rolls[0][1:] == ([0, 2], 2, 7)
    assert cache.invalidations == 1


def test_unknown_batch_kv_subclass_cannot_silently_inherit_adapter():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)

    class UndeclaredLedgerCache(Batch):
        pass

    with pytest.raises(RaggedCacheUnsupportedError, match="without declaring"):
        UndeclaredLedgerCache().preflight_ragged_trim([1, 2], verify_size=2)


def test_declared_batch_kv_auxiliary_ledger_rolls_with_kv():
    ops = FakeOps()

    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(ops, arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)

    class DeclaredLedgerCache(Batch):
        _RAGGED_TRIM_AUX_ARRAYS = (("ledger", 1),)

        def __init__(self):
            super().__init__()
            self.ledger = type("Ledger", (), {"shape": (2, 8)})()

    cache = DeclaredLedgerCache()
    cache.trim_ragged([1, 2], verify_size=2)
    assert ops.rolls[-1][0] is not None
    assert ops.rolls[-1][2:] == (1, 7)


def test_batch_kv_preflight_rejects_pending_padding_and_overtrim():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    cache = Batch()
    cache._right_padding = Vector([0, 1])
    with pytest.raises(RaggedCacheUnsupportedError, match="finalize"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)
    cache._right_padding = None
    with pytest.raises(ValueError, match="before token zero"):
        cache.preflight_ragged_trim([1, 7], verify_size=7)


def test_arrays_cache_selects_each_rows_exact_verify_boundary():
    class Arrays(FakeArraysCache):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=None)
    cache = Arrays([Rows(["a-live", "b-live"]), Rows(["A-live", "B-live"])])
    cache.rollback_state = [
        [Rows(["a-keep1", "b-keep1"]), Rows(["A-keep1", "B-keep1"])],
        [Rows(["a-keep2", "b-keep2"]), Rows(["A-keep2", "B-keep2"])],
    ]

    assert cache.trim_ragged([2, 0], verify_size=3) == [2, 0]
    assert cache.cache == [
        Rows(["a-keep1", "b-live"]),
        Rows(["A-keep1", "B-live"]),
    ]
    assert cache.rollback_state is None


def test_qwen4_refuses_partially_staged_atomic_state():
    class Arrays(FakeArraysCache):
        pass

    class Qwen(FakeQwen4StateCache, Arrays):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=Qwen, qsa=None)
    cache = Qwen([Rows(["a", "b"])])
    cache.rollback_state = [[Rows(["a0", "b0"])]]
    cache._rollback_slots = {0: [Rows(["staged-a", "staged-b"])]}
    with pytest.raises(RaggedCacheUnsupportedError, match="partially staged"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)


def test_qsa_rewinds_logical_rows_only_within_retained_raw_group():
    class Arrays(FakeArraysCache):
        pass

    class QSA(FakeQSAIndexCache, Arrays):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=QSA)
    cache = QSA()
    assert cache.trim_ragged([1, 3], verify_size=4) == [1, 3]
    assert cache._offsets == [8, 5]
    assert cache._compressed_counts == [2, 1]

    cache._offsets = [9, 8]
    with pytest.raises(RaggedCacheUnsupportedError, match="raw-ring history"):
        cache.preflight_ragged_trim([2, 1], verify_size=3)


def test_public_preflight_accepts_native_quantized_ragged_contract():
    events = []

    class BatchQuantizedKVCache:
        def preflight_ragged_trim(self, values, *, validate=True):
            events.append(("preflight", tuple(values), validate))
            return list(values)

        def trim_ragged(self, values, *, validate=True):
            self.preflight_ragged_trim(values, validate=validate)
            events.append(("trim", tuple(values), validate))
            return list(values)

    cache = BatchQuantizedKVCache()
    assert trim_ragged_cache(cache, [1, 0], verify_size=2) == [1, 0]
    assert events == [
        ("preflight", (1, 0), True),
        ("preflight", (1, 0), False),
        ("trim", (1, 0), False),
    ]


def test_public_preflight_fails_loud_for_windowed_cache_family():
    cache = type("BatchRotatingKVCache", (), {})()
    with pytest.raises(RaggedCacheUnsupportedError, match="windowed"):
        preflight_ragged_cache(cache, [1, 1], verify_size=2)


def test_cache_tree_preflights_every_member_before_mutating_anything():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    first = Batch()
    unsupported = type("UnknownCache", (), {})()
    tree = SimpleNamespace(caches=(first, unsupported))

    with pytest.raises(RaggedCacheUnsupportedError, match="no ragged"):
        trim_ragged_cache(tree, [1, 1], verify_size=2)
    assert first._idx == 8
    assert first.offset.tolist() == [8, 6]


def test_supported_cache_tree_applies_after_successful_preflight():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    first, second = Batch(), Batch()
    tree = SimpleNamespace(caches=(first, second))
    assert trim_ragged_cache(tree, [1, 1], verify_size=2) == [1, 1]
    assert first._idx == second._idx == 7
