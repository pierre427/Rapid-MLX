# SPDX-License-Identifier: Apache-2.0
"""Version-gated ragged rollback adapter for mlx-lm 0.31.x caches.

mlx-lm 0.31.3 has scalar cache rollback but no per-row rewind contract.  A
continuous self-MTP verify batch needs exactly that contract because lanes may
accept different numbers of draft tokens.  This module vendors the narrow
control/cache adapter while Rapid remains pinned to ``mlx-lm>=0.31.3,<0.32``.

Installation is explicit and idempotent.  Existing scalar ``trim`` and
``restore_rollback`` methods are never replaced.  Unknown and rotating/windowed
caches fail loudly instead of receiving a uniform rewind.  Quantized caches
are admitted only through their native mlx-lm-unified ragged methods.
The module imports no MLX surface until :func:`install_ragged_cache_rollback`
is called, which keeps its contract testable with pure Python fakes.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class RaggedCacheUnsupportedError(RuntimeError):
    """A cache cannot prove exact per-row rollback under this adapter."""


@dataclass(frozen=True)
class RaggedCacheInstallReport:
    version: str
    patched: tuple[str, ...]
    already_present: tuple[str, ...]


_INSTALL_LOCK = threading.Lock()
_ADAPTER_MARKER = "__rapid_ragged_cache_adapter__"
_UNSET = object()


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise RaggedCacheUnsupportedError(f"cannot parse mlx-lm version {version!r}")
    return tuple(int(part) for part in match.groups())


def _validate_version(version: str) -> None:
    parsed = _version_tuple(version)
    if parsed < (0, 31, 3) or parsed >= (0, 32, 0):
        raise RaggedCacheUnsupportedError(
            f"ragged rollback adapter supports mlx-lm>=0.31.3,<0.32; found {version}"
        )


def _drop_vector(values: Any, batch: int, who: str) -> list[int]:
    if isinstance(values, (bool, int)):
        raise TypeError(f"{who} requires one rollback count per row")
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        raw = list(values)
    except TypeError as exc:
        raise TypeError(f"{who} requires an iterable of row counts") from exc
    if len(raw) != batch:
        raise ValueError(f"{who} got {len(raw)} counts for {batch} rows")
    drops: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{who} requires non-negative integer counts")
        if value < 0:
            raise ValueError(f"{who} received a negative rollback count")
        drops.append(value)
    return drops


class _DefaultArrayOps:
    def __init__(self, cache_module: Any) -> None:
        self.mx = cache_module.mx
        self._dynamic_roll = cache_module.dynamic_roll

    def vector(self, values: list[int]):
        return self.mx.array(values)

    @staticmethod
    def tolist(value: Any) -> list[int]:
        return [int(item) for item in value.tolist()]

    def concat(self, rows: list[Any]):
        return self.mx.concatenate(rows, axis=0)

    def roll_rows(self, value: Any, shifts: Any, *, axis: int, stop: int):
        if value is None or stop <= 0:
            return value
        window = (slice(None),) * axis + (slice(0, stop),)
        shaped = shifts.reshape((-1,) + (1,) * (axis - 1))
        value[window] = self._dynamic_roll(value[window], shaped, axis)
        return value


def _snapshot_rows(
    cache: Any,
    drops: list[int],
    verify_size: int,
    ops: Any,
) -> list[Any]:
    snapshots = getattr(cache, "rollback_state", None)
    if snapshots is None:
        raise RaggedCacheUnsupportedError("recurrent cache has no rollback snapshots")
    batch = int(cache.batch_size)
    if isinstance(snapshots, list):
        if not snapshots:
            raise RaggedCacheUnsupportedError(
                "recurrent rollback snapshot list is empty"
            )
        boundaries = snapshots
        slot_count = len(boundaries[0])
        if any(len(boundary) != slot_count for boundary in boundaries):
            raise RaggedCacheUnsupportedError("recurrent rollback slot counts diverge")
    else:
        boundaries = [snapshots]
        slot_count = len(snapshots)

    selected: list[list[Any]] = [[] for _ in range(slot_count)]
    for row, drop in enumerate(drops):
        if drop == 0:
            source = cache.cache
        elif isinstance(snapshots, list):
            keep = verify_size - drop
            if keep < 1 or keep > len(boundaries):
                raise RaggedCacheUnsupportedError(
                    f"row {row} needs verify boundary {keep}, but only "
                    f"{len(boundaries)} snapshots exist"
                )
            source = boundaries[keep - 1]
        else:
            if drop != 1:
                raise RaggedCacheUnsupportedError(
                    "legacy recurrent snapshot can only rewind one token"
                )
            source = boundaries[0]
        for slot, value in enumerate(source):
            if value is None:
                raise RaggedCacheUnsupportedError(
                    f"recurrent rollback slot {slot} is None for row {row}"
                )
            if not hasattr(value, "shape") or int(value.shape[0]) != batch:
                raise RaggedCacheUnsupportedError(
                    f"recurrent rollback slot {slot} does not cover {batch} rows"
                )
            selected[slot].append(value[row : row + 1])
    return [ops.concat(rows) for rows in selected]


def _arrays_preflight(self, n, *, verify_size: int, validate: bool = True):
    del validate
    if isinstance(verify_size, bool) or not isinstance(verify_size, int):
        raise ValueError("verify_size must be a positive integer")
    if verify_size < 1:
        raise ValueError("verify_size must be a positive integer")
    drops = _drop_vector(n, int(self.batch_size), "ArraysCache.trim_ragged")
    _snapshot_rows(self, drops, verify_size, self._rapid_ragged_ops)
    return drops


def _arrays_trim(self, n, *, verify_size: int, validate: bool = True):
    drops = self.preflight_ragged_trim(n, verify_size=verify_size, validate=validate)
    if max(drops, default=0) == 0:
        self.rollback_state = None
        return drops
    self.cache = _snapshot_rows(self, drops, verify_size, self._rapid_ragged_ops)
    self.rollback_state = None
    return drops


def _batch_kv_preflight(self, n, *, verify_size=None, validate: bool = True):
    del verify_size
    base = self._rapid_ragged_batch_kv_base
    if type(self) is base:
        aux_spec = ()
        aux_hook = None
    else:
        aux_spec = None
        aux_hook = None
        for cls in type(self).__mro__:
            if cls is base:
                break
            if "_trim_ragged_aux" in cls.__dict__:
                aux_hook = cls.__dict__["_trim_ragged_aux"]
                break
            if "_RAGGED_TRIM_AUX_ARRAYS" in cls.__dict__:
                aux_spec = cls.__dict__["_RAGGED_TRIM_AUX_ARRAYS"]
                break
        if aux_spec is None and aux_hook is None:
            raise RaggedCacheUnsupportedError(
                f"{type(self).__name__} extends BatchKVCache without declaring "
                "_RAGGED_TRIM_AUX_ARRAYS or _trim_ragged_aux"
            )
    batch = len(self._rapid_ragged_ops.tolist(self.offset))
    drops = _drop_vector(n, batch, "BatchKVCache.trim_ragged")
    if getattr(self, "_right_padding", None) is not None:
        raise RaggedCacheUnsupportedError(
            "BatchKVCache requires finalize() before ragged rollback"
        )
    if max(drops, default=0) > int(self._idx):
        raise ValueError("ragged rollback exceeds the shared KV cursor")
    if validate:
        offsets = self._rapid_ragged_ops.tolist(self.offset)
        if any(offset - drop < 0 for offset, drop in zip(offsets, drops)):
            raise ValueError("ragged rollback would move a KV row before token zero")
    uniform = min(drops, default=0)
    residual = [drop - uniform for drop in drops]
    if max(residual, default=0) and aux_spec is not None:
        stop = int(self._idx) - uniform
        for name, axis in aux_spec:
            ledger = getattr(self, name, None)
            if ledger is not None and int(ledger.shape[axis]) < stop:
                raise RaggedCacheUnsupportedError(
                    f"{type(self).__name__}.{name} does not reach KV cursor {stop}"
                )
    return drops, uniform, residual, aux_spec, aux_hook


def _batch_kv_trim(self, n, *, verify_size=None, validate: bool = True):
    drops, uniform, residual, aux_spec, aux_hook = self.preflight_ragged_trim(
        n, verify_size=verify_size, validate=validate
    )
    invalidate = getattr(self, "_invalidate_attention_groups", None)
    if callable(invalidate):
        invalidate()
    if uniform:
        self._idx -= uniform
        self.offset = self.offset - uniform
    if max(residual, default=0):
        shifts = self._rapid_ragged_ops.vector(residual)
        self.keys = self._rapid_ragged_ops.roll_rows(
            self.keys, shifts, axis=2, stop=self._idx
        )
        self.values = self._rapid_ragged_ops.roll_rows(
            self.values, shifts, axis=2, stop=self._idx
        )
        if aux_hook is not None:
            aux_hook(
                self,
                shifts,
                stop=self._idx,
                array_ops=self._rapid_ragged_ops,
            )
        else:
            for name, axis in aux_spec:
                ledger = getattr(self, name, None)
                if ledger is not None:
                    setattr(
                        self,
                        name,
                        self._rapid_ragged_ops.roll_rows(
                            ledger, shifts, axis=axis, stop=self._idx
                        ),
                    )
        self.left_padding = self.left_padding + shifts
        self.offset = self.offset - shifts
    return drops


def _qsa_preflight(self, n, *, verify_size=None, validate: bool = True):
    del verify_size, validate
    drops = _drop_vector(n, len(self._offsets), "QSAIndexCache.trim_ragged")
    if (
        getattr(self, "_right_padding", None) is not None
        or getattr(self, "lengths", None) is not None
    ):
        raise RaggedCacheUnsupportedError(
            "QSAIndexCache requires finalize() before ragged rollback"
        )
    refused = [
        row
        for row, (offset, drop) in enumerate(zip(self._offsets, drops))
        if not self._can_trim_row(offset, drop)
    ]
    if refused:
        raise RaggedCacheUnsupportedError(
            f"QSA raw-ring history cannot rewind rows {refused} by the requested counts"
        )
    return drops


def _qsa_trim(self, n, *, verify_size=None, validate: bool = True):
    drops = self.preflight_ragged_trim(n, verify_size=verify_size, validate=validate)
    self._offsets = [offset - drop for offset, drop in zip(self._offsets, drops)]
    self._compressed_counts = [
        offset // self.compress_ratio for offset in self._offsets
    ]
    return drops


def _qwen4_preflight(self, n, *, verify_size: int, validate: bool = True):
    if getattr(self, "_rollback_slots", None):
        raise RaggedCacheUnsupportedError(
            "Qwen4 state cache has partially staged PLE/GDN rollback slots"
        )
    return _arrays_preflight(self, n, verify_size=verify_size, validate=validate)


def _mark(function):
    setattr(function, _ADAPTER_MARKER, True)
    return function


for _function in (
    _arrays_preflight,
    _arrays_trim,
    _batch_kv_preflight,
    _batch_kv_trim,
    _qsa_preflight,
    _qsa_trim,
    _qwen4_preflight,
):
    _mark(_function)


def _patch_specs(cache_module, qwen4_state_cls, qsa_cls, ops):
    arrays = cache_module.ArraysCache
    batch_kv = cache_module.BatchKVCache
    specs = [
        (arrays, "preflight_ragged_trim", _arrays_preflight),
        (arrays, "trim_ragged", _arrays_trim),
        (batch_kv, "preflight_ragged_trim", _batch_kv_preflight),
        (batch_kv, "trim_ragged", _batch_kv_trim),
    ]
    if qwen4_state_cls is not None:
        specs.append((qwen4_state_cls, "preflight_ragged_trim", _qwen4_preflight))
    if qsa_cls is not None:
        specs.extend(
            [
                (qsa_cls, "preflight_ragged_trim", _qsa_preflight),
                (qsa_cls, "trim_ragged", _qsa_trim),
            ]
        )
    classes = {arrays, batch_kv, qwen4_state_cls, qsa_cls} - {None}
    return specs, classes


def install_ragged_cache_rollback(
    *,
    mlx_lm_version: str | None = None,
    cache_module: Any = None,
    qwen4_state_cls: Any = _UNSET,
    qsa_cls: Any = _UNSET,
    array_ops: Any = None,
) -> RaggedCacheInstallReport:
    """Install exact per-row rollback on the supported 0.31.x cache classes."""

    version = mlx_lm_version or importlib.metadata.version("mlx-lm")
    _validate_version(version)
    if cache_module is None:
        from mlx_lm.models import cache as cache_module
    if qwen4_state_cls is _UNSET or qsa_cls is _UNSET:
        from vllm_mlx.models.qwen4_exp_cache import (
            QSAIndexCache,
            Qwen4ExpStateCache,
        )

        if qwen4_state_cls is _UNSET:
            qwen4_state_cls = Qwen4ExpStateCache
        if qsa_cls is _UNSET:
            qsa_cls = QSAIndexCache
    arrays_cls = getattr(cache_module, "ArraysCache", None)
    batch_kv_cls = getattr(cache_module, "BatchKVCache", None)
    if not isinstance(arrays_cls, type) or not isinstance(batch_kv_cls, type):
        raise RaggedCacheUnsupportedError(
            "mlx-lm cache module lacks ArraysCache or BatchKVCache"
        )
    for label, cls in (
        ("Qwen4ExpStateCache", qwen4_state_cls),
        ("QSAIndexCache", qsa_cls),
    ):
        if cls is not None and (
            not isinstance(cls, type) or not issubclass(cls, arrays_cls)
        ):
            raise RaggedCacheUnsupportedError(
                f"{label} is not an ArraysCache subclass from this mlx-lm runtime"
            )
    ops = array_ops or _DefaultArrayOps(cache_module)
    specs, classes = _patch_specs(cache_module, qwen4_state_cls, qsa_cls, ops)

    patched: list[str] = []
    already: list[str] = []
    with _INSTALL_LOCK:
        # Preflight every method before mutating any class.
        for cls, name, _ in specs:
            existing = cls.__dict__.get(name)
            if existing is not None and not getattr(existing, _ADAPTER_MARKER, False):
                raise RaggedCacheUnsupportedError(
                    f"refusing to replace existing {cls.__name__}.{name}"
                )
        for cls in classes:
            existing_ops = cls.__dict__.get("_rapid_ragged_ops")
            if existing_ops is not None and existing_ops is not ops:
                raise RaggedCacheUnsupportedError(
                    f"{cls.__name__} already uses a different ragged array adapter"
                )
        for cls in classes:
            if "_rapid_ragged_ops" not in cls.__dict__:
                cls._rapid_ragged_ops = ops
        arrays = cache_module.ArraysCache
        batch_kv = cache_module.BatchKVCache
        if "_rapid_ragged_batch_kv_base" not in batch_kv.__dict__:
            batch_kv._rapid_ragged_batch_kv_base = batch_kv
        if "rollback_state" not in arrays.__dict__:
            arrays.rollback_state = None
        for cls, name, function in specs:
            label = f"{cls.__name__}.{name}"
            if name in cls.__dict__:
                already.append(label)
            else:
                setattr(cls, name, function)
                patched.append(label)
    return RaggedCacheInstallReport(version, tuple(patched), tuple(already))


def _cache_children(cache: Any) -> tuple[Any, ...]:
    children = getattr(cache, "caches", None)
    if children is not None:
        return tuple(children)
    if isinstance(cache, (list, tuple)):
        return tuple(cache)
    return ()


def _reject_unsupported_shape(cache: Any) -> None:
    name = type(cache).__name__.lower()
    if "rotating" in name or "window" in name:
        raise RaggedCacheUnsupportedError(
            f"windowed cache {type(cache).__name__} has no supported ragged contract"
        )


def _call_ragged_method(
    method: Any, values: list[int], *, verify_size: int, validate: bool
):
    """Call a cache's ragged preflight/trim, forwarding ``verify_size`` only if
    it is accepted.  mlx-lm's cache classes expose ``(n, *, validate)``; some
    Rapid/test caches also take a ``verify_size`` hint.  Introspecting keeps one
    adapter working across both without assuming an ABI the real cache lacks."""
    try:
        accepts_verify = "verify_size" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        accepts_verify = False
    if accepts_verify:
        return method(values, verify_size=verify_size, validate=validate)
    return method(values, validate=validate)


def preflight_ragged_cache(
    cache: Any,
    drops: Iterable[int],
    *,
    verify_size: int,
    validate: bool = True,
) -> list[int]:
    """Preflight a cache tree without mutating any member."""

    values = list(drops)
    children = _cache_children(cache)
    if children:
        for child in children:
            preflight_ragged_cache(
                child, values, verify_size=verify_size, validate=validate
            )
        return values
    _reject_unsupported_shape(cache)
    preflight = getattr(cache, "preflight_ragged_trim", None)
    if not callable(preflight):
        raise RaggedCacheUnsupportedError(
            f"cache {type(cache).__name__} has no ragged rollback adapter"
        )
    result = _call_ragged_method(
        preflight, values, verify_size=verify_size, validate=validate
    )
    return result[0] if isinstance(result, tuple) and len(result) == 5 else result


def trim_ragged_cache(
    cache: Any,
    drops: Iterable[int],
    *,
    verify_size: int,
    validate: bool = True,
) -> list[int]:
    """Atomically preflight a cache tree, then apply the per-row rewind."""

    values = list(drops)
    preflight_ragged_cache(cache, values, verify_size=verify_size, validate=validate)

    def apply(node: Any) -> None:
        children = _cache_children(node)
        if children:
            for child in children:
                apply(child)
            return
        _call_ragged_method(
            node.trim_ragged, values, verify_size=verify_size, validate=False
        )

    apply(cache)
    return values


__all__ = [
    "RaggedCacheInstallReport",
    "RaggedCacheUnsupportedError",
    "install_ragged_cache_rollback",
    "preflight_ragged_cache",
    "trim_ragged_cache",
]
