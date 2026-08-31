# SPDX-License-Identifier: Apache-2.0
"""APC sidecars for live continuous self-MTP multi-turn restore.

Rapid's prefix cache remains the sole owner of target KV.  This module keeps
only the state APC cannot represent: the MTP cache, the final-prefix hidden
state, and enough immutable metadata to prove that all three describe the same
token boundary and runtime contract.  A lookup composes the target cache
returned by APC with a matching sidecar; every mismatch refuses only the joint
self-MTP restore, leaving the valid target APC hit available to plain decode.

The bridge is process-local by design.  Persisting MTP sidecars needs a
separate wire-format and durability review; silently mixing them into Rapid's
existing prefix-cache files would make old readers unsafe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .prepared_state import (
    ABSENT_STATE_LAYOUT,
    PreparedMTPState,
    PreparedStateIdentity,
    PreparedStateMetadata,
    RestoreEligibility,
    RestoreReason,
    evaluate_restore,
    fingerprint_config,
    fingerprint_tokens,
    prepare_mtp_state,
)

_WINDOWED_MARKERS = ("rotating", "window", "sink")


@dataclass(frozen=True)
class ContinuousMTPAPCNamespace:
    """Stable identity shared by every sidecar in one scheduler."""

    model_id: str
    model_revision: str
    speculative_config: Mapping[str, Any]
    adapter_id: str | None = None
    tokenizer_fingerprint: str | None = None
    model_family: str | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        # Validate canonicalizability once at construction, not on a request.
        fingerprint_config(self.speculative_config)
        if self.model_family is not None and (
            not isinstance(self.model_family, str) or not self.model_family.strip()
        ):
            raise ValueError("model_family must be None or a non-empty string")
        object.__setattr__(
            self, "speculative_config", MappingProxyType(dict(self.speculative_config))
        )


@dataclass(frozen=True)
class ContinuousMTPAPCSidecar:
    """The state stored beside, but not duplicating, an APC target entry."""

    metadata: PreparedStateMetadata
    mtp_cache: Any
    seed_hidden: Any
    payload_nbytes: int


@dataclass(frozen=True)
class ContinuousMTPAPCRestore:
    """Non-throwing result of composing an APC hit with an MTP sidecar."""

    eligibility: RestoreEligibility
    state: PreparedMTPState | None = None
    expected_identity: PreparedStateIdentity | None = None


@dataclass(frozen=True)
class _PreparedSurfaces:
    """Live architecture state already owned by the target APC cache."""

    gdn_state: tuple[Any, ...] | None
    ple_state: tuple[Any, ...] | None
    qsa_state: tuple[Any, ...] | None


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_layout(value: Any) -> dict[str, Any] | None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    try:
        dimensions: list[Any] = [int(dimension) for dimension in shape]
        # KV backing capacity can be rounded to a step size and then compacted
        # by the owning APC before fetch.  The exact logical cursor is checked
        # separately; normalize only this dynamic sequence axis so compaction
        # cannot masquerade as a precision/topology mismatch.
        if len(dimensions) >= 2:
            dimensions[-2] = "sequence"
        dimensions = tuple(dimensions)
    except (TypeError, ValueError):
        dimensions = (repr(shape),)
    return {
        "kind": "array",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "shape": dimensions,
        "dtype": str(dtype),
    }


def _layout_tree(value: Any, *, seen: set[int] | None = None) -> Any:
    """Describe type/shape/dtype without reading or evaluating array values."""

    if seen is None:
        seen = set()
    array = _array_layout(value)
    if array is not None:
        return array
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "scalar", "type": type(value).__name__}
    value_id = id(value)
    if value_id in seen:
        return {"kind": "cycle", "type": type(value).__qualname__}
    seen.add(value_id)
    try:
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "items": [
                    (str(key), _layout_tree(item, seen=seen))
                    for key, item in sorted(
                        value.items(), key=lambda pair: str(pair[0])
                    )
                ],
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": "sequence",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "items": [_layout_tree(item, seen=seen) for item in value],
            }
        children = getattr(value, "caches", None)
        if isinstance(children, (list, tuple)):
            return {
                "kind": "cache_group",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "caches": [_layout_tree(item, seen=seen) for item in children],
            }
        state = getattr(value, "state", None)
        meta_state = getattr(value, "meta_state", None)
        return {
            "kind": "cache",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _layout_tree(state, seen=seen) if state is not None else None,
            "meta": _layout_tree(meta_state, seen=seen)
            if meta_state is not None
            else None,
        }
    finally:
        seen.discard(value_id)


def layout_fingerprint(value: Any) -> str:
    """Fingerprint cache/hidden topology, shapes, and dtypes only."""

    return _canonical_hash(_layout_tree(value))


def _cache_nodes(value: Any, *, seen: set[int] | None = None):
    """Yield cache-level objects without descending into their tensor state."""

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    if isinstance(value, Mapping):
        # Scheduler-extracted cache records retain the source class name and
        # complete state/meta tuple.  Treat each record as one cache node.
        if "class_name" in value and "state" in value:
            yield value
            return
        for child in value.values():
            yield from _cache_nodes(child, seen=seen)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _cache_nodes(child, seen=seen)
        return
    children = getattr(value, "caches", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _cache_nodes(child, seen=seen)
        return
    yield value


def _node_name(value: Any) -> str:
    if isinstance(value, Mapping):
        declared = value.get("class_name")
        if isinstance(declared, str):
            return declared.lower()
    return type(value).__name__.lower()


def _extract_prepared_surfaces(target_cache: Any) -> _PreparedSurfaces:
    """Project live GDN/PLE/QSA objects from one restored target cache.

    These are views into the target cache returned by APC, not detached copies.
    Runtime execution therefore consumes the exact objects whose layouts are
    identity-bound below.  Qwen4's PLE cache also owns GDN state and appears in
    both projections intentionally.
    """

    gdn: list[Any] = []
    ple: list[Any] = []
    qsa: list[Any] = []
    for node in _cache_nodes(target_cache):
        name = _node_name(node)
        module = type(node).__module__.lower()
        state = (
            node.get("state")
            if isinstance(node, Mapping)
            else getattr(node, "state", None)
        )
        # The production Qwen4 recurrent cache is named
        # ``Qwen4ExpStateCache``.  The original bridge fixtures used the older
        # ``Qwen4ArraysCache`` spelling, so matching only ``arrayscache`` made
        # every real Qwen4 capture fail closed even though the live GDN/PLE
        # state was present.  Bind the concrete vendored owner as well as the
        # compatibility spellings; slot count distinguishes GDN-only (2) from
        # GDN+PLE (4) layers.
        is_qwen4_state = (
            "qwen4expstatecache" in name
            or "qwen4arrays" in name
            or (
                module == "vllm_mlx.models.qwen4_exp_cache"
                and name.endswith("statecache")
            )
        )
        owned_slots = getattr(node, "cache", None)
        if not isinstance(owned_slots, (list, tuple)):
            # ArraysCache.state is ``(cache_slots, left_padding, lengths)`` in
            # current mlx-lm, while older/fake owners expose the slots
            # directly.  Project only the first component when it is the
            # nested slot list.
            owned_slots = (
                state[0]
                if isinstance(state, (list, tuple))
                and state
                and isinstance(state[0], (list, tuple))
                else state
            )
        slot_count = (
            len(owned_slots) if isinstance(owned_slots, (list, tuple)) else 0
        )
        is_qsa = "qsa" in name or any(
            hasattr(node, attribute)
            for attribute in ("index_keys", "pooled_keys", "_mtp_shared_topk")
        )
        is_ple = (
            is_qwen4_state
            and slot_count >= 4
        ) or (
            hasattr(node, "_ple_rollback")
            or (
                "arrayscache" in name
                and slot_count >= 4
            )
        )
        is_gdn = (
            "arrayscache" in name
            or "mambacache" in name
            or "gateddelta" in name
            or is_qwen4_state
            or is_ple
        ) and not is_qsa
        if is_gdn:
            gdn.append(node)
        if is_ple:
            ple.append(node)
        if is_qsa:
            qsa.append(node)
    return _PreparedSurfaces(
        tuple(gdn) or None,
        tuple(ple) or None,
        tuple(qsa) or None,
    )


def _payload_nbytes(value: Any, *, seen: set[int] | None = None) -> int:
    """Best-effort unique tensor-byte charge for bounded sidecar telemetry."""

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None and not isinstance(nbytes, bool):
        try:
            return max(0, int(nbytes))
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return sum(_payload_nbytes(child, seen=seen) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_nbytes(child, seen=seen) for child in value)
    children = getattr(value, "caches", None)
    if isinstance(children, (list, tuple)):
        return sum(_payload_nbytes(child, seen=seen) for child in children)
    state = getattr(value, "state", None)
    if state is not None:
        return _payload_nbytes(state, seen=seen)
    return 0


def _walk_type_names(value: Any, *, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    yield type(value).__name__.lower()
    if isinstance(value, Mapping):
        children = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        children = getattr(value, "caches", ())
    if isinstance(children, (list, tuple, Mapping)):
        iterable = children.values() if isinstance(children, Mapping) else children
        for child in iterable:
            yield from _walk_type_names(child, seen=seen)


def _contains_windowed_cache(value: Any) -> bool:
    return any(
        marker in name
        for name in _walk_type_names(value)
        for marker in _WINDOWED_MARKERS
    )


def _cache_offsets(value: Any, *, seen: set[int] | None = None) -> tuple[int, ...]:
    """Collect declared cache cursors without inspecting tensor values."""

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return ()
    seen.add(value_id)
    offsets: list[int] = []
    offset = getattr(value, "offset", None)
    if offset is not None and not isinstance(offset, bool):
        try:
            parsed_offset = int(offset)
        except (TypeError, ValueError):
            parsed_offset = -1
        if parsed_offset >= 0:
            offsets.append(parsed_offset)
    if isinstance(value, Mapping):
        children = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        children = getattr(value, "caches", ())
    if isinstance(children, (list, tuple)):
        for child in children:
            offsets.extend(_cache_offsets(child, seen=seen))
    return tuple(offsets)


def _cursor_matches(value: Any, expected: int) -> bool:
    offsets = _cache_offsets(value)
    return not offsets or max(offsets) == expected


def _seed_shape_is_exact(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    try:
        dimensions = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return False
    return len(dimensions) == 3 and dimensions[0] == 1 and dimensions[1] == 1


class ContinuousMTPAPCBridge:
    """Bounded, process-local MTP sidecar index keyed by exact APC boundary."""

    def __init__(
        self,
        namespace: ContinuousMTPAPCNamespace,
        *,
        max_entries: int = 100,
        max_age_seconds: float | None = None,
        min_useful_prefix_tokens: int = 64,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise ValueError("max_entries must be an integer")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if (
            isinstance(min_useful_prefix_tokens, bool)
            or not isinstance(min_useful_prefix_tokens, int)
            or min_useful_prefix_tokens < 1
        ):
            raise ValueError("min_useful_prefix_tokens must be positive")
        self.namespace = namespace
        self.max_entries = max_entries
        self.max_age_seconds = max_age_seconds
        self.min_useful_prefix_tokens = min_useful_prefix_tokens
        self._entries: OrderedDict[tuple[int, ...], ContinuousMTPAPCSidecar] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._restore_reasons = {reason.value: 0 for reason in RestoreReason}
        self._captures_attempted = 0
        self._captures_accepted = 0
        self._commits_accepted = 0
        self._commits_rejected = 0
        self._attach_failures = 0
        self._payload_nbytes = 0

    @property
    def _required_surfaces(self) -> frozenset[str]:
        if self.namespace.model_family == "qwen4_exp":
            return frozenset(("gdn", "ple", "qsa"))
        return frozenset()

    def _surfaces_complete(self, surfaces: _PreparedSurfaces) -> bool:
        return all(
            getattr(surfaces, f"{name}_state") is not None
            for name in self._required_surfaces
        )

    def _restore_result(
        self,
        eligibility: RestoreEligibility,
        *,
        state: PreparedMTPState | None = None,
        expected_identity: PreparedStateIdentity | None = None,
    ) -> ContinuousMTPAPCRestore:
        with self._lock:
            self._restore_reasons[eligibility.reason.value] += 1
        return ContinuousMTPAPCRestore(eligibility, state, expected_identity)

    def _identity(
        self,
        *,
        target_cache: Any,
        mtp_cache: Any,
        seed_hidden: Any,
        surfaces: _PreparedSurfaces,
    ) -> PreparedStateIdentity:
        return PreparedStateIdentity(
            model_id=self.namespace.model_id,
            model_revision=self.namespace.model_revision,
            speculative_config_fingerprint=fingerprint_config(
                self.namespace.speculative_config
            ),
            target_cache_layout=layout_fingerprint(target_cache),
            mtp_cache_layout=layout_fingerprint(mtp_cache),
            seed_hidden_layout=layout_fingerprint(seed_hidden),
            gdn_state_layout=(
                layout_fingerprint(surfaces.gdn_state)
                if surfaces.gdn_state is not None
                else ABSENT_STATE_LAYOUT
            ),
            ple_state_layout=(
                layout_fingerprint(surfaces.ple_state)
                if surfaces.ple_state is not None
                else ABSENT_STATE_LAYOUT
            ),
            qsa_state_layout=(
                layout_fingerprint(surfaces.qsa_state)
                if surfaces.qsa_state is not None
                else ABSENT_STATE_LAYOUT
            ),
            adapter_id=self.namespace.adapter_id,
            tokenizer_fingerprint=self.namespace.tokenizer_fingerprint,
        )

    def capture(
        self,
        tokens: Sequence[int],
        target_cache: Any,
        mtp_state: Any,
        *,
        captured_at: float | None = None,
    ) -> ContinuousMTPAPCSidecar | None:
        """Build a sidecar, returning ``None`` for unsafe runtime state."""

        with self._lock:
            self._captures_attempted += 1
        prefix = tuple(tokens)
        if not prefix or target_cache is None:
            return None
        if not isinstance(mtp_state, (list, tuple)) or len(mtp_state) != 2:
            return None
        mtp_cache, seed_hidden = mtp_state
        if mtp_cache is None or seed_hidden is None:
            return None
        if not _seed_shape_is_exact(seed_hidden):
            return None
        if _contains_windowed_cache(target_cache) or _contains_windowed_cache(
            mtp_cache
        ):
            return None
        try:
            surfaces = _extract_prepared_surfaces(target_cache)
            if not self._surfaces_complete(surfaces):
                return None
            identity = self._identity(
                target_cache=target_cache,
                mtp_cache=mtp_cache,
                seed_hidden=seed_hidden,
                surfaces=surfaces,
            )
            timestamp = time.time() if captured_at is None else float(captured_at)
            if not math.isfinite(timestamp) or timestamp < 0:
                return None
            if not _cursor_matches(target_cache, len(prefix)):
                return None
            if not _cursor_matches(mtp_cache, len(prefix) - 1):
                return None
            prepared = prepare_mtp_state(
                identity=identity,
                prefix_tokens=prefix,
                target_cache=target_cache,
                target_cache_tokens=len(prefix),
                mtp_cache=mtp_cache,
                mtp_cache_pairs=len(prefix) - 1,
                seed_hidden=seed_hidden,
                gdn_state=surfaces.gdn_state,
                ple_state=surfaces.ple_state,
                qsa_state=surfaces.qsa_state,
                captured_at=timestamp,
            )
            copied_mtp = copy.deepcopy(mtp_cache)
            copied_seed = copy.deepcopy(seed_hidden)
            sidecar = ContinuousMTPAPCSidecar(
                metadata=prepared.metadata,
                mtp_cache=copied_mtp,
                seed_hidden=copied_seed,
                payload_nbytes=_payload_nbytes((copied_mtp, copied_seed)),
            )
            with self._lock:
                self._captures_accepted += 1
            return sidecar
        except Exception:  # noqa: BLE001 - optional sidecar capture fails closed
            return None

    def commit(
        self,
        tokens: Sequence[int],
        sidecar: ContinuousMTPAPCSidecar | None,
    ) -> bool:
        """Commit only after the owning target APC entry was stored."""

        if sidecar is None:
            with self._lock:
                self._commits_rejected += 1
            return False
        key = tuple(tokens)
        metadata = sidecar.metadata
        if (
            len(key) != metadata.covered_tokens
            or fingerprint_tokens(key) != metadata.boundary_fingerprint
        ):
            with self._lock:
                self._commits_rejected += 1
            return False
        with self._lock:
            replaced = self._entries.get(key)
            if replaced is not None:
                self._payload_nbytes -= replaced.payload_nbytes
            self._entries[key] = sidecar
            self._entries.move_to_end(key)
            self._payload_nbytes += sidecar.payload_nbytes
            self._commits_accepted += 1
            while len(self._entries) > self.max_entries:
                _evicted_key, evicted = self._entries.popitem(last=False)
                self._payload_nbytes -= evicted.payload_nbytes
        return True

    def restore(
        self,
        request_tokens: Sequence[int],
        *,
        target_cache: Any,
        cached_tokens: int,
        now: float | None = None,
    ) -> ContinuousMTPAPCRestore:
        """Compose an exact target APC hit with its matching sidecar."""

        request = tuple(request_tokens)
        if (
            target_cache is None
            or isinstance(cached_tokens, bool)
            or not isinstance(cached_tokens, int)
            or cached_tokens < 1
            or cached_tokens >= len(request)
        ):
            return self._restore_result(
                RestoreEligibility(False, RestoreReason.BOUNDARY_MISMATCH)
            )
        key = request[:cached_tokens]
        with self._lock:
            sidecar = self._entries.get(key)
            if sidecar is not None:
                self._entries.move_to_end(key)
        if sidecar is None:
            return self._restore_result(
                RestoreEligibility(
                    False,
                    RestoreReason.BOUNDARY_MISMATCH,
                    covered_tokens=cached_tokens,
                )
            )
        metadata = sidecar.metadata
        if not _cursor_matches(target_cache, cached_tokens):
            return self._restore_result(
                RestoreEligibility(
                    False,
                    RestoreReason.BOUNDARY_MISMATCH,
                    covered_tokens=cached_tokens,
                )
            )
        try:
            mtp_cache = copy.deepcopy(sidecar.mtp_cache)
            seed_hidden = copy.deepcopy(sidecar.seed_hidden)
            surfaces = _extract_prepared_surfaces(target_cache)
            if not self._surfaces_complete(surfaces):
                return self._restore_result(
                    RestoreEligibility(
                        False,
                        RestoreReason.BOUNDARY_MISMATCH,
                        covered_tokens=cached_tokens,
                    )
                )
            if not _seed_shape_is_exact(seed_hidden):
                return self._restore_result(
                    RestoreEligibility(
                        False,
                        RestoreReason.BOUNDARY_MISMATCH,
                        covered_tokens=cached_tokens,
                    )
                )
            if not _cursor_matches(mtp_cache, cached_tokens - 1):
                return self._restore_result(
                    RestoreEligibility(
                        False,
                        RestoreReason.BOUNDARY_MISMATCH,
                        covered_tokens=cached_tokens,
                    )
                )
            expected = self._identity(
                target_cache=target_cache,
                mtp_cache=mtp_cache,
                seed_hidden=seed_hidden,
                surfaces=surfaces,
            )
        except Exception:  # noqa: BLE001 - optional cache lookup must fail closed
            return self._restore_result(
                RestoreEligibility(
                    False, RestoreReason.MALFORMED, covered_tokens=cached_tokens
                )
            )
        state = PreparedMTPState(
            metadata=metadata,
            target_cache=target_cache,
            mtp_cache=mtp_cache,
            seed_hidden=seed_hidden,
            gdn_state=surfaces.gdn_state,
            ple_state=surfaces.ple_state,
            qsa_state=surfaces.qsa_state,
        )
        eligibility = evaluate_restore(
            state,
            expected_identity=expected,
            request_tokens=request,
            target_cache_tokens=cached_tokens,
            mtp_cache_pairs=cached_tokens - 1,
            now=now,
            max_age_seconds=self.max_age_seconds,
            min_useful_prefix_tokens=self.min_useful_prefix_tokens,
        )
        return self._restore_result(
            eligibility,
            state=state if eligibility.eligible else None,
            expected_identity=expected if eligibility.eligible else None,
        )

    def stats_snapshot(self) -> dict[str, Any]:
        """Return bounded observability without request- or token-derived labels."""

        with self._lock:
            return {
                "entries": len(self._entries),
                "payload_nbytes": self._payload_nbytes,
                "captures_attempted": self._captures_attempted,
                "captures_accepted": self._captures_accepted,
                "commits_accepted": self._commits_accepted,
                "commits_rejected": self._commits_rejected,
                "attach_failures": self._attach_failures,
                "restore_reasons": dict(self._restore_reasons),
            }

    def record_attach_failure(self) -> None:
        """Record a post-validation request-attachment rollback."""

        with self._lock:
            self._attach_failures += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._payload_nbytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = [
    "ContinuousMTPAPCBridge",
    "ContinuousMTPAPCNamespace",
    "ContinuousMTPAPCRestore",
    "ContinuousMTPAPCSidecar",
    "layout_fingerprint",
]
