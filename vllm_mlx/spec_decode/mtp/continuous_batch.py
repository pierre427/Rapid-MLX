# SPDX-License-Identifier: Apache-2.0
"""Scheduler-neutral generation wrapper for continuous self-MTP.

The transaction engine in :mod:`continuous_engine` deliberately knows
nothing about request delivery.  This module adds that missing, pure-Python
boundary: it delivers the prepared first token, limits one proposal to each
lane's remaining token budget and stop-token set, and commits exactly the
prefix that the caller receives.

This milestone is intentionally fixed-cohort.  A lane reaching a terminal
condition tears down the *whole* cohort after the open proposal is committed.
Terminal lanes are marked for finalization; companion lanes are returned as
resumable detach packages.  A later scheduler integration may form a new
cohort from those companions, but it must not turn that turnover into an
incremental join.  In particular, the wrapper never enables Flash dynamic
membership, even if a lower-level runtime is accidentally over-attested.

No MLX type is imported here.  Model execution, cache merge/rollback/extract,
and the target/MTP forward calls remain injected through ``continuous_engine``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .continuous_engine import (
    BatchedSelfMTPState,
    ContinuousSelfMTPError,
    ContinuousSelfMTPRuntime,
    ContinuousSelfMTPUnsupportedError,
    DetachedSelfMTPLane,
    MTPToken,
    SelfMTPLaneSpec,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)


class ContinuousMTPGenerationBatchError(ContinuousSelfMTPError):
    """A delivery or lifecycle invariant failed in the generation wrapper."""


@dataclass(frozen=True)
class ContinuousMTPLaneState:
    """Immutable scheduler-facing snapshot of one lane's delivery state."""

    uid: int
    emitted_tokens: int
    max_tokens: int
    remaining_tokens: int
    stop_tokens: frozenset[int]
    terminal: bool
    finish_reason: str | None


@dataclass(frozen=True)
class ContinuousMTPLaneEmission:
    """The exact token prefix delivered for one lane in one burst."""

    uid: int
    tokens: tuple[MTPToken, ...]
    terminal: bool = False
    finish_reason: str | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token for token in self.tokens)

    @property
    def logprobs(self) -> tuple[Any, ...]:
        return tuple(token.logprobs for token in self.tokens)


@dataclass(frozen=True)
class ContinuousMTPDetachPackage:
    """A detached lane plus its exact delivered-token ledger.

    ``terminal`` distinguishes a request the scheduler should finalize from a
    companion that was detached only because fixed-cohort turnover was
    required.  Companion packages retain their canonical lane and cache pair
    and are therefore resumable by a later integration.
    """

    detached: DetachedSelfMTPLane
    tokens: tuple[MTPToken, ...]
    terminal: bool
    finish_reason: str | None

    @property
    def uid(self) -> int:
        return self.detached.lane.uid

    @property
    def lane(self):
        return self.detached.lane

    @property
    def caches(self):
        return self.detached.caches

    @property
    def target_cache(self):
        return self.detached.caches.target

    @property
    def draft_cache(self):
        return self.detached.caches.draft

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token for token in self.tokens)

    @property
    def logprobs(self) -> tuple[Any, ...]:
        return tuple(token.logprobs for token in self.tokens)


@dataclass(frozen=True)
class ContinuousMTPGenerationBurst:
    """One delivery event: initial tokens or one proposal transaction."""

    emissions: tuple[ContinuousMTPLaneEmission, ...]
    emitted_counts: tuple[int, ...]
    initial: bool
    detached: tuple[ContinuousMTPDetachPackage, ...] = ()

    @property
    def cohort_detached(self) -> bool:
        return bool(self.detached)

    @property
    def terminal_detaches(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        return tuple(package for package in self.detached if package.terminal)

    @property
    def resumable_detaches(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        return tuple(package for package in self.detached if not package.terminal)


@dataclass
class _LaneDeliveryState:
    uid: int
    max_tokens: int
    stop_tokens: frozenset[int]
    tokens: list[MTPToken] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.finish_reason is not None


class ContinuousMTPGenerationBatch:
    """Own one fixed cohort from preparation through full-cohort extraction."""

    def __init__(
        self,
        *,
        batch: BatchedSelfMTPState,
        first_tokens: Sequence[MTPToken],
        stop_tokens: Mapping[int, frozenset[int]],
    ) -> None:
        if len(first_tokens) != len(batch.lanes):
            raise ValueError("first_tokens must have one entry per lane")
        self._batch = batch
        self._first_tokens = tuple(first_tokens)
        self._initial_pending = True
        self._closed = False
        self._detached: tuple[ContinuousMTPDetachPackage, ...] = ()
        self._states = {
            lane.uid: _LaneDeliveryState(
                uid=lane.uid,
                max_tokens=lane.max_tokens,
                stop_tokens=stop_tokens[lane.uid],
            )
            for lane in batch.lanes
        }

    @classmethod
    def create(
        cls,
        specs: Sequence[SelfMTPLaneSpec],
        runtime: ContinuousSelfMTPRuntime,
        *,
        stop_tokens: Mapping[int, Iterable[int]] | None = None,
    ) -> ContinuousMTPGenerationBatch:
        """Prepare and attach one initial cohort.

        ``stop_tokens`` is keyed by lane uid.  Unknown keys are rejected so a
        scheduler typo cannot silently disable a request's stop condition.
        """
        specs = tuple(specs)
        if not specs:
            raise ValueError("cannot create an empty continuous MTP cohort")
        uids = tuple(spec.uid for spec in specs)
        if len(uids) != len(set(uids)):
            raise ValueError("continuous MTP lane uid values must be unique")
        normalized_stops = _normalize_stop_tokens(uids, stop_tokens)

        prepared: list[DetachedSelfMTPLane] = []
        first_tokens: list[MTPToken] = []
        for spec in specs:
            detached, first = prepare_self_mtp_lane(spec, runtime)
            prepared.append(detached)
            first_tokens.append(first)
        batch = attach_self_mtp_lanes(None, prepared, runtime=runtime)
        return cls(
            batch=batch,
            first_tokens=first_tokens,
            stop_tokens=normalized_stops,
        )

    @property
    def lane_uids(self) -> tuple[int, ...]:
        return tuple(self._states)

    @property
    def initial_pending(self) -> bool:
        return self._initial_pending

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def lane_states(self) -> tuple[ContinuousMTPLaneState, ...]:
        snapshots = []
        for state in self._states.values():
            emitted = len(state.tokens)
            snapshots.append(
                ContinuousMTPLaneState(
                    uid=state.uid,
                    emitted_tokens=emitted,
                    max_tokens=state.max_tokens,
                    remaining_tokens=max(state.max_tokens - emitted, 0),
                    stop_tokens=state.stop_tokens,
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
        return tuple(snapshots)

    def attach_lanes(self, joining: Sequence[DetachedSelfMTPLane]) -> None:
        """Refuse incremental joins at the wrapper's fixed-cohort boundary.

        This is unconditional, including for a Flash runtime whose lower-level
        dynamic flags were set.  ``joining`` is accepted only to make the
        scheduler integration seam explicit; its contents are never touched.
        """
        del joining
        raise ContinuousSelfMTPUnsupportedError(
            "ContinuousMTPGenerationBatch is fixed-cohort; incremental join "
            "is unsupported (including Flash)"
        )

    def next_burst(self) -> ContinuousMTPGenerationBurst:
        """Deliver initial tokens or run, bound, and commit one proposal."""
        self._require_open()
        if self._initial_pending:
            return self._deliver_initial()

        proposal = propose_batched_self_mtp(self._batch)
        planned: list[tuple[int, tuple[MTPToken, ...], str | None]] = []
        emitted_counts: list[int] = []
        terminal: list[bool] = []
        for uid, outputs in zip(proposal.lane_uids, proposal.outputs):
            state = self._states[uid]
            delivered, finish_reason = _bounded_prefix(state, outputs)
            planned.append((uid, delivered, finish_reason))
            emitted_counts.append(len(delivered))
            terminal.append(finish_reason is not None)

        commit_batched_self_mtp(
            self._batch,
            proposal,
            emitted_counts=emitted_counts,
            terminal=terminal,
        )
        emissions: list[ContinuousMTPLaneEmission] = []
        for uid, delivered, finish_reason in planned:
            state = self._states[uid]
            state.tokens.extend(delivered)
            state.finish_reason = finish_reason
            emissions.append(
                ContinuousMTPLaneEmission(
                    uid=uid,
                    tokens=delivered,
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
        detached = self._detach_cohort() if any(terminal) else ()
        return ContinuousMTPGenerationBurst(
            emissions=tuple(emissions),
            emitted_counts=tuple(emitted_counts),
            initial=False,
            detached=detached,
        )

    def detach_all(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Extract every lane/cache pair without inventing finish reasons.

        This is idempotent after a successful teardown.  It is suitable for
        cancellation, shutdown, or scheduler turnover.  If the prepared first
        tokens have not yet been delivered, their token ledger is intentionally
        empty even though the detached canonical lane retains its ``cur``.
        """
        if self._closed:
            return self._detached
        return self._detach_cohort()

    def _deliver_initial(self) -> ContinuousMTPGenerationBurst:
        emissions: list[ContinuousMTPLaneEmission] = []
        terminal: list[bool] = []
        for uid, token in zip(self.lane_uids, self._first_tokens):
            state = self._states[uid]
            state.tokens.append(token)
            if token.token in state.stop_tokens:
                state.finish_reason = "stop"
            elif len(state.tokens) >= state.max_tokens:
                state.finish_reason = "length"
            emissions.append(
                ContinuousMTPLaneEmission(
                    uid=uid,
                    tokens=(token,),
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
            terminal.append(state.terminal)
        self._initial_pending = False
        detached = self._detach_cohort() if any(terminal) else ()
        return ContinuousMTPGenerationBurst(
            emissions=tuple(emissions),
            emitted_counts=tuple(1 for _ in emissions),
            initial=True,
            detached=detached,
        )

    def _detach_cohort(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        if self._closed:
            return self._detached
        indices = tuple(range(len(self._batch.lanes)))
        self._batch, detached = detach_self_mtp_lanes(self._batch, indices)
        packages = []
        for item in detached:
            state = self._states[item.lane.uid]
            packages.append(
                ContinuousMTPDetachPackage(
                    detached=item,
                    tokens=tuple(state.tokens),
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
        self._detached = tuple(packages)
        self._closed = True
        return self._detached

    def _require_open(self) -> None:
        if self._closed:
            raise ContinuousMTPGenerationBatchError(
                "continuous MTP generation cohort is already detached"
            )


def _normalize_stop_tokens(
    uids: Sequence[int],
    stop_tokens: Mapping[int, Iterable[int]] | None,
) -> dict[int, frozenset[int]]:
    raw = {} if stop_tokens is None else dict(stop_tokens)
    unknown = set(raw).difference(uids)
    if unknown:
        raise ValueError(
            f"stop_tokens contains unknown lane uid values: {sorted(unknown)}"
        )
    normalized: dict[int, frozenset[int]] = {}
    for uid in uids:
        values = frozenset(raw.get(uid, ()))
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in values
        ):
            raise ValueError("stop token ids must be integers")
        normalized[uid] = values
    return normalized


def _bounded_prefix(
    state: _LaneDeliveryState,
    outputs: Sequence[MTPToken],
) -> tuple[tuple[MTPToken, ...], str | None]:
    remaining = state.max_tokens - len(state.tokens)
    if remaining <= 0:
        raise ContinuousMTPGenerationBatchError(
            f"lane {state.uid} was proposed after exhausting max_tokens"
        )
    bounded = tuple(outputs[:remaining])
    for index, token in enumerate(bounded):
        if token.token in state.stop_tokens:
            return bounded[: index + 1], "stop"
    if len(bounded) < len(outputs) or len(bounded) == remaining:
        return bounded, "length"
    return bounded, None


__all__ = [
    "ContinuousMTPDetachPackage",
    "ContinuousMTPGenerationBatch",
    "ContinuousMTPGenerationBatchError",
    "ContinuousMTPGenerationBurst",
    "ContinuousMTPLaneEmission",
    "ContinuousMTPLaneState",
]
