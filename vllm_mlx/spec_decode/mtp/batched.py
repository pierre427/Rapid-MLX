# SPDX-License-Identifier: Apache-2.0
"""Pure-Python contracts for continuous batched self-MTP.

This module deliberately does not import MLX or the model-specific MTP
implementations.  It owns the control-plane invariants that a later MLX
coordinator must satisfy before it may run a batched propose/verify pass:

* the feature is explicit-opt-in and every required capability is fail-closed;
* lane admission respects a hard memory reserve and degrades draft depth;
* a proposal is bound to an ordered membership epoch; and
* membership cannot change until that proposal is committed or aborted.

It does not execute model work, mutate caches, or sample tokens.  Those remain
the responsibility of the future coordinator, which should use
:class:`BatchedMTPBookkeeper` as its transaction ledger.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


class BatchedMTPRoute(str, enum.Enum):
    """Control-plane route chosen for a lane set."""

    BATCHED_MTP = "batched_mtp"
    PLAIN_DECODE = "plain_decode"
    QUEUE = "queue"


@dataclass(frozen=True)
class BatchedMTPCapabilities:
    """Observed runtime capabilities; every field defaults to refusal.

    These are runtime facts, not architecture-name promises.  An injector or
    coordinator must explicitly attest each capability after installing the
    corresponding model/cache surface.
    """

    target_batch_forward: bool = False
    mtp_batch_forward: bool = False
    ragged_target_rollback: bool = False
    ragged_mtp_rollback: bool = False
    atomic_cache_commit: bool = False
    per_lane_rng: bool = False
    transformed_distribution_verify: bool = False
    dynamic_membership: bool = False
    xtc_exact_verify: bool = False

    def missing_core(self) -> tuple[str, ...]:
        required = (
            "target_batch_forward",
            "mtp_batch_forward",
            "ragged_target_rollback",
            "ragged_mtp_rollback",
            "atomic_cache_commit",
        )
        return tuple(name for name in required if getattr(self, name) is not True)


@dataclass(frozen=True)
class BatchedMTPConfig:
    """Default-off policy for the future continuous-batching coordinator."""

    enabled: bool = False
    max_lanes: int = 4
    max_draft_tokens: int = 2
    min_batch_lanes: int = 2
    hard_reserve_bytes: int = 8 * 1024**3
    allow_dynamic_membership: bool = False
    queue_on_memory_pressure: bool = True
    multi_lane_max_prompt_tokens: int = 0
    single_lane_max_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        boolean_fields = (
            "enabled",
            "allow_dynamic_membership",
            "queue_on_memory_pressure",
        )
        if any(not isinstance(getattr(self, name), bool) for name in boolean_fields):
            raise ValueError("batched MTP policy flags must be booleans")
        if isinstance(self.max_lanes, bool) or not isinstance(self.max_lanes, int):
            raise ValueError("max_lanes must be a positive integer")
        if self.max_lanes < 1:
            raise ValueError("max_lanes must be positive")
        if isinstance(self.max_draft_tokens, bool) or not isinstance(
            self.max_draft_tokens, int
        ):
            raise ValueError("max_draft_tokens must be a positive integer")
        if self.max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be positive")
        if isinstance(self.min_batch_lanes, bool) or not isinstance(
            self.min_batch_lanes, int
        ):
            raise ValueError("min_batch_lanes must be an integer")
        if self.min_batch_lanes < 1:
            raise ValueError("min_batch_lanes must be positive")
        if self.min_batch_lanes > self.max_lanes:
            raise ValueError("min_batch_lanes cannot exceed max_lanes")
        if self.hard_reserve_bytes < 0:
            raise ValueError("hard_reserve_bytes cannot be negative")
        for name in (
            "multi_lane_max_prompt_tokens",
            "single_lane_max_prompt_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class SamplingContract:
    """Sampling features that must remain exact through verification."""

    greedy: bool = True
    has_logits_processors: bool = False
    uses_xtc: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (self.greedy, self.has_logits_processors, self.uses_xtc)
        ):
            raise ValueError("sampling contract flags must be booleans")


@dataclass(frozen=True)
class LaneAdmission:
    """Scheduler-neutral description of one candidate request lane."""

    lane_id: str
    base_bytes: int = 0
    bytes_per_draft_token: int = 0
    sampling: SamplingContract = SamplingContract()
    cache_ready: bool = True
    terminal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.lane_id, str) or not self.lane_id:
            raise ValueError("lane_id must be a non-empty string")
        estimates = (self.base_bytes, self.bytes_per_draft_token)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in estimates
        ):
            raise ValueError("lane memory estimates must be integers")
        if self.base_bytes < 0 or self.bytes_per_draft_token < 0:
            raise ValueError("lane memory estimates cannot be negative")
        if not isinstance(self.cache_ready, bool) or not isinstance(
            self.terminal, bool
        ):
            raise ValueError("lane lifecycle flags must be booleans")

    def estimated_bytes(self, draft_tokens: int) -> int:
        return self.base_bytes + self.bytes_per_draft_token * draft_tokens


@dataclass(frozen=True)
class LaneGate:
    """Fail-closed eligibility verdict for one lane."""

    eligible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionDecision:
    """A deterministic routing plan; it performs no scheduler mutation."""

    route: BatchedMTPRoute
    batched_lane_ids: tuple[str, ...] = ()
    plain_lane_ids: tuple[str, ...] = ()
    queued_lane_ids: tuple[str, ...] = ()
    draft_tokens: int = 0
    estimated_bytes: int = 0
    reasons: tuple[str, ...] = ()


def assess_lane(
    lane: LaneAdmission,
    *,
    config: BatchedMTPConfig,
    capabilities: BatchedMTPCapabilities,
) -> LaneGate:
    """Return an explicit, fail-closed eligibility verdict for ``lane``."""

    reasons: list[str] = []
    if config.enabled is not True:
        reasons.append("batched self-MTP is disabled")
    reasons.extend(
        f"missing capability: {name}" for name in capabilities.missing_core()
    )
    if (
        config.allow_dynamic_membership is True
        and capabilities.dynamic_membership is not True
    ):
        reasons.append("missing capability: dynamic_membership")
    if lane.cache_ready is not True:
        reasons.append("lane cache is not ready")
    if lane.terminal is True:
        reasons.append("lane is terminal")
    if lane.sampling.has_logits_processors is True:
        reasons.append("logits processors are not supported")
    if lane.sampling.greedy is not True:
        if capabilities.per_lane_rng is not True:
            reasons.append("missing capability: per_lane_rng")
        if capabilities.transformed_distribution_verify is not True:
            reasons.append("missing capability: transformed_distribution_verify")
    if lane.sampling.uses_xtc is True and capabilities.xtc_exact_verify is not True:
        reasons.append("XTC verification is not exact")
    return LaneGate(not reasons, tuple(reasons))


def plan_admission(
    lanes: Iterable[LaneAdmission],
    *,
    config: BatchedMTPConfig,
    capabilities: BatchedMTPCapabilities,
    free_bytes: int,
) -> AdmissionDecision:
    """Choose a lane set and draft depth without allocating memory.

    The input order is scheduler priority.  The planner maximizes admitted
    lane count first, then draft depth.  Consequently it lowers K when doing
    so admits more concurrent requests.  Ineligible lanes always use plain
    decode; eligible lanes that cannot fit the hard reserve are queued when
    ``queue_on_memory_pressure`` is true.
    """

    if isinstance(free_bytes, bool) or not isinstance(free_bytes, int):
        raise ValueError("free_bytes must be an integer")
    if free_bytes < 0:
        raise ValueError("free_bytes cannot be negative")
    lane_list = tuple(lanes)
    ids = [lane.lane_id for lane in lane_list]
    if len(ids) != len(set(ids)):
        raise ValueError("lane_id values must be unique")
    if not lane_list:
        return AdmissionDecision(
            BatchedMTPRoute.PLAIN_DECODE, reasons=("no candidate lanes",)
        )

    eligible: list[LaneAdmission] = []
    plain: list[str] = []
    refusal_reasons: list[str] = []
    for lane in lane_list:
        gate = assess_lane(lane, config=config, capabilities=capabilities)
        if gate.eligible:
            eligible.append(lane)
        else:
            plain.append(lane.lane_id)
            refusal_reasons.extend(
                f"{lane.lane_id}: {reason}" for reason in gate.reasons
            )

    candidates = eligible[: config.max_lanes]
    over_limit = tuple(lane.lane_id for lane in eligible[config.max_lanes :])
    if len(candidates) < config.min_batch_lanes:
        plain.extend(lane.lane_id for lane in candidates)
        plain.extend(over_limit)
        return AdmissionDecision(
            BatchedMTPRoute.PLAIN_DECODE,
            plain_lane_ids=tuple(plain),
            reasons=tuple(refusal_reasons or ("fewer than minimum eligible lanes",)),
        )

    usable = max(0, free_bytes - config.hard_reserve_bytes)
    best: tuple[int, int, int] | None = None
    # Score by lane count first, then K.  Each depth considers the longest
    # scheduler-priority prefix that fits the reserve.
    for depth in range(config.max_draft_tokens, 0, -1):
        used = 0
        count = 0
        for lane in candidates:
            cost = lane.estimated_bytes(depth)
            if used + cost > usable:
                break
            used += cost
            count += 1
        candidate = (count, depth, used)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    count, depth, used = best
    if count >= config.min_batch_lanes:
        admitted = tuple(lane.lane_id for lane in candidates[:count])
        overflow = tuple(lane.lane_id for lane in candidates[count:]) + over_limit
        if config.queue_on_memory_pressure:
            queued = overflow
        else:
            plain.extend(overflow)
            queued = ()
        return AdmissionDecision(
            BatchedMTPRoute.BATCHED_MTP,
            batched_lane_ids=admitted,
            plain_lane_ids=tuple(plain),
            queued_lane_ids=queued,
            draft_tokens=depth,
            estimated_bytes=used,
            reasons=tuple(refusal_reasons),
        )

    pressured = tuple(lane.lane_id for lane in candidates) + over_limit
    if config.queue_on_memory_pressure:
        return AdmissionDecision(
            BatchedMTPRoute.QUEUE,
            plain_lane_ids=tuple(plain),
            queued_lane_ids=pressured,
            reasons=tuple(refusal_reasons + ["hard memory reserve prevents a batch"]),
        )
    plain.extend(pressured)
    return AdmissionDecision(
        BatchedMTPRoute.PLAIN_DECODE,
        plain_lane_ids=tuple(plain),
        reasons=tuple(refusal_reasons + ["hard memory reserve prevents a batch"]),
    )


class BatchedMTPTransactionError(RuntimeError):
    """A propose/commit or membership-epoch invariant was violated."""


@dataclass(frozen=True)
class ProposalTicket:
    """Opaque proof that a proposal belongs to one ordered membership epoch."""

    transaction_id: int
    membership_epoch: int
    lane_ids: tuple[str, ...]
    draft_tokens: int
    verify_rows: int


@dataclass(frozen=True)
class CommitReceipt:
    transaction_id: int
    membership_epoch: int
    emitted_counts: tuple[tuple[str, int], ...]
    terminal_lane_ids: tuple[str, ...]
    total_emitted: int


class BatchedMTPBookkeeper:
    """Pure-Python membership and propose-to-commit transaction ledger."""

    def __init__(self, lane_ids: Iterable[str] = ()) -> None:
        self._lane_ids: list[str] = []
        self._membership_epoch = 0
        self._next_transaction_id = 1
        self._outstanding: ProposalTicket | None = None
        self._committed_tokens: dict[str, int] = {}
        initial = tuple(lane_ids)
        if initial:
            self.attach(initial)

    @property
    def lane_ids(self) -> tuple[str, ...]:
        return tuple(self._lane_ids)

    @property
    def membership_epoch(self) -> int:
        return self._membership_epoch

    @property
    def outstanding(self) -> ProposalTicket | None:
        return self._outstanding

    def committed_tokens(self, lane_id: str) -> int:
        try:
            return self._committed_tokens[lane_id]
        except KeyError as exc:
            raise KeyError(f"unknown lane_id: {lane_id}") from exc

    def _require_membership_boundary(self) -> None:
        if self._outstanding is not None:
            raise BatchedMTPTransactionError(
                "membership cannot change while a proposal is outstanding"
            )

    @staticmethod
    def _validate_ids(lane_ids: Iterable[str]) -> tuple[str, ...]:
        ids = tuple(lane_ids)
        if any(not isinstance(lane_id, str) or not lane_id for lane_id in ids):
            raise ValueError("lane_id values must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError("lane_id values must be unique")
        return ids

    def attach(self, lane_ids: Iterable[str]) -> int:
        """Attach lanes at a transaction boundary and return the new epoch."""

        self._require_membership_boundary()
        ids = self._validate_ids(lane_ids)
        duplicates = set(ids).intersection(self._lane_ids)
        if duplicates:
            raise BatchedMTPTransactionError(
                f"lane already attached: {', '.join(sorted(duplicates))}"
            )
        if ids:
            self._lane_ids.extend(ids)
            self._committed_tokens.update((lane_id, 0) for lane_id in ids)
            self._membership_epoch += 1
        return self._membership_epoch

    def detach(self, lane_ids: Iterable[str]) -> int:
        """Detach lanes at a transaction boundary and return the new epoch."""

        self._require_membership_boundary()
        ids = self._validate_ids(lane_ids)
        unknown = set(ids).difference(self._lane_ids)
        if unknown:
            raise BatchedMTPTransactionError(
                f"lane not attached: {', '.join(sorted(unknown))}"
            )
        if ids:
            removed = set(ids)
            self._lane_ids = [
                lane_id for lane_id in self._lane_ids if lane_id not in removed
            ]
            for lane_id in ids:
                del self._committed_tokens[lane_id]
            self._membership_epoch += 1
        return self._membership_epoch

    def begin_proposal(self, *, draft_tokens: int) -> ProposalTicket:
        """Open one proposal for the current ordered membership."""

        if self._outstanding is not None:
            raise BatchedMTPTransactionError("a proposal is already outstanding")
        if not self._lane_ids:
            raise BatchedMTPTransactionError("cannot propose with no attached lanes")
        if (
            isinstance(draft_tokens, bool)
            or not isinstance(draft_tokens, int)
            or draft_tokens < 1
        ):
            raise ValueError("draft_tokens must be a positive integer")
        ticket = ProposalTicket(
            transaction_id=self._next_transaction_id,
            membership_epoch=self._membership_epoch,
            lane_ids=tuple(self._lane_ids),
            draft_tokens=draft_tokens,
            verify_rows=(draft_tokens + 1) * len(self._lane_ids),
        )
        self._next_transaction_id += 1
        self._outstanding = ticket
        return ticket

    def _require_ticket(self, ticket: ProposalTicket) -> None:
        if self._outstanding is None:
            raise BatchedMTPTransactionError("no proposal is outstanding")
        if ticket != self._outstanding:
            raise BatchedMTPTransactionError("stale or foreign proposal ticket")
        if ticket.membership_epoch != self._membership_epoch:
            raise BatchedMTPTransactionError("proposal membership epoch is stale")
        if ticket.lane_ids != tuple(self._lane_ids):
            raise BatchedMTPTransactionError("proposal lane order is stale")

    def commit(
        self,
        ticket: ProposalTicket,
        *,
        emitted_counts: Mapping[str, int],
        terminal_lane_ids: Iterable[str] = (),
    ) -> CommitReceipt:
        """Commit per-lane accepted counts, without changing membership."""

        self._require_ticket(ticket)
        if set(emitted_counts) != set(ticket.lane_ids):
            raise BatchedMTPTransactionError(
                "emitted_counts must contain every proposal lane exactly once"
            )
        ordered: list[tuple[str, int]] = []
        maximum = ticket.draft_tokens + 1
        for lane_id in ticket.lane_ids:
            count = emitted_counts[lane_id]
            if isinstance(count, bool) or not isinstance(count, int):
                raise BatchedMTPTransactionError("emitted counts must be integers")
            if count < 1 or count > maximum:
                raise BatchedMTPTransactionError(
                    f"emitted count for {lane_id} must be between 1 and {maximum}"
                )
            ordered.append((lane_id, count))

        terminal = self._validate_ids(terminal_lane_ids)
        unknown_terminal = set(terminal).difference(ticket.lane_ids)
        if unknown_terminal:
            raise BatchedMTPTransactionError(
                "terminal lanes must belong to the committed proposal"
            )

        for lane_id, count in ordered:
            self._committed_tokens[lane_id] += count
        self._outstanding = None
        return CommitReceipt(
            transaction_id=ticket.transaction_id,
            membership_epoch=ticket.membership_epoch,
            emitted_counts=tuple(ordered),
            terminal_lane_ids=terminal,
            total_emitted=sum(count for _, count in ordered),
        )

    def abort(self, ticket: ProposalTicket) -> None:
        """Discard bookkeeping for an uncommitted proposal."""

        self._require_ticket(ticket)
        self._outstanding = None


__all__ = [
    "AdmissionDecision",
    "BatchedMTPBookkeeper",
    "BatchedMTPCapabilities",
    "BatchedMTPConfig",
    "BatchedMTPRoute",
    "BatchedMTPTransactionError",
    "CommitReceipt",
    "LaneAdmission",
    "LaneGate",
    "ProposalTicket",
    "SamplingContract",
    "assess_lane",
    "plan_admission",
]
