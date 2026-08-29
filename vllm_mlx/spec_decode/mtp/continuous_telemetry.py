# SPDX-License-Identifier: Apache-2.0
"""Bounded telemetry and offline qualification for continuous self-MTP.

This module is deliberately pure Python.  The production continuous engine
writes its process-global, fixed-cardinality counters here, while the offline
qualification helpers use independently constructed counter instances.  Metric
dimensions accept enums only: request IDs, model IDs, arbitrary exception text,
and other unbounded labels cannot enter a snapshot.

Synthetic/model-free evidence can prove schema and reconciliation behavior but
can never set ``performance_qualified``.  That property additionally requires a
hardware evidence declaration and a raw-artifact digest bound to an exact
candidate/model identity.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class AdmissionOutcome(str, Enum):
    BATCHED = "batched"
    PLAIN = "plain"
    QUEUED = "queued"
    REFUSED = "refused"


class AdmissionReason(str, Enum):
    ELIGIBLE = "eligible"
    DEPTH_REDUCED = "depth_reduced"
    FEATURE_DISABLED = "feature_disabled"
    CAPABILITY = "capability"
    SAMPLING = "sampling"
    CACHE_TOPOLOGY = "cache_topology"
    MEMORY = "memory"
    LANE_LIMIT = "lane_limit"
    MEMBERSHIP = "membership"
    APC_STATE = "apc_state"
    INTERNAL = "internal"


class TransactionOutcome(str, Enum):
    PROPOSED = "proposed"
    COMMITTED = "committed"
    ABORTED = "aborted"
    FAILED = "failed"


class CommitKind(str, Enum):
    FULL = "full"
    TERMINAL_PARTIAL = "terminal_partial"


class AbortReason(str, Enum):
    CALLER = "caller"
    CANCELLED = "cancelled"
    DISCONNECTED = "disconnected"
    COMPUTE_ERROR = "compute_error"
    INVARIANT = "invariant"


class RollbackPhase(str, Enum):
    VERIFY = "verify"
    DELIVERY = "delivery"


class FailurePhase(str, Enum):
    PROPOSAL = "proposal"
    COMMIT = "commit"


class EvidenceKind(str, Enum):
    SYNTHETIC = "synthetic"
    APPLE_SILICON_HARDWARE = "apple_silicon_hardware"


class DigestClassification(str, Enum):
    EXACT = "exact"
    DISTRIBUTIONAL = "distributional"
    DIVERGENT = "divergent"
    NOT_RUN = "not_run"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True)
class TransactionTicket:
    """One non-exported proposal identity; never used as a metric label."""

    sequence: int
    lanes: int
    proposed_draft_tokens: int


@dataclass(frozen=True)
class ContinuousMTPSnapshot:
    """Immutable fixed-cardinality telemetry snapshot."""

    admissions: tuple[tuple[str, int], ...]
    admission_reasons: tuple[tuple[str, int], ...]
    transactions: tuple[tuple[str, int], ...]
    commits: tuple[tuple[str, int], ...]
    aborts: tuple[tuple[str, int], ...]
    rollbacks: tuple[tuple[str, int], ...]
    failures: tuple[tuple[str, int], ...]
    admitted_lanes: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    committed_tokens: int
    terminal_lanes: int
    cleaned_lanes: int
    draft_seconds: float
    target_verify_seconds: float
    open_transaction: bool

    @property
    def accept_ratio(self) -> float:
        if self.proposed_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready data with a statically bounded key space."""

        return {
            "admissions": dict(self.admissions),
            "admission_reasons": dict(self.admission_reasons),
            "transactions": dict(self.transactions),
            "commits": dict(self.commits),
            "aborts": dict(self.aborts),
            "rollbacks": dict(self.rollbacks),
            "failures": dict(self.failures),
            "totals": {
                "admitted_lanes": self.admitted_lanes,
                "proposed_draft_tokens": self.proposed_draft_tokens,
                "accepted_draft_tokens": self.accepted_draft_tokens,
                "committed_tokens": self.committed_tokens,
                "terminal_lanes": self.terminal_lanes,
                "cleaned_lanes": self.cleaned_lanes,
                "draft_seconds": self.draft_seconds,
                "target_verify_seconds": self.target_verify_seconds,
                "accept_ratio": self.accept_ratio,
                "open_transaction": self.open_transaction,
            },
        }


class ContinuousMTPCounters:
    """Thread-safe, bounded counters around one transaction stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._admissions = {value: 0 for value in AdmissionOutcome}
        self._admission_reasons = {value: 0 for value in AdmissionReason}
        self._transactions = {value: 0 for value in TransactionOutcome}
        self._commits = {value: 0 for value in CommitKind}
        self._aborts = {value: 0 for value in AbortReason}
        self._rollbacks = {value: 0 for value in RollbackPhase}
        self._failures = {value: 0 for value in FailurePhase}
        self._admitted_lanes = 0
        self._proposed_draft_tokens = 0
        self._accepted_draft_tokens = 0
        self._committed_tokens = 0
        self._terminal_lanes = 0
        self._cleaned_lanes = 0
        self._draft_seconds = 0.0
        self._target_verify_seconds = 0.0
        self._next_sequence = 1
        self._open: TransactionTicket | None = None

    def record_admission(
        self,
        outcome: AdmissionOutcome,
        reason: AdmissionReason,
        *,
        lanes: int,
        draft_depth: int,
    ) -> None:
        """Record one bounded admission/fallback decision."""

        _require_enum(outcome, AdmissionOutcome, "outcome")
        _require_enum(reason, AdmissionReason, "reason")
        lanes = _positive_int(lanes, "lanes")
        draft_depth = _non_negative_int(draft_depth, "draft_depth")
        if outcome is AdmissionOutcome.BATCHED:
            if reason not in (
                AdmissionReason.ELIGIBLE,
                AdmissionReason.DEPTH_REDUCED,
            ):
                raise ValueError("batched admission requires an eligible reason")
            if draft_depth < 1:
                raise ValueError("batched admission requires positive draft_depth")
        else:
            if reason in (
                AdmissionReason.ELIGIBLE,
                AdmissionReason.DEPTH_REDUCED,
            ):
                raise ValueError("fallback/refusal requires a refusal reason")
            if draft_depth != 0:
                raise ValueError("non-batched admission must report draft_depth=0")
        with self._lock:
            self._admissions[outcome] += 1
            self._admission_reasons[reason] += 1
            if outcome is AdmissionOutcome.BATCHED:
                self._admitted_lanes += lanes

    def begin_transaction(
        self,
        *,
        lanes: int,
        proposed_draft_tokens: int,
    ) -> TransactionTicket:
        """Open one proposal and return its single-use ticket."""

        lanes = _positive_int(lanes, "lanes")
        proposed = _positive_int(proposed_draft_tokens, "proposed_draft_tokens")
        with self._lock:
            if self._open is not None:
                raise RuntimeError("a telemetry transaction is already open")
            ticket = TransactionTicket(self._next_sequence, lanes, proposed)
            self._next_sequence += 1
            self._open = ticket
            self._transactions[TransactionOutcome.PROPOSED] += 1
            self._proposed_draft_tokens += proposed
            return ticket

    def commit_transaction(
        self,
        ticket: TransactionTicket,
        *,
        accepted_draft_tokens: int,
        committed_tokens: int,
        terminal_lanes: int = 0,
        kind: CommitKind = CommitKind.FULL,
    ) -> None:
        """Close a proposal after validating aggregate delivery counts."""

        _require_enum(kind, CommitKind, "kind")
        accepted = _non_negative_int(accepted_draft_tokens, "accepted_draft_tokens")
        committed = _non_negative_int(committed_tokens, "committed_tokens")
        terminal = _non_negative_int(terminal_lanes, "terminal_lanes")
        with self._lock:
            self._require_ticket(ticket)
            if accepted > ticket.proposed_draft_tokens:
                raise ValueError("accepted drafts cannot exceed proposed drafts")
            if accepted > committed:
                raise ValueError("accepted drafts cannot exceed committed tokens")
            if terminal > ticket.lanes:
                raise ValueError("terminal_lanes cannot exceed transaction lanes")
            if kind is CommitKind.TERMINAL_PARTIAL and terminal == 0:
                raise ValueError("terminal partial commit requires a terminal lane")
            self._transactions[TransactionOutcome.COMMITTED] += 1
            self._commits[kind] += 1
            self._accepted_draft_tokens += accepted
            self._committed_tokens += committed
            self._terminal_lanes += terminal
            self._open = None

    def abort_transaction(
        self,
        ticket: TransactionTicket,
        *,
        reason: AbortReason,
        failed: bool = False,
    ) -> None:
        """Close a proposal without publishing commit totals."""

        _require_enum(reason, AbortReason, "reason")
        if not isinstance(failed, bool):
            raise ValueError("failed must be a boolean")
        with self._lock:
            self._require_ticket(ticket)
            outcome = (
                TransactionOutcome.FAILED if failed else TransactionOutcome.ABORTED
            )
            self._transactions[outcome] += 1
            self._aborts[reason] += 1
            self._open = None

    def record_cleanup(self, *, lanes: int) -> None:
        lanes = _positive_int(lanes, "lanes")
        with self._lock:
            self._cleaned_lanes += lanes

    def record_cycle(
        self,
        *,
        proposed_draft_tokens: int,
        accepted_draft_tokens: int,
        committed_tokens: int,
        verify_rollbacks: int,
        delivery_rollbacks: int,
        draft_seconds: float,
        target_verify_seconds: float,
        kind: CommitKind = CommitKind.FULL,
    ) -> None:
        """Atomically publish one successfully committed synthetic cycle.

        Production uses the split proposal/commit methods so a failed commit
        does not erase completed draft/verify work.  This convenience API keeps
        synthetic reconciliation records atomic and never evaluates an MLX
        array.
        """

        proposed = _positive_int(proposed_draft_tokens, "proposed_draft_tokens")
        accepted = _non_negative_int(accepted_draft_tokens, "accepted_draft_tokens")
        committed = _positive_int(committed_tokens, "committed_tokens")
        verify = _non_negative_int(verify_rollbacks, "verify_rollbacks")
        delivery = _non_negative_int(delivery_rollbacks, "delivery_rollbacks")
        draft_elapsed = _non_negative_float(draft_seconds, "draft_seconds")
        verify_elapsed = _non_negative_float(
            target_verify_seconds, "target_verify_seconds"
        )
        _require_enum(kind, CommitKind, "kind")
        if accepted > proposed:
            raise ValueError("accepted drafts cannot exceed proposed drafts")
        with self._lock:
            self._transactions[TransactionOutcome.PROPOSED] += 1
            self._transactions[TransactionOutcome.COMMITTED] += 1
            self._commits[kind] += 1
            self._proposed_draft_tokens += proposed
            self._accepted_draft_tokens += accepted
            self._committed_tokens += committed
            self._rollbacks[RollbackPhase.VERIFY] += verify
            self._rollbacks[RollbackPhase.DELIVERY] += delivery
            self._draft_seconds += draft_elapsed
            self._target_verify_seconds += verify_elapsed

    def record_proposal(
        self,
        *,
        proposed_draft_tokens: int,
        verify_rollbacks: int,
        draft_seconds: float,
        target_verify_seconds: float,
    ) -> None:
        """Publish a validated production proposal before its commit attempt."""

        proposed = _positive_int(proposed_draft_tokens, "proposed_draft_tokens")
        verify = _non_negative_int(verify_rollbacks, "verify_rollbacks")
        draft_elapsed = _non_negative_float(draft_seconds, "draft_seconds")
        target_elapsed = _non_negative_float(
            target_verify_seconds, "target_verify_seconds"
        )
        with self._lock:
            self._transactions[TransactionOutcome.PROPOSED] += 1
            self._proposed_draft_tokens += proposed
            self._rollbacks[RollbackPhase.VERIFY] += verify
            self._draft_seconds += draft_elapsed
            self._target_verify_seconds += target_elapsed

    def record_commit(
        self,
        *,
        accepted_draft_tokens: int,
        committed_tokens: int,
        delivery_rollbacks: int,
        kind: CommitKind = CommitKind.FULL,
    ) -> None:
        """Publish the delivery side of a successfully committed proposal."""

        accepted = _non_negative_int(accepted_draft_tokens, "accepted_draft_tokens")
        committed = _positive_int(committed_tokens, "committed_tokens")
        delivery = _non_negative_int(delivery_rollbacks, "delivery_rollbacks")
        _require_enum(kind, CommitKind, "kind")
        with self._lock:
            if self._accepted_draft_tokens + accepted > self._proposed_draft_tokens:
                raise ValueError("cumulative accepted drafts exceed proposals")
            self._transactions[TransactionOutcome.COMMITTED] += 1
            self._commits[kind] += 1
            self._accepted_draft_tokens += accepted
            self._committed_tokens += committed
            self._rollbacks[RollbackPhase.DELIVERY] += delivery

    def record_failure(self, phase: FailurePhase) -> None:
        """Record one failed production proposal or commit boundary."""

        _require_enum(phase, FailurePhase, "phase")
        with self._lock:
            self._transactions[TransactionOutcome.FAILED] += 1
            self._failures[phase] += 1

    def snapshot(self) -> ContinuousMTPSnapshot:
        with self._lock:
            return ContinuousMTPSnapshot(
                admissions=_enum_counts(self._admissions),
                admission_reasons=_enum_counts(self._admission_reasons),
                transactions=_enum_counts(self._transactions),
                commits=_enum_counts(self._commits),
                aborts=_enum_counts(self._aborts),
                rollbacks=_enum_counts(self._rollbacks),
                failures=_enum_counts(self._failures),
                admitted_lanes=self._admitted_lanes,
                proposed_draft_tokens=self._proposed_draft_tokens,
                accepted_draft_tokens=self._accepted_draft_tokens,
                committed_tokens=self._committed_tokens,
                terminal_lanes=self._terminal_lanes,
                cleaned_lanes=self._cleaned_lanes,
                draft_seconds=self._draft_seconds,
                target_verify_seconds=self._target_verify_seconds,
                open_transaction=self._open is not None,
            )

    def reset(self) -> None:
        """Reset this counter instance for tests only."""

        with self._lock:
            for counts in (
                self._admissions,
                self._admission_reasons,
                self._transactions,
                self._commits,
                self._aborts,
                self._rollbacks,
                self._failures,
            ):
                for key in counts:
                    counts[key] = 0
            self._admitted_lanes = 0
            self._proposed_draft_tokens = 0
            self._accepted_draft_tokens = 0
            self._committed_tokens = 0
            self._terminal_lanes = 0
            self._cleaned_lanes = 0
            self._draft_seconds = 0.0
            self._target_verify_seconds = 0.0
            self._next_sequence = 1
            self._open = None

    def _require_ticket(self, ticket: TransactionTicket) -> None:
        if not isinstance(ticket, TransactionTicket) or ticket != self._open:
            raise RuntimeError("stale, foreign, or already closed transaction")


@dataclass(frozen=True)
class QualificationIdentity:
    """Exact offline identity for one candidate/model/config battery."""

    candidate_sha: str
    model_id: str
    model_revision: str
    config_fingerprint: str
    prompt_manifest_sha256: str
    environment_fingerprint: str

    def __post_init__(self) -> None:
        _require_hex(self.candidate_sha, 40, "candidate_sha")
        for name in ("model_id", "model_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "config_fingerprint",
            "prompt_manifest_sha256",
            "environment_fingerprint",
        ):
            _require_hex(getattr(self, name), 64, name)


@dataclass(frozen=True)
class LaneQualification:
    lane_index: int
    output_tokens: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    elapsed_seconds: float
    batched_b1_digest: DigestClassification
    batch_shape_digest: DigestClassification
    legacy_single_lane_digest: DigestClassification
    finish_reason: FinishReason

    def __post_init__(self) -> None:
        _non_negative_int(self.lane_index, "lane_index")
        _non_negative_int(self.output_tokens, "output_tokens")
        proposed = _non_negative_int(
            self.proposed_draft_tokens, "proposed_draft_tokens"
        )
        accepted = _non_negative_int(
            self.accepted_draft_tokens, "accepted_draft_tokens"
        )
        if accepted > proposed:
            raise ValueError("lane accepted drafts cannot exceed proposals")
        _positive_float(self.elapsed_seconds, "elapsed_seconds")
        _require_enum(
            self.batched_b1_digest,
            DigestClassification,
            "batched_b1_digest",
        )
        _require_enum(
            self.batch_shape_digest,
            DigestClassification,
            "batch_shape_digest",
        )
        _require_enum(
            self.legacy_single_lane_digest,
            DigestClassification,
            "legacy_single_lane_digest",
        )
        _require_enum(self.finish_reason, FinishReason, "finish_reason")


@dataclass(frozen=True)
class CohortQualification:
    """One N-lane result plus the counters produced during that run."""

    identity: QualificationIdentity
    evidence_kind: EvidenceKind
    cohort_size: int
    draft_depth: int
    aggregate_elapsed_seconds: float
    active_memory_bytes: int
    peak_memory_bytes: int
    cache_equal: bool
    lanes: tuple[LaneQualification, ...]
    telemetry: ContinuousMTPSnapshot
    raw_artifact_sha256: str | None = None

    @property
    def aggregate_tokens(self) -> int:
        return sum(lane.output_tokens for lane in self.lanes)

    @property
    def aggregate_tokens_per_second(self) -> float:
        return self.aggregate_tokens / self.aggregate_elapsed_seconds


@dataclass(frozen=True)
class QualificationResult:
    structurally_valid: bool
    performance_qualified: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class QualificationSuiteResult:
    structurally_complete: bool
    performance_qualified: bool
    errors: tuple[str, ...]


def evaluate_cohort_qualification(
    record: CohortQualification,
) -> QualificationResult:
    """Reconcile one offline record without executing a model."""

    errors: list[str] = []
    if not isinstance(record, CohortQualification):
        return QualificationResult(False, False, ("record type",))
    if record.cohort_size not in (1, 2, 4):
        errors.append("cohort_size must be one of 1, 2, 4")
    if record.draft_depth < 1:
        errors.append("draft_depth must be positive")
    if not math.isfinite(record.aggregate_elapsed_seconds) or (
        record.aggregate_elapsed_seconds <= 0
    ):
        errors.append("aggregate elapsed time must be positive")
    if record.active_memory_bytes < 0 or (
        record.peak_memory_bytes < record.active_memory_bytes
    ):
        errors.append("memory totals are inconsistent")
    if not isinstance(record.cache_equal, bool):
        errors.append("cache_equal must be boolean")
    expected_indices = tuple(range(record.cohort_size))
    observed_indices = tuple(sorted(lane.lane_index for lane in record.lanes))
    if observed_indices != expected_indices:
        errors.append("lane indices must cover the cohort exactly once")

    proposed = sum(lane.proposed_draft_tokens for lane in record.lanes)
    accepted = sum(lane.accepted_draft_tokens for lane in record.lanes)
    committed = sum(lane.output_tokens for lane in record.lanes)
    snapshot = record.telemetry
    if snapshot.proposed_draft_tokens != proposed:
        errors.append("proposed draft totals do not reconcile")
    if snapshot.accepted_draft_tokens != accepted:
        errors.append("accepted draft totals do not reconcile")
    if snapshot.committed_tokens != committed:
        errors.append("committed token totals do not reconcile")
    transaction_counts = dict(snapshot.transactions)
    if transaction_counts.get(TransactionOutcome.PROPOSED.value) != 1:
        errors.append("qualification run must contain one proposal")
    if transaction_counts.get(TransactionOutcome.COMMITTED.value) != 1:
        errors.append("qualification run must contain one commit")
    if transaction_counts.get(TransactionOutcome.ABORTED.value) != 0 or (
        transaction_counts.get(TransactionOutcome.FAILED.value) != 0
    ):
        errors.append("qualification run contains an abort or failure")
    admissions = dict(snapshot.admissions)
    if admissions.get(AdmissionOutcome.BATCHED.value) != 1:
        errors.append("qualification run must contain one batched admission")
    if snapshot.open_transaction:
        errors.append("qualification run has an open transaction")
    if not record.cache_equal:
        errors.append("cache equality gate failed")
    if any(
        lane.batched_b1_digest is not DigestClassification.EXACT
        for lane in record.lanes
    ):
        errors.append("batched-B1 exact digest gate failed")
    if any(
        lane.batch_shape_digest
        in (DigestClassification.DIVERGENT, DigestClassification.NOT_RUN)
        for lane in record.lanes
    ):
        errors.append("B>1 batch-shape digest gate failed")
    if record.cohort_size == 1 and any(
        lane.batch_shape_digest is not DigestClassification.EXACT
        for lane in record.lanes
    ):
        errors.append("batched-B1 cannot use a batch-shape tolerance")

    structurally_valid = not errors
    hardware_evidence = (
        record.evidence_kind is EvidenceKind.APPLE_SILICON_HARDWARE
        and isinstance(record.raw_artifact_sha256, str)
        and _is_hex(record.raw_artifact_sha256, 64)
    )
    return QualificationResult(
        structurally_valid=structurally_valid,
        performance_qualified=structurally_valid and hardware_evidence,
        errors=tuple(errors),
    )


def evaluate_qualification_suite(
    records: tuple[CohortQualification, ...],
) -> QualificationSuiteResult:
    """Require one identity-matched N=1/2/4 battery."""

    errors: list[str] = []
    if not records:
        return QualificationSuiteResult(False, False, ("suite is empty",))
    sizes = tuple(sorted(record.cohort_size for record in records))
    if sizes != (1, 2, 4):
        errors.append("suite must contain exactly N=1,2,4")
    identity = records[0].identity
    if any(record.identity != identity for record in records[1:]):
        errors.append("suite identities differ")
    results = tuple(evaluate_cohort_qualification(record) for record in records)
    for index, result in enumerate(results):
        if not result.structurally_valid:
            errors.append(f"cohort[{index}] is structurally invalid")
    structurally_complete = not errors
    return QualificationSuiteResult(
        structurally_complete=structurally_complete,
        performance_qualified=(
            structurally_complete
            and all(result.performance_qualified for result in results)
        ),
        errors=tuple(errors),
    )


def _enum_counts(values: dict[Enum, int]) -> tuple[tuple[str, int], ...]:
    return tuple((key.value, values[key]) for key in type(next(iter(values))))


def _require_enum(value: Any, expected: type[Enum], name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return numeric


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_hex(value: Any, length: int, name: str) -> None:
    if not _is_hex(value, length):
        raise ValueError(f"{name} must be {length} lowercase hex characters")


BOUNDED_METRIC_DIMENSIONS = MappingProxyType(
    {
        "admission_outcome": tuple(value.value for value in AdmissionOutcome),
        "admission_reason": tuple(value.value for value in AdmissionReason),
        "transaction_outcome": tuple(value.value for value in TransactionOutcome),
        "commit_kind": tuple(value.value for value in CommitKind),
        "abort_reason": tuple(value.value for value in AbortReason),
        "rollback_phase": tuple(value.value for value in RollbackPhase),
        "failure_phase": tuple(value.value for value in FailurePhase),
    }
)


_global_counter = ContinuousMTPCounters()


def get_global_continuous_counter() -> ContinuousMTPCounters:
    """Return the process-global continuous-engine counter registry."""

    return _global_counter


def reset_global_continuous_counter_for_tests() -> None:
    """Reset the process-global registry for isolated tests only."""

    _global_counter.reset()


__all__ = [
    "BOUNDED_METRIC_DIMENSIONS",
    "AbortReason",
    "AdmissionOutcome",
    "AdmissionReason",
    "CohortQualification",
    "CommitKind",
    "ContinuousMTPCounters",
    "ContinuousMTPSnapshot",
    "DigestClassification",
    "EvidenceKind",
    "FailurePhase",
    "FinishReason",
    "LaneQualification",
    "QualificationIdentity",
    "QualificationResult",
    "QualificationSuiteResult",
    "RollbackPhase",
    "TransactionOutcome",
    "TransactionTicket",
    "evaluate_cohort_qualification",
    "evaluate_qualification_suite",
    "get_global_continuous_counter",
    "reset_global_continuous_counter_for_tests",
]
