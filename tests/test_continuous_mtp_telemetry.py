# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.spec_decode.mtp.continuous_telemetry import (
    BOUNDED_METRIC_DIMENSIONS,
    AbortReason,
    AdmissionOutcome,
    AdmissionReason,
    CohortQualification,
    CommitKind,
    ContinuousMTPCounters,
    DigestClassification,
    EvidenceKind,
    FinishReason,
    LaneQualification,
    QualificationIdentity,
    TransactionOutcome,
    TransactionTicket,
    evaluate_cohort_qualification,
    evaluate_qualification_suite,
)


def _identity(**changes) -> QualificationIdentity:
    values = {
        "candidate_sha": "a" * 40,
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "model_revision": "f5d08274",
        "config_fingerprint": "b" * 64,
        "prompt_manifest_sha256": "c" * 64,
        "environment_fingerprint": "d" * 64,
    }
    values.update(changes)
    return QualificationIdentity(**values)


def _clean_record(
    cohort_size: int = 4,
    *,
    identity: QualificationIdentity | None = None,
    evidence_kind: EvidenceKind = EvidenceKind.SYNTHETIC,
    raw_artifact_sha256: str | None = None,
) -> CohortQualification:
    counters = ContinuousMTPCounters()
    counters.record_admission(
        AdmissionOutcome.BATCHED,
        AdmissionReason.ELIGIBLE,
        lanes=cohort_size,
        draft_depth=2,
    )
    ticket = counters.begin_transaction(
        lanes=cohort_size,
        proposed_draft_tokens=cohort_size * 2,
    )
    counters.commit_transaction(
        ticket,
        accepted_draft_tokens=cohort_size,
        committed_tokens=cohort_size * 2,
    )
    counters.record_cleanup(lanes=cohort_size)
    lanes = tuple(
        LaneQualification(
            lane_index=index,
            output_tokens=2,
            proposed_draft_tokens=2,
            accepted_draft_tokens=1,
            elapsed_seconds=1.0 + index / 10,
            digest=DigestClassification.EXACT,
            finish_reason=FinishReason.STOP,
        )
        for index in range(cohort_size)
    )
    return CohortQualification(
        identity=identity or _identity(),
        evidence_kind=evidence_kind,
        cohort_size=cohort_size,
        draft_depth=2,
        aggregate_elapsed_seconds=1.0,
        active_memory_bytes=100,
        peak_memory_bytes=120,
        cache_equal=True,
        lanes=lanes,
        telemetry=counters.snapshot(),
        raw_artifact_sha256=raw_artifact_sha256,
    )


def test_metric_dimensions_are_fixed_enums_without_identity_labels() -> None:
    assert set(BOUNDED_METRIC_DIMENSIONS) == {
        "admission_outcome",
        "admission_reason",
        "transaction_outcome",
        "commit_kind",
        "abort_reason",
    }
    flattened = {
        value for dimension in BOUNDED_METRIC_DIMENSIONS.values() for value in dimension
    }
    assert "request_id" not in flattened
    assert "model_id" not in flattened
    with pytest.raises(TypeError):
        BOUNDED_METRIC_DIMENSIONS["request_id"] = ("uid-1",)


def test_zero_snapshot_exports_every_bounded_counter() -> None:
    payload = ContinuousMTPCounters().snapshot().to_dict()

    assert payload["admissions"] == {value.value: 0 for value in AdmissionOutcome}
    assert payload["transactions"] == {value.value: 0 for value in TransactionOutcome}
    assert payload["totals"]["open_transaction"] is False


def test_admission_counts_batch_fallback_queue_and_refusal() -> None:
    counters = ContinuousMTPCounters()
    counters.record_admission(
        AdmissionOutcome.BATCHED,
        AdmissionReason.DEPTH_REDUCED,
        lanes=4,
        draft_depth=1,
    )
    counters.record_admission(
        AdmissionOutcome.PLAIN,
        AdmissionReason.SAMPLING,
        lanes=1,
        draft_depth=0,
    )
    counters.record_admission(
        AdmissionOutcome.QUEUED,
        AdmissionReason.MEMORY,
        lanes=2,
        draft_depth=0,
    )
    counters.record_admission(
        AdmissionOutcome.REFUSED,
        AdmissionReason.CAPABILITY,
        lanes=1,
        draft_depth=0,
    )

    payload = counters.snapshot().to_dict()
    assert payload["admissions"] == {
        "batched": 1,
        "plain": 1,
        "queued": 1,
        "refused": 1,
    }
    assert payload["totals"]["admitted_lanes"] == 4


@pytest.mark.parametrize(
    "outcome,reason,draft_depth",
    [
        ("batched", AdmissionReason.ELIGIBLE, 2),
        (AdmissionOutcome.BATCHED, AdmissionReason.MEMORY, 2),
        (AdmissionOutcome.PLAIN, AdmissionReason.ELIGIBLE, 0),
        (AdmissionOutcome.QUEUED, AdmissionReason.MEMORY, 1),
    ],
)
def test_admission_rejects_unbounded_or_incoherent_labels(
    outcome, reason, draft_depth
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ContinuousMTPCounters().record_admission(
            outcome,
            reason,
            lanes=1,
            draft_depth=draft_depth,
        )


def test_proposal_commit_and_cleanup_reconcile() -> None:
    counters = ContinuousMTPCounters()
    ticket = counters.begin_transaction(lanes=4, proposed_draft_tokens=8)
    assert counters.snapshot().open_transaction is True

    counters.commit_transaction(
        ticket,
        accepted_draft_tokens=5,
        committed_tokens=9,
        terminal_lanes=1,
        kind=CommitKind.TERMINAL_PARTIAL,
    )
    counters.record_cleanup(lanes=1)

    snapshot = counters.snapshot()
    assert dict(snapshot.transactions) == {
        "proposed": 1,
        "committed": 1,
        "aborted": 0,
        "failed": 0,
    }
    assert dict(snapshot.commits) == {"full": 0, "terminal_partial": 1}
    assert snapshot.proposed_draft_tokens == 8
    assert snapshot.accepted_draft_tokens == 5
    assert snapshot.committed_tokens == 9
    assert snapshot.terminal_lanes == 1
    assert snapshot.cleaned_lanes == 1
    assert snapshot.open_transaction is False


def test_abort_is_single_use_and_does_not_publish_commit_totals() -> None:
    counters = ContinuousMTPCounters()
    ticket = counters.begin_transaction(lanes=2, proposed_draft_tokens=4)

    counters.abort_transaction(ticket, reason=AbortReason.DISCONNECTED)

    snapshot = counters.snapshot()
    assert dict(snapshot.transactions)["aborted"] == 1
    assert dict(snapshot.aborts)["disconnected"] == 1
    assert snapshot.committed_tokens == 0
    with pytest.raises(RuntimeError, match="stale"):
        counters.abort_transaction(ticket, reason=AbortReason.CALLER)


def test_failed_transaction_is_distinct_from_benign_abort() -> None:
    counters = ContinuousMTPCounters()
    ticket = counters.begin_transaction(lanes=1, proposed_draft_tokens=2)

    counters.abort_transaction(
        ticket,
        reason=AbortReason.COMPUTE_ERROR,
        failed=True,
    )

    snapshot = counters.snapshot()
    assert dict(snapshot.transactions)["failed"] == 1
    assert dict(snapshot.transactions)["aborted"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted_draft_tokens": 9},
        {"accepted_draft_tokens": 5, "committed_tokens": 4},
        {"terminal_lanes": 5},
        {"kind": CommitKind.TERMINAL_PARTIAL, "terminal_lanes": 0},
    ],
)
def test_invalid_commit_does_not_close_transaction(changes) -> None:
    counters = ContinuousMTPCounters()
    ticket = counters.begin_transaction(lanes=4, proposed_draft_tokens=8)
    kwargs = {
        "accepted_draft_tokens": 4,
        "committed_tokens": 8,
        "terminal_lanes": 0,
        "kind": CommitKind.FULL,
    }
    kwargs.update(changes)

    with pytest.raises(ValueError):
        counters.commit_transaction(ticket, **kwargs)

    assert counters.snapshot().open_transaction is True
    counters.abort_transaction(ticket, reason=AbortReason.INVARIANT, failed=True)


def test_foreign_ticket_cannot_mutate_counters() -> None:
    counters = ContinuousMTPCounters()
    ticket = counters.begin_transaction(lanes=1, proposed_draft_tokens=2)
    foreign = TransactionTicket(ticket.sequence + 1, 1, 2)

    with pytest.raises(RuntimeError, match="foreign"):
        counters.commit_transaction(
            foreign,
            accepted_draft_tokens=1,
            committed_tokens=2,
        )

    counters.abort_transaction(ticket, reason=AbortReason.CALLER)


def test_clean_synthetic_record_reconciles_but_cannot_qualify_performance() -> None:
    record = _clean_record()

    result = evaluate_cohort_qualification(record)

    assert result.structurally_valid is True
    assert result.errors == ()
    assert result.performance_qualified is False
    assert record.aggregate_tokens == 8
    assert record.aggregate_tokens_per_second == 8.0


@pytest.mark.parametrize(
    "record_change,error",
    [
        ({"cache_equal": False}, "cache equality"),
        ({"peak_memory_bytes": 99}, "memory totals"),
        ({"lanes": ()}, "lane indices"),
    ],
)
def test_qualification_refuses_failed_structural_gates(record_change, error) -> None:
    record = replace(_clean_record(), **record_change)

    result = evaluate_cohort_qualification(record)

    assert result.structurally_valid is False
    assert any(error in message for message in result.errors)
    assert result.performance_qualified is False


def test_qualification_reconciles_lane_and_counter_totals() -> None:
    record = _clean_record()
    bad_lane = replace(record.lanes[0], output_tokens=3)
    record = replace(record, lanes=(bad_lane,) + record.lanes[1:])

    result = evaluate_cohort_qualification(record)

    assert result.structurally_valid is False
    assert "committed token totals do not reconcile" in result.errors


def test_divergent_or_unrun_digest_refuses_structural_record() -> None:
    record = _clean_record()
    for classification in (
        DigestClassification.DIVERGENT,
        DigestClassification.NOT_RUN,
    ):
        lane = replace(record.lanes[0], digest=classification)
        changed = replace(record, lanes=(lane,) + record.lanes[1:])
        result = evaluate_cohort_qualification(changed)
        assert "digest classification gate failed" in result.errors


def test_hardware_label_without_artifact_digest_cannot_qualify() -> None:
    record = _clean_record(evidence_kind=EvidenceKind.APPLE_SILICON_HARDWARE)

    result = evaluate_cohort_qualification(record)

    assert result.structurally_valid is True
    assert result.performance_qualified is False


def test_suite_requires_identity_matched_n1_n2_n4() -> None:
    identity = _identity()
    records = tuple(_clean_record(size, identity=identity) for size in (1, 2, 4))

    result = evaluate_qualification_suite(records)

    assert result.structurally_complete is True
    assert result.performance_qualified is False
    assert result.errors == ()

    wrong_identity = _identity(model_revision="different")
    mismatched = records[:2] + (_clean_record(4, identity=wrong_identity),)
    mismatch_result = evaluate_qualification_suite(mismatched)
    assert mismatch_result.structurally_complete is False
    assert "suite identities differ" in mismatch_result.errors


def test_suite_refuses_missing_cohort() -> None:
    records = (_clean_record(1), _clean_record(4))

    result = evaluate_qualification_suite(records)

    assert result.structurally_complete is False
    assert "suite must contain exactly N=1,2,4" in result.errors


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"candidate_sha": "a" * 39}, "candidate_sha"),
        ({"candidate_sha": "A" * 40}, "candidate_sha"),
        ({"model_id": ""}, "model_id"),
        ({"config_fingerprint": "x" * 64}, "config_fingerprint"),
    ],
)
def test_qualification_identity_requires_exact_immutable_fingerprints(
    changes, field
) -> None:
    with pytest.raises(ValueError, match=field):
        _identity(**changes)
