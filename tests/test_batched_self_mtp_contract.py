# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for the future continuous self-MTP coordinator."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.spec_decode.mtp.batched import (
    BatchedMTPBookkeeper,
    BatchedMTPCapabilities,
    BatchedMTPConfig,
    BatchedMTPRoute,
    BatchedMTPTransactionError,
    LaneAdmission,
    SamplingContract,
    assess_lane,
    plan_admission,
)


def _capabilities(**overrides) -> BatchedMTPCapabilities:
    values = dict(
        target_batch_forward=True,
        mtp_batch_forward=True,
        ragged_target_rollback=True,
        ragged_mtp_rollback=True,
        atomic_cache_commit=True,
        per_lane_rng=True,
        transformed_distribution_verify=True,
        dynamic_membership=True,
    )
    values.update(overrides)
    return BatchedMTPCapabilities(**values)


def _config(**overrides) -> BatchedMTPConfig:
    values = dict(enabled=True, hard_reserve_bytes=100, max_draft_tokens=2)
    values.update(overrides)
    return BatchedMTPConfig(**values)


def test_feature_is_default_off_and_capabilities_fail_closed():
    lane = LaneAdmission("a")
    gate = assess_lane(
        lane,
        config=BatchedMTPConfig(),
        capabilities=BatchedMTPCapabilities(),
    )

    assert not gate.eligible
    assert "batched self-MTP is disabled" in gate.reasons
    assert "missing capability: target_batch_forward" in gate.reasons


def test_truthy_non_boolean_capability_does_not_open_gate():
    caps = replace(_capabilities(), target_batch_forward="yes")
    gate = assess_lane(LaneAdmission("a"), config=_config(), capabilities=caps)
    assert gate.reasons == ("missing capability: target_batch_forward",)


@pytest.mark.parametrize(
    "missing",
    [
        "target_batch_forward",
        "mtp_batch_forward",
        "ragged_target_rollback",
        "ragged_mtp_rollback",
        "atomic_cache_commit",
    ],
)
def test_each_core_capability_is_mandatory(missing):
    gate = assess_lane(
        LaneAdmission("a"),
        config=_config(),
        capabilities=replace(_capabilities(), **{missing: False}),
    )
    assert not gate.eligible
    assert gate.reasons == (f"missing capability: {missing}",)


def test_sampled_lanes_require_rng_and_transformed_verifier():
    lane = LaneAdmission("sampled", sampling=SamplingContract(greedy=False))
    gate = assess_lane(
        lane,
        config=_config(),
        capabilities=_capabilities(
            per_lane_rng=False, transformed_distribution_verify=False
        ),
    )
    assert gate.reasons == (
        "missing capability: per_lane_rng",
        "missing capability: transformed_distribution_verify",
    )


def test_xtc_and_logits_processors_fail_closed():
    xtc = LaneAdmission("xtc", sampling=SamplingContract(greedy=False, uses_xtc=True))
    processors = LaneAdmission(
        "processors", sampling=SamplingContract(has_logits_processors=True)
    )
    caps = _capabilities(xtc_exact_verify=False)

    assert assess_lane(xtc, config=_config(), capabilities=caps).reasons == (
        "XTC verification is not exact",
    )
    assert assess_lane(processors, config=_config(), capabilities=caps).reasons == (
        "logits processors are not supported",
    )


def test_dynamic_membership_requires_explicit_capability():
    gate = assess_lane(
        LaneAdmission("a"),
        config=_config(allow_dynamic_membership=True),
        capabilities=_capabilities(dynamic_membership=False),
    )
    assert gate.reasons == ("missing capability: dynamic_membership",)


def test_admission_lowers_depth_to_admit_more_lanes():
    lanes = [
        LaneAdmission(str(index), base_bytes=10, bytes_per_draft_token=20)
        for index in range(4)
    ]
    # usable=140: K=2 fits 2 lanes (50 each), K=1 fits 4 lanes (30 each).
    decision = plan_admission(
        lanes,
        config=_config(),
        capabilities=_capabilities(),
        free_bytes=240,
    )

    assert decision.route is BatchedMTPRoute.BATCHED_MTP
    assert decision.batched_lane_ids == ("0", "1", "2", "3")
    assert decision.draft_tokens == 1
    assert decision.estimated_bytes == 120


def test_admission_keeps_ineligible_lanes_plain_and_overflow_queued():
    lanes = [
        LaneAdmission("plain", cache_ready=False),
        LaneAdmission("a", base_bytes=60),
        LaneAdmission("b", base_bytes=60),
        LaneAdmission("c", base_bytes=60),
    ]
    decision = plan_admission(
        lanes,
        config=_config(max_lanes=3),
        capabilities=_capabilities(),
        free_bytes=230,
    )

    assert decision.batched_lane_ids == ("a", "b")
    assert decision.plain_lane_ids == ("plain",)
    assert decision.queued_lane_ids == ("c",)


def test_lanes_over_configured_limit_are_explicitly_queued():
    lanes = [LaneAdmission(str(index)) for index in range(5)]
    decision = plan_admission(
        lanes,
        config=_config(max_lanes=3),
        capabilities=_capabilities(),
        free_bytes=1000,
    )

    assert decision.batched_lane_ids == ("0", "1", "2")
    assert decision.queued_lane_ids == ("3", "4")


def test_memory_reserve_queues_an_otherwise_eligible_batch():
    lanes = [LaneAdmission("a", base_bytes=1), LaneAdmission("b", base_bytes=1)]
    decision = plan_admission(
        lanes,
        config=_config(),
        capabilities=_capabilities(),
        free_bytes=100,
    )

    assert decision.route is BatchedMTPRoute.QUEUE
    assert decision.queued_lane_ids == ("a", "b")
    assert decision.draft_tokens == 0


def test_single_eligible_lane_uses_plain_decode_instead_of_waiting():
    decision = plan_admission(
        [LaneAdmission("a")],
        config=_config(),
        capabilities=_capabilities(),
        free_bytes=1000,
    )
    assert decision.route is BatchedMTPRoute.PLAIN_DECODE
    assert decision.plain_lane_ids == ("a",)


def test_proposal_ticket_binds_epoch_order_and_verify_shape():
    ledger = BatchedMTPBookkeeper(("a", "b", "c"))
    ticket = ledger.begin_proposal(draft_tokens=2)

    assert ticket.membership_epoch == 1
    assert ticket.lane_ids == ("a", "b", "c")
    assert ticket.verify_rows == 9

    receipt = ledger.commit(
        ticket,
        emitted_counts={"c": 3, "a": 1, "b": 2},
        terminal_lane_ids=("c",),
    )
    assert receipt.emitted_counts == (("a", 1), ("b", 2), ("c", 3))
    assert receipt.total_emitted == 6
    assert ledger.committed_tokens("b") == 2
    # Commit reports terminal lanes; the coordinator detaches them only after
    # it has committed the corresponding model/cache transaction.
    assert ledger.lane_ids == ("a", "b", "c")


def test_membership_cannot_change_during_proposal_and_abort_unlocks_it():
    ledger = BatchedMTPBookkeeper(("a", "b"))
    ticket = ledger.begin_proposal(draft_tokens=1)

    with pytest.raises(BatchedMTPTransactionError, match="membership cannot change"):
        ledger.attach(("c",))
    with pytest.raises(BatchedMTPTransactionError, match="membership cannot change"):
        ledger.detach(("a",))

    ledger.abort(ticket)
    assert ledger.attach(("c",)) == 2
    assert ledger.lane_ids == ("a", "b", "c")


def test_stale_or_double_commit_is_rejected():
    ledger = BatchedMTPBookkeeper(("a", "b"))
    first = ledger.begin_proposal(draft_tokens=1)
    ledger.abort(first)
    second = ledger.begin_proposal(draft_tokens=1)

    with pytest.raises(BatchedMTPTransactionError, match="stale or foreign"):
        ledger.commit(first, emitted_counts={"a": 1, "b": 1})

    ledger.commit(second, emitted_counts={"a": 1, "b": 2})
    with pytest.raises(BatchedMTPTransactionError, match="no proposal"):
        ledger.commit(second, emitted_counts={"a": 1, "b": 2})


@pytest.mark.parametrize(
    "counts",
    [
        {"a": 1},
        {"a": 0, "b": 1},
        {"a": 3, "b": 1},
        {"a": True, "b": 1},
    ],
)
def test_commit_rejects_incomplete_or_impossible_counts(counts):
    ledger = BatchedMTPBookkeeper(("a", "b"))
    ticket = ledger.begin_proposal(draft_tokens=1)
    with pytest.raises(BatchedMTPTransactionError):
        ledger.commit(ticket, emitted_counts=counts)
    assert ledger.outstanding == ticket


def test_detach_advances_epoch_and_preserves_remaining_order():
    ledger = BatchedMTPBookkeeper(("a", "b", "c"))
    assert ledger.detach(("b",)) == 2
    assert ledger.lane_ids == ("a", "c")


def test_inputs_are_validated_without_model_or_array_dependencies():
    with pytest.raises(ValueError, match="unique"):
        plan_admission(
            [LaneAdmission("a"), LaneAdmission("a")],
            config=_config(),
            capabilities=_capabilities(),
            free_bytes=1000,
        )
    with pytest.raises(ValueError, match="positive integer"):
        BatchedMTPBookkeeper(("a",)).begin_proposal(draft_tokens=True)
    with pytest.raises(ValueError, match="positive integer"):
        BatchedMTPBookkeeper(("a",)).begin_proposal(draft_tokens="2")
    with pytest.raises(ValueError, match="policy flags"):
        BatchedMTPConfig(enabled=1)
    with pytest.raises(ValueError, match="free_bytes must be an integer"):
        plan_admission(
            [LaneAdmission("a"), LaneAdmission("b")],
            config=_config(),
            capabilities=_capabilities(),
            free_bytes=True,
        )
