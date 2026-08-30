from __future__ import annotations

import pytest

from vllm_mlx.spec_decode.mtp.flash_next_policy import (
    FLASH_NEXT_POLICY_VERSION,
    FlashNextContextBucket,
    FlashNextDecodeRoute,
    FlashNextRouteCapabilities,
    FlashNextRouteInputs,
    FlashNextRouteReason,
    RecentSelfMTPPerformance,
    select_flash_next_route,
)

GIB = 1024**3


def _capabilities(**changes):
    values = {
        "self_mtp": True,
        "ple_enabled": True,
        "ple_storage": "nvme",
    }
    values.update(changes)
    return FlashNextRouteCapabilities(**values)


def _inputs(projected, **changes):
    values = {
        "effective_context_tokens": projected,
        "projected_context_tokens": projected,
        "real_queue_width": 1,
        "free_bytes": 40 * GIB,
        "memory_reserve_bytes": 20 * GIB,
        "self_mtp_incremental_bytes": 2 * GIB,
    }
    values.update(changes)
    return FlashNextRouteInputs(**values)


def _qualified_recent():
    return RecentSelfMTPPerformance(speedup_ratio=1.05, samples=4, age_seconds=1)


@pytest.mark.parametrize(
    ("projected", "recent", "route", "reason", "bucket"),
    [
        (
            16_384,
            None,
            FlashNextDecodeRoute.LINEAR_SELF_MTP,
            FlashNextRouteReason.SHORT_CONTEXT,
            FlashNextContextBucket.LE_16K,
        ),
        (
            16_385,
            None,
            FlashNextDecodeRoute.PLE_ENABLED_PLAIN,
            FlashNextRouteReason.PERFORMANCE_MISSING,
            FlashNextContextBucket.GT_16K_LE_32K,
        ),
        (
            16_385,
            _qualified_recent(),
            FlashNextDecodeRoute.LINEAR_SELF_MTP,
            FlashNextRouteReason.GUARDED_CONTEXT,
            FlashNextContextBucket.GT_16K_LE_32K,
        ),
        (
            32_768,
            _qualified_recent(),
            FlashNextDecodeRoute.LINEAR_SELF_MTP,
            FlashNextRouteReason.GUARDED_CONTEXT,
            FlashNextContextBucket.GT_16K_LE_32K,
        ),
        (
            32_769,
            _qualified_recent(),
            FlashNextDecodeRoute.PLE_ENABLED_PLAIN,
            FlashNextRouteReason.LONG_CONTEXT,
            FlashNextContextBucket.GT_32K,
        ),
    ],
)
def test_frozen_context_boundaries(projected, recent, route, reason, bucket):
    decision = select_flash_next_route(
        _inputs(projected, recent_performance=recent), _capabilities()
    )

    assert decision.route is route
    assert decision.reason is reason
    assert decision.context_bucket is bucket
    assert decision.policy_version == FLASH_NEXT_POLICY_VERSION


def test_guarded_band_requires_memory_and_recent_counterfactual_evidence():
    memory = select_flash_next_route(
        _inputs(24_000, free_bytes=21 * GIB, recent_performance=_qualified_recent()),
        _capabilities(),
    )
    regressed = select_flash_next_route(
        _inputs(
            24_000,
            recent_performance=RecentSelfMTPPerformance(
                speedup_ratio=0.99, samples=20, age_seconds=2
            ),
        ),
        _capabilities(),
    )
    stale = select_flash_next_route(
        _inputs(
            24_000,
            recent_performance=RecentSelfMTPPerformance(
                speedup_ratio=1.2, samples=20, age_seconds=901
            ),
        ),
        _capabilities(),
    )

    assert memory.reason is FlashNextRouteReason.MEMORY_GUARD
    assert regressed.reason is FlashNextRouteReason.PERFORMANCE_REGRESSION
    assert stale.reason is FlashNextRouteReason.PERFORMANCE_STALE
    assert {memory.route, regressed.route, stale.route} == {
        FlashNextDecodeRoute.PLE_ENABLED_PLAIN
    }


def test_real_queue_width_is_not_speculative_lane_width():
    multi_user = select_flash_next_route(
        _inputs(8_000, real_queue_width=2), _capabilities()
    )
    one_user = select_flash_next_route(
        _inputs(8_000, real_queue_width=1), _capabilities()
    )

    assert multi_user.route is FlashNextDecodeRoute.PLAIN_CONTINUOUS_BATCHING
    assert multi_user.reason is FlashNextRouteReason.QUEUE_WIDTH
    assert multi_user.real_queue_width == 2
    assert one_user.route is FlashNextDecodeRoute.LINEAR_SELF_MTP


def test_projected_total_context_counts_restored_prefix_and_uncached_suffix():
    # 30K restored APC prefix + 3K uncached suffix is already beyond 32K.
    # The selector sees logical context; incremental memory remains the small
    # self-MTP charge and is not inflated by the restored prefix.
    decision = select_flash_next_route(
        _inputs(
            33 * 1024,
            effective_context_tokens=33 * 1024,
            self_mtp_incremental_bytes=2 * GIB,
            apc_cached_tokens=30 * 1024,
            exact_joint_mtp_state=True,
        ),
        _capabilities(),
    )

    assert decision.route is FlashNextDecodeRoute.PLE_ENABLED_PLAIN
    assert decision.reason is FlashNextRouteReason.LONG_CONTEXT
    assert decision.effective_context_tokens == 33 * 1024


@pytest.mark.parametrize(
    ("cached", "joint", "expected_reason"),
    [
        (63, False, FlashNextRouteReason.SHORT_CONTEXT),
        (64, False, FlashNextRouteReason.APC_TARGET_ONLY_PRESERVED),
        (30_000, False, FlashNextRouteReason.APC_TARGET_ONLY_PRESERVED),
        (30_000, True, FlashNextRouteReason.GUARDED_CONTEXT),
    ],
)
def test_apc_target_only_and_exact_joint_sidecar_matrix(cached, joint, expected_reason):
    projected = 8_000 if cached < 30_000 else 30_000
    decision = select_flash_next_route(
        _inputs(
            projected,
            apc_cached_tokens=cached,
            exact_joint_mtp_state=joint,
            recent_performance=_qualified_recent(),
        ),
        _capabilities(),
    )

    assert decision.reason is expected_reason
    if expected_reason is FlashNextRouteReason.APC_TARGET_ONLY_PRESERVED:
        assert decision.route is FlashNextDecodeRoute.PLE_ENABLED_PLAIN
    else:
        assert decision.route is FlashNextDecodeRoute.LINEAR_SELF_MTP


def test_ple_identity_is_orthogonal_and_pld_needs_transactional_hybrid_accept():
    no_ple = select_flash_next_route(
        _inputs(40_000), _capabilities(ple_enabled=False, ple_storage="none")
    )
    unqualified_pld = select_flash_next_route(
        _inputs(40_000),
        _capabilities(adaptive_pld=True, hybrid_recurrent_state_qualified=False),
    )
    qualified_pld = select_flash_next_route(
        _inputs(40_000),
        _capabilities(
            adaptive_pld=True,
            hybrid_recurrent_state_qualified=True,
            request_scoped_pld_executor=True,
        ),
    )
    attested_without_executor = select_flash_next_route(
        _inputs(40_000),
        _capabilities(
            adaptive_pld=True,
            hybrid_recurrent_state_qualified=True,
            request_scoped_pld_executor=False,
        ),
    )

    assert no_ple.route is FlashNextDecodeRoute.PLAIN_DECODE
    assert no_ple.ple_enabled is False
    assert unqualified_pld.route is FlashNextDecodeRoute.PLE_ENABLED_PLAIN
    assert unqualified_pld.reason is FlashNextRouteReason.PLD_HYBRID_UNQUALIFIED
    assert attested_without_executor.route is FlashNextDecodeRoute.PLE_ENABLED_PLAIN
    assert (
        attested_without_executor.reason
        is FlashNextRouteReason.PLD_EXECUTOR_UNAVAILABLE
    )
    assert qualified_pld.route is FlashNextDecodeRoute.ADAPTIVE_PLD
    assert qualified_pld.ple_enabled is True
    assert qualified_pld.ple_storage == "nvme"


def test_decision_payload_has_fixed_observability_keys():
    payload = select_flash_next_route(_inputs(1_000), _capabilities()).to_dict()
    assert set(payload) == {
        "route",
        "reason",
        "effective_context_tokens",
        "projected_context_tokens",
        "real_queue_width",
        "ple_enabled",
        "ple_storage",
        "self_mtp_eligible",
        "adaptive_pld_eligible",
        "policy_version",
        "context_bucket",
    }
