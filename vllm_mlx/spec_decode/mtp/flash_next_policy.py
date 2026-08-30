# SPDX-License-Identifier: Apache-2.0
"""Request-boundary decode policy for Qwen3.8 Flash-Next.

The selector is intentionally model-free and side-effect free.  It chooses a
route from immutable request/runtime facts; the scheduler remains responsible
for installing and executing the selected mechanism.  PLE is part of the
Flash-Next target computation (resident or NVMe-backed), not a proposal source,
so every route carries its PLE identity separately from the decode mechanism.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

FLASH_NEXT_POLICY_VERSION = "flash-next-b1-v1"


class FlashNextContextBucket(str, enum.Enum):
    LE_16K = "le_16k"
    GT_16K_LE_32K = "gt_16k_le_32k"
    GT_32K = "gt_32k"


class FlashNextDecodeRoute(str, enum.Enum):
    LINEAR_SELF_MTP = "linear_self_mtp"
    PLAIN_CONTINUOUS_BATCHING = "plain_continuous_batching"
    ADAPTIVE_PLD = "adaptive_pld"
    PLE_ENABLED_PLAIN = "ple_enabled_plain"
    PLAIN_DECODE = "plain_decode"


class FlashNextRouteReason(str, enum.Enum):
    QUEUE_WIDTH = "real_queue_width_gt_one"
    SHORT_CONTEXT = "short_context"
    GUARDED_CONTEXT = "guarded_context_qualified"
    SELF_MTP_UNSUPPORTED = "self_mtp_unsupported"
    MEMORY_GUARD = "memory_guard_failed"
    PERFORMANCE_MISSING = "recent_performance_missing"
    PERFORMANCE_STALE = "recent_performance_stale"
    PERFORMANCE_REGRESSION = "recent_performance_regression"
    APC_TARGET_ONLY_PRESERVED = "substantial_target_only_apc_preserved"
    LONG_CONTEXT = "self_mtp_context_limit"
    PLD_HYBRID_UNQUALIFIED = "adaptive_pld_hybrid_state_unqualified"
    PLD_EXECUTOR_UNAVAILABLE = "adaptive_pld_request_executor_unavailable"


@dataclass(frozen=True)
class FlashNextRoutePolicyConfig:
    """Frozen product boundaries derived from the corrected B=1 sweep."""

    preferred_self_mtp_max_tokens: int = 16_384
    guarded_self_mtp_max_tokens: int = 32_768
    adaptive_pld_min_tokens: int = 32_768
    min_recent_speedup_ratio: float = 1.0
    min_recent_samples: int = 4
    max_recent_sample_age_seconds: float = 900.0
    min_apc_preserve_tokens: int = 64

    def __post_init__(self) -> None:
        if self.preferred_self_mtp_max_tokens < 1:
            raise ValueError("preferred_self_mtp_max_tokens must be positive")
        if self.guarded_self_mtp_max_tokens < self.preferred_self_mtp_max_tokens:
            raise ValueError("guarded self-MTP limit must not be below preferred limit")
        if self.adaptive_pld_min_tokens < self.preferred_self_mtp_max_tokens:
            raise ValueError("adaptive PLD limit must not be below preferred limit")
        if not math.isfinite(self.min_recent_speedup_ratio):
            raise ValueError("min_recent_speedup_ratio must be finite")
        if self.min_recent_speedup_ratio <= 0:
            raise ValueError("min_recent_speedup_ratio must be positive")
        if self.min_recent_samples < 1:
            raise ValueError("min_recent_samples must be positive")
        if self.min_apc_preserve_tokens < 1:
            raise ValueError("min_apc_preserve_tokens must be positive")
        if not math.isfinite(self.max_recent_sample_age_seconds):
            raise ValueError("max_recent_sample_age_seconds must be finite")
        if self.max_recent_sample_age_seconds <= 0:
            raise ValueError("max_recent_sample_age_seconds must be positive")


@dataclass(frozen=True)
class FlashNextRouteCapabilities:
    self_mtp: bool
    ple_enabled: bool
    ple_storage: str = "none"
    adaptive_pld: bool = False
    hybrid_recurrent_state_qualified: bool = False
    request_scoped_pld_executor: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ple_storage, str) or not self.ple_storage:
            raise ValueError("ple_storage must be a non-empty string")
        if not self.ple_enabled and self.ple_storage != "none":
            raise ValueError("disabled PLE must use ple_storage='none'")


@dataclass(frozen=True)
class RecentSelfMTPPerformance:
    """Bounded request-level evidence used only in the guarded context band."""

    speedup_ratio: float
    samples: int
    age_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.speedup_ratio) or self.speedup_ratio <= 0:
            raise ValueError("speedup_ratio must be finite and positive")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if not math.isfinite(self.age_seconds) or self.age_seconds < 0:
            raise ValueError("age_seconds must be finite and non-negative")


@dataclass(frozen=True)
class FlashNextRouteInputs:
    effective_context_tokens: int
    projected_context_tokens: int
    real_queue_width: int
    free_bytes: int
    memory_reserve_bytes: int
    self_mtp_incremental_bytes: int = 0
    apc_cached_tokens: int = 0
    exact_joint_mtp_state: bool = False
    recent_performance: RecentSelfMTPPerformance | None = None

    def __post_init__(self) -> None:
        for name in (
            "effective_context_tokens",
            "projected_context_tokens",
            "real_queue_width",
            "free_bytes",
            "memory_reserve_bytes",
            "self_mtp_incremental_bytes",
            "apc_cached_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.real_queue_width < 1:
            raise ValueError("real_queue_width must be positive")
        if self.projected_context_tokens < self.effective_context_tokens:
            raise ValueError("projected context must include effective context")
        if not isinstance(self.exact_joint_mtp_state, bool):
            raise ValueError("exact_joint_mtp_state must be a boolean")


@dataclass(frozen=True)
class FlashNextRouteDecision:
    route: FlashNextDecodeRoute
    reason: FlashNextRouteReason
    effective_context_tokens: int
    projected_context_tokens: int
    real_queue_width: int
    ple_enabled: bool
    ple_storage: str
    self_mtp_eligible: bool
    adaptive_pld_eligible: bool
    policy_version: str = FLASH_NEXT_POLICY_VERSION
    context_bucket: FlashNextContextBucket = FlashNextContextBucket.LE_16K

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "route": self.route.value,
            "reason": self.reason.value,
            "effective_context_tokens": self.effective_context_tokens,
            "projected_context_tokens": self.projected_context_tokens,
            "real_queue_width": self.real_queue_width,
            "ple_enabled": self.ple_enabled,
            "ple_storage": self.ple_storage,
            "self_mtp_eligible": self.self_mtp_eligible,
            "adaptive_pld_eligible": self.adaptive_pld_eligible,
            "policy_version": self.policy_version,
            "context_bucket": self.context_bucket.value,
        }


def select_flash_next_route(
    inputs: FlashNextRouteInputs,
    capabilities: FlashNextRouteCapabilities,
    *,
    config: FlashNextRoutePolicyConfig = FlashNextRoutePolicyConfig(),
) -> FlashNextRouteDecision:
    """Choose one route at the request boundary without mutating runtime state."""

    pld_state_qualified = (
        capabilities.adaptive_pld
        and capabilities.hybrid_recurrent_state_qualified
    )
    projected = inputs.projected_context_tokens
    pld_eligible = (
        pld_state_qualified
        and capabilities.request_scoped_pld_executor
        and inputs.real_queue_width == 1
        and projected >= config.adaptive_pld_min_tokens
    )
    if projected <= config.preferred_self_mtp_max_tokens:
        context_bucket = FlashNextContextBucket.LE_16K
    elif projected <= config.guarded_self_mtp_max_tokens:
        context_bucket = FlashNextContextBucket.GT_16K_LE_32K
    else:
        context_bucket = FlashNextContextBucket.GT_32K

    def decision(
        route: FlashNextDecodeRoute,
        reason: FlashNextRouteReason,
        *,
        self_mtp_eligible: bool = False,
    ) -> FlashNextRouteDecision:
        return FlashNextRouteDecision(
            route=route,
            reason=reason,
            effective_context_tokens=inputs.effective_context_tokens,
            projected_context_tokens=inputs.projected_context_tokens,
            real_queue_width=inputs.real_queue_width,
            ple_enabled=capabilities.ple_enabled,
            ple_storage=capabilities.ple_storage,
            self_mtp_eligible=self_mtp_eligible,
            adaptive_pld_eligible=pld_eligible,
            context_bucket=context_bucket,
        )

    def long_context_fallback(reason: FlashNextRouteReason) -> FlashNextRouteDecision:
        if pld_eligible:
            return decision(FlashNextDecodeRoute.ADAPTIVE_PLD, reason)
        if capabilities.ple_enabled:
            return decision(FlashNextDecodeRoute.PLE_ENABLED_PLAIN, reason)
        return decision(FlashNextDecodeRoute.PLAIN_DECODE, reason)

    if inputs.real_queue_width > 1:
        return decision(
            FlashNextDecodeRoute.PLAIN_CONTINUOUS_BATCHING,
            FlashNextRouteReason.QUEUE_WIDTH,
        )

    # A target-only APC hit is already valuable work.  Throwing away a
    # substantial prefix merely to regain self-MTP loses the real-session
    # objective.  An exact joint target+MTP sidecar is the sole exception.
    if (
        inputs.apc_cached_tokens >= config.min_apc_preserve_tokens
        and not inputs.exact_joint_mtp_state
    ):
        return long_context_fallback(FlashNextRouteReason.APC_TARGET_ONLY_PRESERVED)

    if not capabilities.self_mtp:
        return long_context_fallback(FlashNextRouteReason.SELF_MTP_UNSUPPORTED)

    # At and above the selected 32K product floor, the adaptive route owns the
    # request. A lookup miss remains inside the generator and falls through to
    # its self-MTP tail; recent self-MTP telemetry must not steal a retrievable
    # request back from the default PLD latch.
    if pld_eligible:
        return decision(
            FlashNextDecodeRoute.ADAPTIVE_PLD,
            FlashNextRouteReason.LONG_CONTEXT,
        )

    # Routing is immutable for this request.  Gate on the largest context the
    # request can reach so a short prompt cannot cross 32K while pinned to the
    # self-MTP data plane.  A later safe-boundary handoff can relax this.
    context = inputs.projected_context_tokens
    if context <= config.preferred_self_mtp_max_tokens:
        return decision(
            FlashNextDecodeRoute.LINEAR_SELF_MTP,
            FlashNextRouteReason.SHORT_CONTEXT,
            self_mtp_eligible=True,
        )

    if context > config.guarded_self_mtp_max_tokens:
        if not capabilities.adaptive_pld:
            reason = FlashNextRouteReason.LONG_CONTEXT
        elif not pld_state_qualified:
            reason = FlashNextRouteReason.PLD_HYBRID_UNQUALIFIED
        else:
            reason = FlashNextRouteReason.PLD_EXECUTOR_UNAVAILABLE
        return long_context_fallback(reason)

    required = inputs.memory_reserve_bytes + inputs.self_mtp_incremental_bytes
    if inputs.free_bytes < required:
        return long_context_fallback(FlashNextRouteReason.MEMORY_GUARD)

    recent = inputs.recent_performance
    if recent is None or recent.samples < config.min_recent_samples:
        return long_context_fallback(FlashNextRouteReason.PERFORMANCE_MISSING)
    if recent.age_seconds > config.max_recent_sample_age_seconds:
        return long_context_fallback(FlashNextRouteReason.PERFORMANCE_STALE)
    if recent.speedup_ratio < config.min_recent_speedup_ratio:
        return long_context_fallback(FlashNextRouteReason.PERFORMANCE_REGRESSION)
    return decision(
        FlashNextDecodeRoute.LINEAR_SELF_MTP,
        FlashNextRouteReason.GUARDED_CONTEXT,
        self_mtp_eligible=True,
    )


__all__ = [
    "FlashNextDecodeRoute",
    "FlashNextContextBucket",
    "FlashNextRouteCapabilities",
    "FlashNextRouteDecision",
    "FlashNextRouteInputs",
    "FlashNextRoutePolicyConfig",
    "FlashNextRouteReason",
    "RecentSelfMTPPerformance",
    "select_flash_next_route",
    "FLASH_NEXT_POLICY_VERSION",
]
