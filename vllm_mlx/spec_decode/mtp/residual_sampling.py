# SPDX-License-Identifier: Apache-2.0
"""Exact transformed-distribution hooks for continuous self-MTP.

Adapted from immutable source commit ``5d51d766``.  Module import is MLX-free;
constructing the default ops adapter performs the lazy ``mlx.core`` import.
The transform mirrors Rapid/mlx-lm sampler order exactly: native-dtype
normalization, top-p, min-p, top-k, temperature scaling, then float32
renormalization.  XTC is intentionally unsupported because its stochastic mask
is not shared between target and draft distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .continuous_engine import ContinuousSelfMTPUnsupportedError, SelfMTPSampling


class ResidualArrayOps(Protocol):
    def native_logprobs(self, logits: Any) -> Any: ...

    def top_p(self, logprobs: Any, value: float) -> Any: ...

    def min_p(self, logprobs: Any, value: float, min_tokens: int) -> Any: ...

    def top_k(self, logprobs: Any, value: int) -> Any: ...

    def scaled_float32_logprobs(self, logprobs: Any, temperature: float) -> Any: ...

    def categorical(self, logprobs: Any) -> int: ...

    def gather_rows(self, logprobs: Any, tokens: list[int]) -> Any: ...

    def stack(self, values: list[Any]) -> Any: ...

    def exp(self, value: Any) -> Any: ...

    def minimum_scalar(self, value: Any, scalar: float) -> Any: ...

    def maximum_scalar(self, value: Any, scalar: float) -> Any: ...

    def subtract(self, left: Any, right: Any) -> Any: ...

    def sum_float(self, value: Any) -> float: ...

    def normalize_probabilities_to_logprobs(self, probabilities: Any) -> Any: ...

    def uniforms(self, count: int) -> list[float]: ...

    def tolist(self, value: Any) -> list[float]: ...


class _MLXResidualOps:
    """Lazy production ops matching mlx-lm ``sample_utils`` semantics."""

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx

    def native_logprobs(self, logits: Any) -> Any:
        return logits - self.mx.logsumexp(logits, axis=-1, keepdims=True)

    def top_p(self, logprobs: Any, value: float) -> Any:
        probs = self.mx.exp(logprobs)
        indices = self.mx.argsort(logprobs, axis=-1)
        sorted_probs = self.mx.take_along_axis(probs, indices, axis=-1)
        cumulative = self.mx.cumsum(sorted_probs, axis=-1)
        inverse = self.mx.put_along_axis(
            self.mx.zeros_like(indices),
            indices,
            self.mx.arange(indices.shape[-1], dtype=indices.dtype),
            axis=-1,
        )
        cumulative = self.mx.take_along_axis(cumulative, inverse, axis=-1)
        return self.mx.where(cumulative > 1 - value, logprobs, -self.mx.inf)

    def min_p(self, logprobs: Any, value: float, min_tokens: int) -> Any:
        import math

        threshold = self.mx.max(logprobs, axis=-1, keepdims=True) + math.log(value)
        remove = logprobs < threshold
        if min_tokens > 1:
            indices = self.mx.argpartition(logprobs, kth=-min_tokens, axis=-1)[
                ..., -min_tokens:
            ]
            remove = self.mx.put_along_axis(remove, indices, False, axis=-1)
        return self.mx.where(remove, -self.mx.inf, logprobs)

    def top_k(self, logprobs: Any, value: int) -> Any:
        vocab = int(logprobs.shape[-1])
        if not 0 < value < vocab:
            raise ValueError(f"top_k must be in (0, {vocab}); got {value}")
        indices = self.mx.argpartition(-logprobs, kth=value - 1, axis=-1)[..., value:]
        return self.mx.put_along_axis(
            logprobs,
            indices,
            self.mx.array(-self.mx.inf, logprobs.dtype),
            axis=-1,
        )

    def scaled_float32_logprobs(self, logprobs: Any, temperature: float) -> Any:
        scaled = (logprobs * (1 / temperature)).astype(self.mx.float32)
        return scaled - self.mx.logsumexp(scaled, axis=-1, keepdims=True)

    def categorical(self, logprobs: Any) -> int:
        return int(self.mx.random.categorical(logprobs).item())

    def gather_rows(self, logprobs: Any, tokens: list[int]) -> Any:
        token_array = self.mx.array(tokens)[:, None]
        return self.mx.take_along_axis(logprobs, token_array, axis=-1)[:, 0]

    def stack(self, values: list[Any]) -> Any:
        return self.mx.stack(values)

    def exp(self, value: Any) -> Any:
        return self.mx.exp(value)

    def minimum_scalar(self, value: Any, scalar: float) -> Any:
        return self.mx.minimum(value, scalar)

    def maximum_scalar(self, value: Any, scalar: float) -> Any:
        return self.mx.maximum(value, scalar)

    @staticmethod
    def subtract(left: Any, right: Any) -> Any:
        return left - right

    def sum_float(self, value: Any) -> float:
        total = self.mx.sum(value)
        self.mx.eval(total)
        return float(total.item())

    def normalize_probabilities_to_logprobs(self, probabilities: Any) -> Any:
        total = self.mx.sum(probabilities)
        return self.mx.log(probabilities / total)

    def uniforms(self, count: int) -> list[float]:
        values = self.mx.random.uniform(shape=(count,))
        self.mx.eval(values)
        return [float(value) for value in values.tolist()]

    @staticmethod
    def tolist(value: Any) -> list[float]:
        return [float(item) for item in value.tolist()]


@dataclass(frozen=True)
class TransformedSamplingProfile:
    temperature: float
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    xtc_probability: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("temperature must be numeric")
        if self.temperature <= 0:
            raise ValueError("transformed sampling requires temperature > 0")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in [0, 1]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("top_k must be an integer")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if (
            isinstance(self.min_tokens_to_keep, bool)
            or not isinstance(self.min_tokens_to_keep, int)
            or self.min_tokens_to_keep < 1
        ):
            raise ValueError("min_tokens_to_keep must be a positive integer")
        if not 0 <= self.xtc_probability <= 1:
            raise ValueError("xtc_probability must be in [0, 1]")
        if self.xtc_probability > 0:
            raise ContinuousSelfMTPUnsupportedError(
                "XTC has no exact shared-mask residual verifier"
            )


class TransformedResidualSamplingHooks:
    """Shared draft/target transform and exact residual verifier."""

    def __init__(
        self,
        profile: TransformedSamplingProfile,
        *,
        array_ops: ResidualArrayOps | None = None,
    ) -> None:
        self.profile = profile
        self.ops = array_ops or _MLXResidualOps()

    def validate_sampling(self, sampling: SelfMTPSampling) -> None:
        if sampling.uses_xtc:
            raise ContinuousSelfMTPUnsupportedError(
                "XTC has no exact shared-mask residual verifier"
            )
        if sampling.temperature != self.profile.temperature:
            raise ContinuousSelfMTPUnsupportedError(
                "lane temperature does not match transformed hook profile"
            )

    def logprobs(self, logits: Any, temperature: float) -> Any:
        if temperature != self.profile.temperature:
            raise ContinuousSelfMTPUnsupportedError(
                "runtime temperature does not match transformed hook profile"
            )
        values = self.ops.native_logprobs(logits)
        profile = self.profile
        # Exact Rapid/mlx-lm make_sampler order.
        if 0 < profile.top_p < 1:
            values = self.ops.top_p(values, profile.top_p)
        if profile.min_p != 0:
            values = self.ops.min_p(values, profile.min_p, profile.min_tokens_to_keep)
        if profile.top_k > 0:
            values = self.ops.top_k(values, profile.top_k)
        return self.ops.scaled_float32_logprobs(values, temperature)

    def sample(self, logprobs: Any, temperature: float) -> int:
        if temperature != self.profile.temperature:
            raise ContinuousSelfMTPUnsupportedError(
                "runtime temperature does not match transformed hook profile"
            )
        return self.ops.categorical(logprobs)

    def _residual_sample(self, target: Any, draft: Any) -> int:
        residual = self.ops.maximum_scalar(
            self.ops.subtract(self.ops.exp(target), self.ops.exp(draft)), 0.0
        )
        if self.ops.sum_float(residual) <= 0:
            return self.ops.categorical(target)
        return self.ops.categorical(
            self.ops.normalize_probabilities_to_logprobs(residual)
        )

    def verify(
        self,
        target_logprobs: Any,
        draft_logprobs: list[Any],
        draft_tokens: list[int],
        temperature: float,
    ) -> tuple[int, int]:
        if temperature != self.profile.temperature:
            raise ContinuousSelfMTPUnsupportedError(
                "runtime temperature does not match transformed hook profile"
            )
        k = len(draft_tokens)
        if len(draft_logprobs) != k:
            raise ValueError("draft logprob/token lengths disagree")
        if int(target_logprobs.shape[0]) != k + 1:
            raise ValueError("target verification requires K draft rows plus bonus")
        if k == 0:
            return 0, self.ops.categorical(target_logprobs[0])

        target_at = self.ops.gather_rows(target_logprobs[:k], draft_tokens)
        draft_at = self.ops.gather_rows(
            self.ops.stack(list(draft_logprobs)), draft_tokens
        )
        ratios = self.ops.exp(
            self.ops.minimum_scalar(self.ops.subtract(target_at, draft_at), 0.0)
        )
        ratio_values = self.ops.tolist(ratios)
        uniforms = self.ops.uniforms(k)
        accepted = 0
        while (
            accepted < k
            and ratio_values[accepted] > 0
            and uniforms[accepted] <= ratio_values[accepted]
        ):
            accepted += 1
        if accepted < k:
            bonus = self._residual_sample(
                target_logprobs[accepted], draft_logprobs[accepted]
            )
        else:
            bonus = self.ops.categorical(target_logprobs[accepted])
        return accepted, bonus


__all__ = [
    "ResidualArrayOps",
    "TransformedResidualSamplingHooks",
    "TransformedSamplingProfile",
]
