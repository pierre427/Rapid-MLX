# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PFlash endpoints-only detection.

When ``sink_tokens + tail_tokens`` meet or exceed the keep budget, PFlash keeps
only the leading sink and trailing tail and drops the entire middle span. With
the default ``always``-mode settings (sink 256 + tail 2048, min_keep 2048,
keep_ratio 0.20) this happens for every prompt shorter than ~11.5k tokens, yet
``compressed`` stays True and ``reason`` stays ``"compressed"``. These tests pin
the observability added for that regime — ``PFlashResult.endpoints_only`` /
``middle_tokens_kept`` and the matching ``compress_request_tokens`` metadata —
and assert the normal compression path stays unflagged.
"""

from vllm_mlx.pflash import (
    PFlashConfig,
    compress_request_tokens,
    compress_tokens,
)


class TestPFlashEndpointsOnly:
    def test_default_always_config_collapses_short_prompt(self):
        # 3k tokens under the default always-mode config: keep_budget == 2048 but
        # sink(256)+tail(2048) == 2304 > budget, so the whole middle is dropped
        # and kept_tokens (2304) even exceeds the nominal budget.
        result = compress_tokens(list(range(3000)), PFlashConfig(mode="always"))
        assert result.compressed is True
        assert result.reason == "compressed"
        assert result.middle_tokens_kept == 0
        assert result.endpoints_only is True
        assert result.kept_tokens == 2304

    def test_generous_budget_keeps_middle_and_is_not_flagged(self):
        # 20k tokens: 0.2 * 20k == 4000 budget leaves room for middle blocks.
        result = compress_tokens(
            list(range(20000)), PFlashConfig(mode="always", keep_ratio=0.2)
        )
        assert result.compressed is True
        assert result.middle_tokens_kept > 0
        assert result.endpoints_only is False

    def test_metadata_surfaces_endpoints_only(self):
        _, metadata = compress_request_tokens(
            list(range(3000)), PFlashConfig(mode="always")
        )
        assert metadata["endpoints_only"] is True
        assert metadata["middle_tokens_kept"] == 0

    def test_metadata_middle_tokens_kept_positive_on_normal_compression(self):
        _, metadata = compress_request_tokens(
            list(range(20000)), PFlashConfig(mode="always", keep_ratio=0.2)
        )
        assert metadata["endpoints_only"] is False
        assert metadata["middle_tokens_kept"] > 0

    def test_uncompressed_result_is_not_endpoints_only(self):
        result = compress_tokens(list(range(20000)), PFlashConfig(mode="off"))
        assert result.compressed is False
        assert result.endpoints_only is False

    def test_small_sink_tail_leaves_middle_on_short_prompt(self):
        cfg = PFlashConfig(
            mode="always",
            keep_ratio=0.5,
            min_keep_tokens=64,
            sink_tokens=8,
            tail_tokens=8,
            block_size=8,
        )
        result = compress_tokens(list(range(2000)), cfg)
        assert result.compressed is True
        assert result.middle_tokens_kept > 0
        assert result.endpoints_only is False
