# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLX-worker to HTTP-thread logprob transport."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx


def test_materialize_logprobs_for_transport_returns_host_snapshot():
    from vllm_mlx.scheduler import _materialize_logprobs_for_transport

    device_row = mx.array([-3.0, -2.0, -1.0], dtype=mx.bfloat16)
    host_row = _materialize_logprobs_for_transport(device_row)

    assert isinstance(host_row, np.ndarray)
    assert host_row.dtype == np.float32
    np.testing.assert_allclose(host_row, [-3.0, -2.0, -1.0])


def test_logprob_extractor_accepts_worker_host_snapshot():
    from vllm_mlx.service.helpers import _extract_token_logprob

    class Tokenizer:
        @staticmethod
        def decode(token_ids):
            return f"t{token_ids[0]}"

    result = _extract_token_logprob(
        np.array([-4.0, -0.5, -2.0], dtype=np.float32),
        token_id=1,
        tokenizer=Tokenizer(),
        top_k=2,
    )

    assert result.token == "t1"
    assert result.logprob == pytest.approx(-0.5)
    assert [entry.token for entry in result.top_logprobs] == ["t1", "t2"]
