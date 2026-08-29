"""CPU-only source guards for the opt-in QSA Metal route."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_qsa_custom_kernel_is_disabled_during_training():
    source = (ROOT / "vllm_mlx/models/qwen4_exp.py").read_text(encoding="utf-8")
    route = source[source.index("if (\n            isinstance(selected") :]
    route = route[: route.index("block_sparse_attention(")]

    assert "and not self.training" in route


def test_qsa_variable_lengths_are_runtime_uniforms():
    source = (ROOT / "vllm_mlx/kernels/qsa_block_sparse.py").read_text(encoding="utf-8")

    assert '"dims",' in source
    assert "mx.array([query_length, key_length], dtype=mx.int32)" in source
    assert '("QUERY_LENGTH", query_length)' not in source
    assert '("KEY_LENGTH", key_length)' not in source
