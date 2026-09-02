# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from scripts.bench_qwen4_fused_gdn_verify import (
    checkpoint_fingerprint,
    checkpoint_norm_convention,
    completion_status,
    expected_verify_calls,
    greedy_accept_count,
    host_violation,
    is_model_process,
    next_round_tokens,
    parse_free_percent,
    parse_swap_used_mib,
    recenter_gains,
    rollback_like_engine,
    scripted_drafts,
)


def test_expected_verify_calls_prove_fused_execution_per_round():
    assert expected_verify_calls("stock", 36, 40) == (0, 0)
    assert expected_verify_calls("fused", 36, 40) == (1440, 0)
    with pytest.raises(ValueError, match="unknown mode"):
        expected_verify_calls("other", 36, 1)


def test_greedy_acceptance_counts_leading_confirmed_drafts_only():
    assert greedy_accept_count([7, 8, 9], [7, 8]) == 2
    assert greedy_accept_count([7, 3, 9], [7, 8]) == 1
    assert greedy_accept_count([1, 8, 9], [7, 8]) == 0


def test_next_round_tokens_follow_engine_commit_rules():
    # All drafts accepted: bonus is the last prediction, drafts are filler.
    assert next_round_tokens([7, 8, 9], [7, 8], filler=0) == (9, [0, 0], 2)
    # First draft rejected: bonus replaces it; later predictions become drafts.
    assert next_round_tokens([1, 8, 9], [7, 8], filler=0) == (1, [8, 9], 0)
    # Second draft rejected.
    assert next_round_tokens([7, 3, 9], [7, 8], filler=0) == (3, [9, 0], 1)


class _SnapshotCache:
    def __init__(self):
        self.rollback_state = [("a",), ("b",)]
        self.restored = None

    def restore_rollback(self, n_to_drop, verify_size):
        self.restored = (n_to_drop, verify_size)
        self.rollback_state = None


class _Leaf:
    def __init__(self):
        self.rollback_state = "stale"


class _TrimCache:
    """Composite attention cache: no ``restore_rollback``, one stale leaf."""

    def __init__(self):
        self.leaf = _Leaf()
        self.caches = [self.leaf]

    @property
    def rollback_state(self):
        return self.leaf.rollback_state


def test_rollback_like_engine_restores_snapshots_and_trims_the_rest():
    snapshot = _SnapshotCache()
    trim = _TrimCache()
    trimmed = []

    def fake_trim_all(caches, n):
        trimmed.append((list(caches), n))
        return True

    rollback_like_engine([snapshot, trim], 1, 3, fake_trim_all)
    assert snapshot.restored == (1, 3)
    assert trimmed == [([trim], 1)]

    # A fully accepted round clears stale restore points without trimming.
    snapshot = _SnapshotCache()
    trim = _TrimCache()
    rollback_like_engine([snapshot, trim], 0, 3, fake_trim_all)
    assert snapshot.restored is None
    assert snapshot.rollback_state is None
    assert trim.rollback_state is None


def test_rollback_like_engine_fails_closed_when_attention_refuses():
    with pytest.raises(RuntimeError, match="refused"):
        rollback_like_engine([_TrimCache()], 2, 3, lambda caches, n: False)


def test_scripted_drafts_cycle_through_every_acceptance_count():
    vocab = 100
    assert scripted_drafts([11, 12], 2, 0, vocab) == ([11, 12], 2)
    assert scripted_drafts([11, 12], 2, 1, vocab) == ([11, 13], 1)
    assert scripted_drafts([11, 12], 2, 2, vocab) == ([12, 12], 0)
    assert scripted_drafts([13, 14], 2, 3, vocab) == ([13, 14], 2)
    assert scripted_drafts([99], 1, 1, vocab) == ([0], 0)


def test_host_violation_fails_closed_on_unreadable_receipts_and_tenants():
    args = SimpleNamespace(min_free_percent=20, max_swap_growth_mib=2048.0)
    baseline = {"free_percent": 90.0, "swap_used_mib": 1000.0}

    def receipt(**overrides):
        base = {
            "scan_ok": True,
            "free_percent": 50.0,
            "swap_used_mib": 1500.0,
            "port_8282_listeners": 0,
            "model_processes": [],
            "large_processes": [],
        }
        base.update(overrides)
        return base

    assert host_violation(receipt(), baseline, args, "p") is None
    assert host_violation(receipt(scan_ok=False), baseline, args, "p") == (
        "p: host scans unreadable"
    )
    assert host_violation(receipt(free_percent=None), baseline, args, "p") == (
        "p: host memory receipt unreadable"
    )
    assert host_violation(receipt(swap_used_mib=None), baseline, args, "p") == (
        "p: host memory receipt unreadable"
    )
    assert host_violation(receipt(free_percent=10.0), baseline, args, "p") == (
        "p: free memory 10.0% below 20%"
    )
    assert host_violation(receipt(swap_used_mib=4000.0), baseline, args, "p") == (
        "p: swap grew 3000 MiB"
    )
    assert host_violation(receipt(port_8282_listeners=1), baseline, args, "p") == (
        'p: GPU tenant present {"port_8282_listeners": 1}'
    )
    assert host_violation(
        receipt(large_processes=["9 99999999 /x/python server.py"]), baseline, args, "p"
    ).startswith("p: GPU tenant present")
    assert host_violation(
        receipt(model_processes=["9 5 /v/bin/rapid-mlx serve"]), baseline, args, "p"
    ).startswith("p: GPU tenant present")


def test_model_process_filter_matches_console_scripts():
    assert is_model_process("7 5000 /v/bin/rapid-mlx serve /m", 1)
    assert is_model_process("8 5000 /v/bin/rmlx serve /m", 1)
    assert is_model_process("9 5000 /v/bin/vllm-mlx serve /m", 1)
    # The same console scripts launched through the interpreter.
    assert is_model_process("10 5000 /v/bin/python /v/bin/rapid-mlx serve /m", 1)
    assert is_model_process("11 5000 /v/bin/python3.12 /v/bin/rmlx serve", 1)
    assert is_model_process("12 5000 /v/bin/python /v/bin/vllm-mlx serve", 1)
    assert not is_model_process("13 5000 /v/bin/python /x/rapid-mlx-notes.py", 1)


def test_completion_status_reserves_passed_for_complete_runs():
    full = {"layer": {}, "rounds": {}, "e2e": {}}
    assert completion_status(full, True) == {
        "phases_executed": ["layer", "rounds", "e2e"],
        "complete": True,
        "partial_passed": True,
        "passed": True,
    }
    assert completion_status(full, False)["passed"] is False
    partial = completion_status({"layer": {}}, True)
    assert partial["complete"] is False
    assert partial["partial_passed"] is True
    assert partial["passed"] is False


def test_checkpoint_fingerprint_hashes_config_and_index(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    fingerprint = checkpoint_fingerprint(tmp_path)
    assert fingerprint["path"] == str(tmp_path)
    assert len(fingerprint["config.json"]) == 64
    assert "model.safetensors.index.json" not in fingerprint


def test_model_process_filter_matches_engines_not_helpers():
    own = 4242
    assert is_model_process("100 5000 /v/bin/python -m vllm_mlx.cli serve /m", own)
    assert is_model_process("101 5000 /v/bin/python -m mlx_lm.server --model x", own)
    assert is_model_process("102 5000 /v/bin/python scripts/bench_x.py", own)
    assert is_model_process(
        "103 5000 /Applications/LMStudio.app/Contents/MacOS/lmstudio", own
    )
    assert not is_model_process(
        "1724 2960 /A/Python -m http.server 8055 --directory /x/mlxuag-wiki", own
    )
    assert not is_model_process("1744 7000 /x/.venv/bin/python /x/mem0_mcp.py", own)
    assert not is_model_process(f"{own} 5000 /v/bin/python -m vllm_mlx.cli serve", own)
    assert not is_model_process("PID RSS COMMAND", own)
    # An agent or shell whose command line merely quotes a python command.
    assert not is_model_process(
        "16729 31200 node /x/codex exec -C /x 'run python -m mlx_lm.server'", own
    )
    assert not is_model_process(
        "28649 2464 /bin/zsh -c python scripts/bench_qwen4_fused_gdn_verify.py", own
    )


def test_norm_convention_detection_and_recentering():
    class Arr:
        def __init__(self, shape, value=2.0):
            self.shape = shape
            self.ndim = len(shape)
            self.value = value

        def __sub__(self, other):
            return Arr(self.shape, self.value - other)

    raw = {"language_model.model.layers.0.linear_attn.conv1d.weight": Arr((10, 1, 4))}
    converted = {
        "language_model.model.layers.0.linear_attn.conv1d.weight": Arr((10, 4, 1))
    }
    assert checkpoint_norm_convention(raw) == "raw"
    assert checkpoint_norm_convention(converted) == "converted"

    weights = {
        "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight": Arr((4,)),
        "language_model.model.layers.1.ple.norm_key.weight": Arr((4,)),
        "language_model.model.layers.3.self_attn.q_norm.weight": Arr((4,)),
        "pre_fc_norm_hidden.weight": Arr((4,)),
        "language_model.model.layers.0.linear_attn.norm.weight": Arr((4,)),
        "language_model.model.norm.weight": Arr((4,)),
    }
    assert recenter_gains(dict(weights), "raw") == 0
    touched = dict(weights)
    assert recenter_gains(touched, "converted") == 4
    assert touched["pre_fc_norm_hidden.weight"].value == 1.0
    assert touched["language_model.model.norm.weight"].value == 2.0
    assert touched["language_model.model.layers.0.linear_attn.norm.weight"].value == 2.0


def test_host_receipt_parsers():
    assert parse_free_percent("System-wide memory free percentage: 94%\n") == 94.0
    assert parse_free_percent("garbage") is None
    assert (
        parse_swap_used_mib(
            "vm.swapusage: total = 7168.00M  used = 6009.38M  free = 1158.62M"
        )
        == 6009.38
    )
    assert parse_swap_used_mib("") is None
