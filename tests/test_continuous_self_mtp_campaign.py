from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "continuous_self_mtp_campaign.py"
MANIFEST = ROOT / "bench" / "continuous_self_mtp_campaign.json"
QUALITY = ROOT / "bench" / "continuous_self_mtp_quality.json"

spec = importlib.util.spec_from_file_location("continuous_self_mtp_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(campaign)


def manifest():
    return campaign.load_json(MANIFEST)


def test_quality_bank_has_ten_unique_rubric_bearing_prompts():
    bank = campaign.load_json(QUALITY)
    prompts = bank["prompts"]
    assert len(prompts) == 10
    assert len({prompt["prompt_id"] for prompt in prompts}) == 10
    assert all(prompt["rubric"] for prompt in prompts)
    assert prompts[-1]["context_target_tokens"] == 16384


def test_product_plan_is_counterbalanced_and_matched():
    plan = campaign.campaign_plan(manifest(), "product", 8464)
    dense = [phase for phase in plan["phases"] if phase["model"] == "qwen38_27b"]
    assert [(phase["arm"], phase["order"]) for phase in dense] == [
        ("rapid_current", "ascending"),
        ("full_stack", "ascending"),
        ("full_stack", "descending"),
        ("rapid_current", "descending"),
    ]
    assert [cell["service_concurrency"] for cell in dense[0]["cells"]] == [
        16,
        16,
        8,
        4,
        2,
    ]


def test_flash_single_lane_rows_are_not_called_continuous():
    plan = campaign.campaign_plan(manifest(), "product", 8464)
    full = next(
        phase
        for phase in plan["phases"]
        if phase["model"] == "qwen38_flash_next"
        and phase["arm"] == "full_stack"
        and phase["order"] == "ascending"
    )
    assertions = {
        cell["context_tokens"]: cell["continuous_route_expected"]
        for cell in full["cells"]
    }
    assert assertions == {
        1024: True,
        4096: True,
        16384: True,
        32768: False,
        65536: False,
    }


def test_launch_commands_select_source_and_disable_or_enable_stack():
    data = manifest()
    baseline = campaign.shell_launch(data, "rapid_current", "qwen38_27b", 8464)
    candidate = campaign.shell_launch(data, "full_stack", "qwen38_27b", 8464)
    assert data["source_components"]["rapid_current"]["checkout"] in baseline
    assert "--no-spec-decode" in baseline
    assert "--speculative-config" not in baseline
    assert data["source_components"]["rapid_full_stack"]["checkout"] in candidate
    assert data["source_components"]["mlx_lm_candidate"]["checkout"] in candidate
    assert "--speculative-config" in candidate
    assert '"continuous_batching":true' in candidate


def test_flash_full_stack_enables_nax_and_baseline_disables_it():
    data = manifest()
    baseline_env, _ = campaign.arm_environment(
        data, "rapid_current", "qwen38_flash_next"
    )
    candidate_env, _ = campaign.arm_environment(data, "full_stack", "qwen38_flash_next")
    assert baseline_env["RAPID_MLX_QSA_BLOCK_SPARSE"] == "0"
    assert candidate_env["RAPID_MLX_QSA_BLOCK_SPARSE"] == "1"


def test_live_commands_require_two_part_interlock(monkeypatch):
    data = manifest()
    key = data["execution_interlock"]["environment"]
    value = data["execution_interlock"]["value"]
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="disarmed"):
        campaign.require_live_interlock(data, execute=True)
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="disarmed"):
        campaign.require_live_interlock(data, execute=False)
    campaign.require_live_interlock(data, execute=True)


def _quality_result(path: Path, arm: str, prefix: str) -> None:
    bank = campaign.load_json(QUALITY)
    payload = {
        "model_key": "qwen38_27b",
        "arm": arm,
        "responses": [
            {
                "prompt_id": prompt["prompt_id"],
                "status": "ok",
                "content": f"{prefix} {prompt['prompt_id']}",
                "reasoning_content": "SECRET REASONING",
                "wall_seconds": 123.0,
            }
            for prompt in bank["prompts"]
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_blind_packet_strips_arm_timing_reasoning_and_reverses_order(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    packet = tmp_path / "packet.json"
    mapping = tmp_path / "mapping.json"
    _quality_result(baseline, "rapid_current", "BASE")
    _quality_result(candidate, "full_stack", "STACK")

    campaign.blind_quality(
        baseline, candidate, QUALITY, packet, mapping, "predeclared-seed"
    )

    packet_text = packet.read_text(encoding="utf-8")
    assert "rapid_current" not in packet_text
    assert "full_stack" not in packet_text
    assert "SECRET REASONING" not in packet_text
    assert "wall_seconds" not in packet_text
    pairs = json.loads(packet_text)["pairs"]
    assert len(pairs) == 20
    assert pairs[0]["candidate_x"] == pairs[1]["candidate_y"]
    assert pairs[0]["candidate_y"] == pairs[1]["candidate_x"]
    assert len(json.loads(mapping.read_text())["pairs"]) == 20
