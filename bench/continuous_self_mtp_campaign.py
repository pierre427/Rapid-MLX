#!/usr/bin/env python3
"""Plan, validate, capture, and blind the continuous self-MTP A/B campaign.

The default commands are deliberately CPU-only.  The two live client commands
require both ``--execute`` and the manifest's environment interlock; validation
and command rendering never import MLX or Transformers and never start a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "continuous_self_mtp_campaign.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _expected_small_artifacts(model: dict[str, Any]) -> list[tuple[Path, str]]:
    root = Path(model["model_path"])
    identity = model["artifact_identity"]
    return [
        (root / "config.json", identity["config_sha256"]),
        (root / "model.safetensors.index.json", identity["index_sha256"]),
    ]


def validate_manifest(manifest_path: Path, *, deep_hash: bool = False) -> list[str]:
    manifest = load_json(manifest_path)
    errors: list[str] = []
    if manifest.get("schema") != "rapid-mlx.continuous-self-mtp-campaign.v1":
        errors.append("unsupported campaign schema")

    components = manifest.get("source_components", {})
    for name, component in components.items():
        checkout = Path(component["checkout"])
        if not checkout.is_dir():
            errors.append(f"source {name}: checkout missing: {checkout}")
            continue
        try:
            head = git(checkout, "rev-parse", "HEAD")
        except subprocess.CalledProcessError as exc:
            errors.append(f"source {name}: git failed: {exc.output.strip()}")
            continue
        if head != component["revision"]:
            errors.append(
                f"source {name}: HEAD {head} != pinned {component['revision']}"
            )
        status = git(checkout, "status", "--porcelain")
        if status:
            errors.append(f"source {name}: checkout is dirty")
        for ancestor in component.get("required_ancestors", []):
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "merge-base",
                    "--is-ancestor",
                    ancestor,
                    head,
                ],
                check=False,
            )
            if completed.returncode != 0:
                errors.append(f"source {name}: missing required ancestor {ancestor}")

    for runtime_name, runtime in manifest.get("runtimes", {}).items():
        interpreter = Path(runtime["interpreter"])
        if not interpreter.is_file():
            errors.append(f"runtime {runtime_name}: interpreter missing: {interpreter}")

    target_tokens = manifest.get("context_protocol", {}).get("target_tokens")
    if target_tokens != [1024, 4096, 16384, 32768, 65536]:
        errors.append("context ladder must be exactly 1K, 4K, 16K, 32K, 64K")

    for model_name, model in manifest.get("models", {}).items():
        model_path = Path(model["model_path"])
        if not model_path.is_dir():
            errors.append(f"model {model_name}: artifact missing: {model_path}")
            continue
        for path, expected_hash in _expected_small_artifacts(model):
            if not path.is_file():
                errors.append(f"model {model_name}: missing {path.name}")
            elif sha256_file(path) != expected_hash:
                errors.append(f"model {model_name}: {path.name} hash mismatch")
        concurrency = model.get("service_concurrency", {})
        if set(concurrency) != {str(value) for value in target_tokens or []}:
            errors.append(f"model {model_name}: incomplete concurrency ladder")
        for sidecar in model.get("sidecars", []):
            path = Path(sidecar["path"])
            if not path.is_file():
                errors.append(f"model {model_name}: missing sidecar {path}")
                continue
            if path.stat().st_size != sidecar["size_bytes"]:
                errors.append(f"model {model_name}: sidecar size mismatch: {path}")
            if deep_hash and sidecar.get("sha256"):
                if sha256_file(path) != sidecar["sha256"]:
                    errors.append(f"model {model_name}: sidecar hash mismatch: {path}")

    prompt_path = (
        manifest_path.parent.parent / manifest["quality_protocol"]["prompt_bank"]
    )
    if not prompt_path.is_file():
        errors.append(f"quality prompt bank missing: {prompt_path}")
    else:
        bank = load_json(prompt_path)
        prompts = bank.get("prompts", [])
        ids = [prompt.get("prompt_id") for prompt in prompts]
        if len(prompts) != 10:
            errors.append(
                f"quality prompt bank has {len(prompts)} prompts, expected 10"
            )
        if len(set(ids)) != len(ids):
            errors.append("quality prompt ids are not unique")
        if any(not prompt.get("rubric") for prompt in prompts):
            errors.append("every quality prompt must have a rubric")

    for profile_name, profile in manifest.get("profiles", {}).items():
        unknown = set(profile["primary_arms"]) - set(manifest["arms"])
        if unknown:
            errors.append(f"profile {profile_name}: unknown arms {sorted(unknown)}")
    return errors


def require_live_interlock(manifest: dict[str, Any], execute: bool) -> None:
    guard = manifest["execution_interlock"]
    if not execute or os.environ.get(guard["environment"]) != guard["value"]:
        raise RuntimeError(
            "live execution is disarmed; pass --execute and set "
            f"{guard['environment']}={guard['value']} only when model/GPU work is authorized"
        )


def arm_environment(
    manifest: dict[str, Any], arm_name: str, model_name: str
) -> tuple[dict[str, str], list[str]]:
    arm = manifest["arms"][arm_name]
    source = manifest["source_components"][arm["source"]]
    paths = [source["checkout"]]
    if arm.get("dependency_source"):
        dependency = manifest["source_components"][arm["dependency_source"]]
        paths.append(dependency["checkout"])
    environment = dict(arm.get("environment", {}))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    if arm_name == "full_stack":
        environment.update(
            manifest["models"][model_name].get("candidate_environment", {})
        )
    return environment, paths


def server_command(
    manifest: dict[str, Any], arm_name: str, model_name: str, port: int
) -> tuple[dict[str, str], list[str]]:
    arm = manifest["arms"][arm_name]
    model = manifest["models"][model_name]
    runtime = manifest["runtimes"][arm["runtime"]]
    environment, _ = arm_environment(manifest, arm_name, model_name)
    max_lanes = max(int(value) for value in model["service_concurrency"].values())
    command = [
        runtime["interpreter"],
        "-m",
        "vllm_mlx.cli",
        "serve",
        model["model_path"],
        "--host",
        manifest["server"]["host"],
        "--port",
        str(port),
        "--served-model-name",
        model["served_name"],
        "--max-num-seqs",
        str(max_lanes),
        "--max-concurrent-requests",
        str(max_lanes),
        "--prefill-batch-size",
        str(min(max_lanes, 8)),
        "--completion-batch-size",
        str(max_lanes),
        *manifest["server"]["common_args"],
    ]
    if arm["speculative"]:
        config = json.dumps(model["speculative_config"], separators=(",", ":"))
        command.extend(["--speculative-config", config])
    else:
        command.append("--no-spec-decode")
    return environment, command


def shell_launch(
    manifest: dict[str, Any], arm_name: str, model_name: str, port: int
) -> str:
    environment, command = server_command(manifest, arm_name, model_name, port)
    assignments = [f"{key}={shlex.quote(value)}" for key, value in environment.items()]
    return "env " + " ".join(assignments) + " " + shlex.join(command)


def campaign_plan(
    manifest: dict[str, Any], profile_name: str, port: int
) -> dict[str, Any]:
    profile = manifest["profiles"][profile_name]
    phases: list[dict[str, Any]] = []
    for model_name, model in manifest["models"].items():
        for phase_index, (arm, order) in enumerate(profile["phase_order"], 1):
            contexts = list(manifest["context_protocol"]["target_tokens"])
            if order == "descending":
                contexts.reverse()
            cells = [
                {
                    "context_tokens": context,
                    "service_concurrency": model["service_concurrency"][str(context)],
                    "single_lane_control": 1,
                    "continuous_route_expected": (
                        arm == "full_stack"
                        and model["service_concurrency"][str(context)] >= 2
                    ),
                }
                for context in contexts
            ]
            phases.append(
                {
                    "model": model_name,
                    "phase": phase_index,
                    "arm": arm,
                    "order": order,
                    "server_launch": shell_launch(manifest, arm, model_name, port),
                    "cells": cells,
                }
            )
    return {
        "schema": "rapid-mlx.continuous-self-mtp-plan.v1",
        "created_at": utc_now(),
        "campaign_id": manifest["campaign_id"],
        "profile": profile_name,
        "description": profile["description"],
        "phases": phases,
    }


def _token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def exact_context_messages(
    tokenizer: Any, target: int, nonce: str
) -> list[dict[str, str]]:
    filler = (
        "Deterministic benchmark context. Preserve the final instruction and "
        "ignore all repeated filler sentences. "
    )

    def render(characters: int) -> list[dict[str, str]]:
        repeated = (filler * (characters // len(filler) + 1))[:characters]
        return [
            {
                "role": "user",
                "content": (
                    f"nonce={nonce}\n{repeated}\n"
                    "Explain in detail why matched concurrency is required for a fair "
                    "batching benchmark."
                ),
            }
        ]

    low, high = 0, target * 12
    while low < high:
        middle = (low + high + 1) // 2
        if _token_count(tokenizer, render(middle)) <= target:
            low = middle
        else:
            high = middle - 1
    messages = render(low)
    actual = _token_count(tokenizer, messages)
    if not target - 16 <= actual <= target:
        raise RuntimeError(f"could not render target {target}: got {actual}")
    return messages


def _cached_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if value is not None else None


def stream_chat(
    url: str,
    body: dict[str, Any],
    barrier: threading.Barrier,
    timeout: float,
) -> dict[str, Any]:
    barrier.wait(timeout=60)
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_visible: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    raw_events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                raw_events.append(event)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                visible = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                )
                if visible and first_visible is None:
                    first_visible = time.perf_counter()
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        return {"status": "error", "error": f"HTTP {exc.code}: {detail[:2000]}"}
    finished = time.perf_counter()
    return {
        "status": "ok",
        "started_monotonic": started,
        "first_visible_monotonic": first_visible,
        "finished_monotonic": finished,
        "ttft_seconds": (first_visible - started) if first_visible else None,
        "wall_seconds": finished - started,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "cached_tokens": _cached_tokens(usage),
        "finish_reason": finish_reason,
        "raw_event_sha256": canonical_sha256(raw_events),
    }


def context_cohort(
    *,
    url: str,
    model: str,
    tokenizer: Any,
    target: int,
    concurrency: int,
    decode_tokens: int,
    repetition: int,
    timeout: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency)
    bodies = []
    for lane in range(concurrency):
        nonce = f"ctx{target}-rep{repetition}-lane{lane}"
        messages = exact_context_messages(tokenizer, target, nonce)
        bodies.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": decode_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "prefix_cache": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(stream_chat, url, body, barrier, timeout): lane
            for lane, body in enumerate(bodies)
        }
        rows = []
        for future in as_completed(futures):
            row = future.result()
            row["lane"] = futures[future]
            rows.append(row)
    rows.sort(key=lambda row: row["lane"])
    ok = [row for row in rows if row["status"] == "ok"]
    starts = [
        row["first_visible_monotonic"] for row in ok if row["first_visible_monotonic"]
    ]
    finishes = [row["finished_monotonic"] for row in ok]
    completion_tokens = sum(row["usage"].get("completion_tokens", 0) for row in ok)
    decode_span = max(finishes) - min(starts) if starts and finishes else None
    cached = [row["cached_tokens"] for row in ok]
    return {
        "target_prompt_tokens": target,
        "concurrency": concurrency,
        "repetition": repetition,
        "status": "ok" if len(ok) == concurrency else "error",
        "aggregate_completion_tokens": completion_tokens,
        "aggregate_decode_tokens_per_second": (
            completion_tokens / decode_span if decode_span else None
        ),
        "median_lane_ttft_seconds": statistics.median(
            row["ttft_seconds"] for row in ok if row["ttft_seconds"] is not None
        )
        if ok
        else None,
        "cached_token_assertion": "pass"
        if all(value in (None, 0) for value in cached)
        else "fail",
        "lanes": rows,
    }


def run_context_client(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    require_live_interlock(manifest, args.execute)
    from transformers import AutoTokenizer

    model_config = manifest["models"][args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_path"], local_files_only=True, trust_remote_code=True
    )
    targets = list(manifest["context_protocol"]["target_tokens"])
    if args.order == "descending":
        targets.reverse()
    url = args.url.rstrip("/") + "/v1/chat/completions"
    output = Path(args.output)
    payload = {
        "schema": "rapid-mlx.continuous-self-mtp-context-results.v1",
        "created_at": utc_now(),
        "manifest_sha256": canonical_sha256(manifest),
        "campaign_id": manifest["campaign_id"],
        "model_key": args.model_key,
        "arm": args.arm,
        "order": args.order,
        "mode": args.mode,
        "rows": [],
    }
    atomic_json(output, payload)
    for target in targets:
        concurrency = (
            1
            if args.mode == "single-lane"
            else model_config["service_concurrency"][str(target)]
        )
        for repetition in range(1, manifest["context_protocol"]["repetitions"] + 1):
            row = context_cohort(
                url=url,
                model=model_config["served_name"],
                tokenizer=tokenizer,
                target=target,
                concurrency=concurrency,
                decode_tokens=manifest["context_protocol"]["decode_tokens"],
                repetition=repetition,
                timeout=manifest["server"]["request_timeout_seconds"],
            )
            payload["rows"].append(row)
            atomic_json(output, payload)
            print(
                f"context={target} N={concurrency} rep={repetition} "
                f"status={row['status']} agg_tps={row['aggregate_decode_tokens_per_second']}"
            )
            if row["status"] != "ok" or row["cached_token_assertion"] == "fail":
                return 1
    return 0


def _expand_quality_prompt(tokenizer: Any, prompt: dict[str, Any]) -> str:
    target = prompt.get("context_target_tokens")
    if not target:
        return prompt["prompt"]
    distractor = prompt["distractor"] + "\n"

    def render(copies: int) -> str:
        return distractor * copies + "\n" + prompt["prompt"]

    low, high = 0, int(target)
    while low < high:
        middle = (low + high + 1) // 2
        messages = [{"role": "user", "content": render(middle)}]
        if _token_count(tokenizer, messages) <= target:
            low = middle
        else:
            high = middle - 1
    rendered = render(low)
    actual = _token_count(tokenizer, [{"role": "user", "content": rendered}])
    if not target - 16 <= actual <= target:
        raise RuntimeError(f"quality context target {target} rendered as {actual}")
    return rendered


def post_quality(
    url: str,
    body: dict[str, Any],
    barrier: threading.Barrier,
    timeout: float,
) -> dict[str, Any]:
    barrier.wait(timeout=60)
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        return {"status": "error", "error": f"HTTP {exc.code}: {detail[:2000]}"}
    finished = time.perf_counter()
    choice = raw["choices"][0]
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}
    return {
        "status": "ok",
        "wall_seconds": finished - started,
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "usage": usage,
        "cached_tokens": _cached_tokens(usage),
        "raw_response": raw,
        "raw_response_sha256": canonical_sha256(raw),
    }


def run_quality_client(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    require_live_interlock(manifest, args.execute)
    from transformers import AutoTokenizer

    manifest_path = Path(args.manifest).resolve()
    bank_path = (
        manifest_path.parent.parent / manifest["quality_protocol"]["prompt_bank"]
    )
    bank = load_json(bank_path)
    prompts = bank["prompts"]
    model = manifest["models"][args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model["model_path"], local_files_only=True, trust_remote_code=True
    )
    rendered = [_expand_quality_prompt(tokenizer, prompt) for prompt in prompts]
    output = Path(args.output)
    payload = {
        "schema": "rapid-mlx.semantic-quality-results.v1",
        "created_at": utc_now(),
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": canonical_sha256(manifest),
        "prompt_bank_sha256": canonical_sha256(bank),
        "model_key": args.model_key,
        "arm": args.arm,
        "sampling": {"temperature": 0.0, "top_p": 1.0},
        "route_requirement": "greedy continuous self-MTP",
        "responses": [],
    }
    atomic_json(output, payload)
    url = args.url.rstrip("/") + "/v1/chat/completions"
    for cohort_number, indexes in enumerate(manifest["quality_protocol"]["cohorts"], 1):
        barrier = threading.Barrier(len(indexes))
        with ThreadPoolExecutor(max_workers=len(indexes)) as executor:
            futures = {}
            for lane, prompt_index in enumerate(indexes):
                prompt = prompts[prompt_index]
                body: dict[str, Any] = {
                    "model": model["served_name"],
                    "messages": [{"role": "user", "content": rendered[prompt_index]}],
                    "max_tokens": prompt["max_tokens"],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                    "prefix_cache": False,
                    "chat_template_kwargs": {"enable_thinking": prompt["thinking"]},
                }
                if prompt.get("reasoning_max_tokens") is not None:
                    body["reasoning_max_tokens"] = prompt["reasoning_max_tokens"]
                future = executor.submit(
                    post_quality,
                    url,
                    body,
                    barrier,
                    manifest["server"]["request_timeout_seconds"],
                )
                futures[future] = (lane, prompt_index, body)
            rows = []
            for future in as_completed(futures):
                lane, prompt_index, body = futures[future]
                prompt = prompts[prompt_index]
                row = future.result()
                row.update(
                    {
                        "cohort": cohort_number,
                        "lane": lane,
                        "prompt_index": prompt_index,
                        "prompt_id": prompt["prompt_id"],
                        "prompt_sha256": hashlib.sha256(
                            rendered[prompt_index].encode()
                        ).hexdigest(),
                        "request": body,
                    }
                )
                rows.append(row)
            rows.sort(key=lambda row: row["prompt_index"])
            payload["responses"].extend(rows)
            atomic_json(output, payload)
            if any(row["status"] != "ok" for row in rows):
                return 1
            if any(row["cached_tokens"] not in (None, 0) for row in rows):
                return 1
            if any(not row["content"] for row in rows):
                return 1
    return 0


def blind_quality(
    baseline_path: Path,
    candidate_path: Path,
    bank_path: Path,
    packet_path: Path,
    mapping_path: Path,
    seed: str,
) -> None:
    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    bank = load_json(bank_path)
    baseline_rows = {row["prompt_id"]: row for row in baseline["responses"]}
    candidate_rows = {row["prompt_id"]: row for row in candidate["responses"]}
    rng = random.Random(seed)
    packet = {
        "schema": "rapid-mlx.semantic-quality-blind-packet.v1",
        "suite_id": bank["suite_id"],
        "pairs": [],
    }
    mapping = {
        "schema": "rapid-mlx.semantic-quality-blind-mapping.v1",
        "seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
        "pairs": [],
    }
    for prompt in bank["prompts"]:
        prompt_id = prompt["prompt_id"]
        left, right = baseline_rows[prompt_id], candidate_rows[prompt_id]
        if left["status"] != "ok" or right["status"] != "ok":
            raise ValueError(f"cannot blind failed pair {prompt_id}")
        presentation = [
            ("baseline", left["content"]),
            ("candidate", right["content"]),
        ]
        rng.shuffle(presentation)
        for reversal in (False, True):
            shown = list(reversed(presentation)) if reversal else presentation
            pair_id = f"{prompt_id}-{'reverse' if reversal else 'forward'}"
            packet["pairs"].append(
                {
                    "pair_id": pair_id,
                    "model_key": baseline["model_key"],
                    "prompt_id": prompt_id,
                    "prompt": prompt["prompt"],
                    "rubric": prompt["rubric"],
                    "candidate_x": {"content": shown[0][1]},
                    "candidate_y": {"content": shown[1][1]},
                }
            )
            mapping["pairs"].append(
                {"pair_id": pair_id, "x": shown[0][0], "y": shown[1][0]}
            )
    atomic_json(packet_path, packet)
    atomic_json(mapping_path, mapping)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="CPU-only pin/schema check")
    validate.add_argument("--deep-artifact-hash", action="store_true")

    plan = subparsers.add_parser("plan", help="CPU-only command/matrix renderer")
    plan.add_argument(
        "--profile", choices=["product", "attribution"], default="product"
    )
    plan.add_argument("--port", type=int, default=8464)
    plan.add_argument("--output", type=Path)

    launch = subparsers.add_parser("launch-command", help="print; never execute")
    launch.add_argument("--arm", required=True)
    launch.add_argument("--model-key", required=True)
    launch.add_argument("--port", type=int, default=8464)

    context = subparsers.add_parser("context-client", help="run against a live server")
    context.add_argument("--execute", action="store_true")
    context.add_argument("--url", default="http://127.0.0.1:8464")
    context.add_argument("--arm", required=True)
    context.add_argument("--model-key", required=True)
    context.add_argument("--order", choices=["ascending", "descending"], required=True)
    context.add_argument(
        "--mode", choices=["service", "single-lane"], default="service"
    )
    context.add_argument("--output", required=True)

    quality = subparsers.add_parser("quality-client", help="run against a live server")
    quality.add_argument("--execute", action="store_true")
    quality.add_argument("--url", default="http://127.0.0.1:8464")
    quality.add_argument("--arm", required=True)
    quality.add_argument("--model-key", required=True)
    quality.add_argument("--output", required=True)

    blind = subparsers.add_parser("blind-quality", help="offline A/B packet builder")
    blind.add_argument("--baseline", type=Path, required=True)
    blind.add_argument("--candidate", type=Path, required=True)
    blind.add_argument(
        "--prompt-bank", type=Path, default=HERE / "continuous_self_mtp_quality.json"
    )
    blind.add_argument("--packet", type=Path, required=True)
    blind.add_argument("--mapping", type=Path, required=True)
    blind.add_argument("--seed", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    if args.command == "validate":
        errors = validate_manifest(manifest_path, deep_hash=args.deep_artifact_hash)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("READY (CPU-only validation; no model or MLX runtime was loaded)")
        return 0
    if args.command == "plan":
        payload = campaign_plan(manifest, args.profile, args.port)
        if args.output:
            atomic_json(args.output, payload)
            print(args.output)
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.command == "launch-command":
        print(shell_launch(manifest, args.arm, args.model_key, args.port))
        return 0
    if args.command == "context-client":
        return run_context_client(args, manifest)
    if args.command == "quality-client":
        return run_quality_client(args, manifest)
    if args.command == "blind-quality":
        blind_quality(
            args.baseline,
            args.candidate,
            args.prompt_bank,
            args.packet,
            args.mapping,
            args.seed,
        )
        print(f"packet={args.packet}\nmapping={args.mapping}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
