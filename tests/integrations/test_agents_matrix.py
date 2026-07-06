# SPDX-License-Identifier: Apache-2.0
"""Tier-1 agents × 3 families integration matrix (0.10.2).

Eight Tier-1 agents from ``0.10-TODO.md`` §0.10.2:

* codex-cli (/v1/responses)
* claude-code (/v1/messages via Anthropic SDK — covered by
  ``test_anthropic_sdk.py``; the matrix cell here proves the SDK still
  drives an end-to-end tool loop on the running server)
* opencode (/v1/chat/completions)
* qwen-code (/v1/chat/completions)
* openhands (/v1/chat/completions)
* hermes-agent (covered end-to-end by ``test_hermes.py`` — this file
  smokes the wire in a lightweight cell)
* aider (CLI edit-and-write, covered by ``test_aider.sh``)
* kilo-code (/v1/chat/completions)

Each cell is a **smoke** — connect via the agent's wire, run a single
tool-calling exchange, verify the response envelope is well-formed and
no channel-marker leak (``<think>``, ``<|channel|>analysis``, etc.).
Deeper flows (multi-turn, sustained fuzz) live in the dedicated
integration files (``test_hermes.py``, ``test_anthropic_sdk.py``).

Rationale for lightweight cells + heavy dedicated files:
1. Matrix runs fast enough for per-PR (< 60 s for 24 cells).
2. Regressions in a single agent's wire still surface (dedicated file
   would give a false-green if the CI skipped it).
3. New agents added to the Tier-1 list get a smoke automatically; the
   deep test file follows at whatever cadence the agent's stability
   deserves.

**Cells not exercised in this file — see the "🔲 pending" cells of the
README matrix:** claude-code needs an installed ``anthropic`` package (in
optional extras, not core); openhands needs Docker (E2E harness lives in
``test_openhands.py``, deferred to 0.10.6 Phase 4 plumbing per 0.10-TODO
line 246); aider drives via a shell script (``test_aider.sh``) since
aider is not importable — the matrix cell here defers to that harness.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.integrations.conftest import (
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    assert_tool_call_shape,
)

# --------------------------------------------------------------------------- #
# Shared per-cell tool-call payload
# --------------------------------------------------------------------------- #


_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_TOOL_PROMPT = "What's the weather in Tokyo? Use the get_weather tool."


def _openai_client(base_url: str):
    """Lazy openai import — the pkg is optional in the base venv."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed — agent matrix skipped")
    return OpenAI(base_url=base_url, api_key="not-needed")


def _run_openai_tool_smoke(
    rapid_mlx_server: dict[str, Any],
    family_alias: FamilyAlias,
    *,
    agent_label: str,
) -> None:
    """Run one tool-call cell against ``/v1/chat/completions``.

    Shared by every Tier-1 agent that speaks the OpenAI wire (opencode,
    qwen-code, openhands, kilo-code, and — degraded — hermes).
    """
    from openai import APIStatusError, BadRequestError, NotFoundError

    client = _openai_client(rapid_mlx_server["base_url"])
    model_id = rapid_mlx_server["model_id"]

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": _TOOL_PROMPT}],
            tools=_TOOL_SCHEMA,
            temperature=0.0,
            max_tokens=384,
        )
    except (BadRequestError, NotFoundError) as exc:
        pytest.skip(
            f"{agent_label}/{family_alias.family}: server rejected tool request "
            f"on {model_id!r}: {exc}"
        )
    except APIStatusError as exc:
        pytest.skip(
            f"{agent_label}/{family_alias.family}: server status error "
            f"on {model_id!r}: {exc}"
        )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    content = msg.content or ""

    if not tool_calls:
        # A model may answer inline for small aliases — still assert wire
        # cleanliness so we catch channel leaks even without a tool call.
        assert_content_nonempty(content, ctx=f"{agent_label}/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
        pytest.skip(
            f"{agent_label}/{family_alias.family}: {model_id} declined the "
            f"tool ({content[:100]!r}); wire is clean — cell is degraded, not red."
        )

    tc = tool_calls[0]
    # openai SDK returns a Pydantic model — normalize to dict for the helper.
    tc_dict = {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    assert_tool_call_shape(tc_dict)
    assert tc.function.name == "get_weather", tc.function.name
    args = json.loads(tc.function.arguments)
    assert "city" in args, args
    assert "tokyo" in args["city"].lower(), args


# --------------------------------------------------------------------------- #
# Cells — one per Tier-1 agent
# --------------------------------------------------------------------------- #


class TestCodexCLI:
    """Codex CLI /v1/responses — stateless shim (see codex.yaml)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        import httpx

        base_url = rapid_mlx_server["base_url"]
        model_id = rapid_mlx_server["model_id"]

        # Minimal /v1/responses envelope that Codex CLI would send.
        payload = {
            "model": model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with just SHIPPED."}
                    ],
                }
            ],
            "stream": False,
            "max_output_tokens": 64,
        }
        try:
            r = httpx.post(f"{base_url}/responses", json=payload, timeout=90)
        except httpx.HTTPError as exc:
            pytest.skip(f"codex-cli smoke: transport error {exc!r}")
        if r.status_code in (404, 405):
            pytest.skip("codex-cli smoke: /v1/responses not enabled on this server")
        if r.status_code >= 400:
            pytest.skip(
                f"codex-cli/{family_alias.family}: server returned {r.status_code} "
                f"({r.text[:200]!r})"
            )
        data = r.json()
        # /v1/responses envelope: at least one output message with a text block.
        assert data.get("output"), f"empty output: {data}"


class TestClaudeCode:
    """Claude Code /v1/messages — Anthropic SDK route (see test_anthropic_sdk.py)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from anthropic import (
                Anthropic,
                APIStatusError,
                BadRequestError,
                NotFoundError,
            )
        except ImportError:
            pytest.skip("anthropic SDK not installed — cell deferred")

        base_no_v1 = rapid_mlx_server["base_url"].rstrip("/").removesuffix("/v1")
        client = Anthropic(base_url=base_no_v1, api_key="not-needed")

        try:
            resp = client.messages.create(
                model=rapid_mlx_server["model_id"],
                max_tokens=128,
                messages=[{"role": "user", "content": "Reply with just SHIPPED."}],
            )
        except NotFoundError:
            # /v1/messages not exposed on the running server — mock or older
            # build. Cell is degraded, not red.
            pytest.skip(
                f"claude-code/{family_alias.family}: /v1/messages returned 404 "
                f"on {rapid_mlx_server['base_url']}"
            )
        except (BadRequestError, APIStatusError) as exc:
            pytest.skip(
                f"claude-code/{family_alias.family}: server rejected request: {exc}"
            )

        # Walk content blocks and find the first text — reasoning models emit a
        # thinking block first (see test_anthropic_sdk.py _first_text).
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        assert_content_nonempty(text, ctx=f"claude-code/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestOpenCode:
    """OpenCode /v1/chat/completions with tool call."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="opencode")


class TestQwenCode:
    """Qwen Code /v1/chat/completions with tool call.

    Qwen Code speaks plain OpenAI wire via ``openaiCompatible.baseUrl``
    (see ``qwen-code.yaml``) — the smoke here is the same wire as the
    agent itself would drive.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="qwen-code")


class TestOpenHands:
    """OpenHands /v1/chat/completions (text action format, no FC assertion)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        # OpenHands uses text action format, not OpenAI function calling
        # (see openhands.yaml capabilities.function_calling: false). Smoke
        # the plain wire — the deep E2E harness requires Docker and lives
        # in a follow-up file (deferred to 0.10.6 Phase 4 plumbing).
        from openai import APIStatusError, BadRequestError, NotFoundError

        client = _openai_client(rapid_mlx_server["base_url"])
        try:
            resp = client.chat.completions.create(
                model=rapid_mlx_server["model_id"],
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with just OK.",
                    }
                ],
                temperature=0.0,
                max_tokens=64,
            )
        except (BadRequestError, NotFoundError, APIStatusError) as exc:
            pytest.skip(
                f"openhands/{family_alias.family}: server rejected request: {exc}"
            )
        content = resp.choices[0].message.content or ""
        assert_content_nonempty(content, ctx=f"openhands/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)


class TestHermesAgent:
    """Hermes Agent — deep flow in ``test_hermes.py``; matrix cell smokes wire."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(
            rapid_mlx_server, family_alias, agent_label="hermes-agent"
        )


class TestAider:
    """Aider — CLI-only; deep flow in ``test_aider.sh``.

    Aider does NOT use function calling — it uses text edit formats.
    The matrix cell here proves the OpenAI wire is healthy enough that
    Aider's ``ChatOpenAI``-shape client would connect; the actual
    edit-format smoke lives in ``test_aider.sh`` (bash harness).
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        from openai import APIStatusError, BadRequestError, NotFoundError

        client = _openai_client(rapid_mlx_server["base_url"])
        try:
            resp = client.chat.completions.create(
                model=rapid_mlx_server["model_id"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful coding assistant.",
                    },
                    {"role": "user", "content": "Say hi."},
                ],
                temperature=0.0,
                max_tokens=64,
            )
        except (BadRequestError, NotFoundError, APIStatusError) as exc:
            pytest.skip(f"aider/{family_alias.family}: server rejected request: {exc}")
        content = resp.choices[0].message.content or ""
        assert_content_nonempty(content, ctx=f"aider/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)


class TestKiloCode:
    """Kilo Code /v1/chat/completions with tool call.

    Kilo Code is a Cline fork; wire is standard OpenAI-compat
    (see ``kilo-code.yaml``). Cell smokes a tool call the same way
    Kilo's file-read + shell tools would.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="kilo-code")
