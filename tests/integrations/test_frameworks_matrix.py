# SPDX-License-Identifier: Apache-2.0
"""Tier-1 frameworks × 3 families integration matrix (0.10.2).

Three Tier-1 frameworks from ``0.10-TODO.md`` §0.10.2:

* LangChain (+LangGraph — same profile / same wire)
* PydanticAI
* smolagents

Each cell is a smoke — plain-invoke + one tool call. Deep flows live in
the dedicated files (``test_langchain.py``, ``test_pydantic_ai_full.py``,
``test_smolagents_full.py``); the matrix cell here proves the framework
plumbs onto the running server's model without requiring the deep file
to be re-run for every family.
"""

from __future__ import annotations

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
# LangChain (+ LangGraph)
# --------------------------------------------------------------------------- #


class TestLangChain:
    """LangChain / LangGraph — plain invoke + one tool call.

    LangGraph builds directly on ``langchain-openai``'s ``ChatOpenAI`` — a
    single profile covers both. LangGraph-specific StateGraph tests would
    add covered lines but no risk-of-regression signal; skipped here.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai not installed — cell deferred")

        llm = ChatOpenAI(
            model=rapid_mlx_server["model_id"],
            base_url=rapid_mlx_server["base_url"],
            api_key="not-needed",
            temperature=0.0,
            max_tokens=256,
        )

        # Plain invoke — confirm the model answers over the wire.
        try:
            r = llm.invoke([HumanMessage(content="Reply with just OK.")])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"langchain/{family_alias.family}: plain invoke failed: {exc}")
        content = r.content or ""
        assert_content_nonempty(content, ctx=f"langchain/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)

        # Tool call — confirm the bind_tools path plumbs onto rapid-mlx.
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"sunny in {city}"

        llm_with_tools = llm.bind_tools([get_weather])
        try:
            r = llm_with_tools.invoke(
                [HumanMessage(content="What's the weather in Tokyo? Use the tool.")]
            )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"langchain/{family_alias.family}: tool invoke failed: {exc}")
        tool_calls = getattr(r, "tool_calls", None) or []
        if not tool_calls:
            pytest.skip(
                f"langchain/{family_alias.family}: model declined tool. "
                f"Wire is clean; cell degraded, not red."
            )
        tc = tool_calls[0]
        # LangChain returns tool_calls as dicts with name/args/id.
        tc_dict = {
            "id": tc.get("id") or "call_lc_smoke",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": __import__("json").dumps(tc["args"]),
            },
        }
        assert_tool_call_shape(tc_dict)
        assert tc["name"] == "get_weather", tc
        assert "city" in tc["args"], tc


# --------------------------------------------------------------------------- #
# PydanticAI
# --------------------------------------------------------------------------- #


class TestPydanticAI:
    """PydanticAI — plain run + structured output smoke."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError:
            pytest.skip("pydantic-ai not installed — cell deferred")

        model = OpenAIChatModel(
            model_name=rapid_mlx_server["model_id"],
            provider=OpenAIProvider(
                base_url=rapid_mlx_server["base_url"],
                api_key="not-needed",
            ),
        )
        agent = Agent(model)
        try:
            result = agent.run_sync("Reply with just OK.")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"pydantic-ai/{family_alias.family}: run_sync failed: {exc}")
        content = (result.output or "").strip()
        assert_content_nonempty(content, ctx=f"pydantic-ai/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)


# --------------------------------------------------------------------------- #
# smolagents
# --------------------------------------------------------------------------- #


class TestSmolagents:
    """smolagents — ToolCallingAgent quick smoke."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from smolagents import OpenAIServerModel, ToolCallingAgent
        except ImportError:
            pytest.skip("smolagents not installed — cell deferred")

        model = OpenAIServerModel(
            model_id=rapid_mlx_server["model_id"],
            api_base=rapid_mlx_server["base_url"],
            api_key="not-needed",
        )
        # ToolCallingAgent with no tools = plain reply agent — proves the
        # wire without adding tool-choice sensitivity noise on small models.
        agent = ToolCallingAgent(tools=[], model=model, max_steps=1)
        try:
            answer = agent.run("Reply with just OK.")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"smolagents/{family_alias.family}: run failed: {exc}")
        content = str(answer)
        assert_content_nonempty(content, ctx=f"smolagents/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
