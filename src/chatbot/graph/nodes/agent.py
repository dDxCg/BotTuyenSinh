from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
)

from src.logger import logger

from ...llm import LLMCall, raw_tool_calls_to_lc
from ...prompts import render_rag_context, render_system_prompt
from ..state import GraphState


def make_agent_node(
    llm_call: LLMCall, tools: list[dict[str, Any]], policy_text: str, threshold: float
) -> Callable[[GraphState], dict]:
    def agent(state: GraphState) -> dict:
        new_messages: list[BaseMessage] = []
        if not state["messages"]:
            system = render_system_prompt(
                tool_signatures=[],
                context=policy_text,
                react=False,
                threshold=threshold,
            )
            new_messages.append(SystemMessage(content=system))

        if state.get("turn_active"):
            rag_context = render_rag_context(retrieved=state["retrieved"], threshold=threshold)
            new_messages.append(SystemMessage(content=rag_context))
            new_messages.append(HumanMessage(content=state["question"]))

        history = state["messages"] + new_messages
        openai_messages = convert_to_openai_messages(history)
        logger.debug("[agent] prompt sent (%d messages): %s", len(openai_messages), openai_messages)
        raw = llm_call(openai_messages, tools)
        ai_message = AIMessage(content=raw.content or "", tool_calls=raw_tool_calls_to_lc(raw))
        logger.debug(
            "[agent] response content=%r tool_calls=%s", ai_message.content, ai_message.tool_calls
        )
        return {"messages": [*new_messages, ai_message], "turn_active": False}

    return agent


def route_after_agent(state: GraphState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "finalize"


__all__ = ["make_agent_node", "route_after_agent"]
