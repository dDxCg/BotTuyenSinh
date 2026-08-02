from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import BaseMessage, ToolMessage

from src.logger import logger

from ...types import ToolRegistry
from ..state import GraphState

MAX_CONSECUTIVE_SAME_TOOL = 3


def make_tools_node(registry: ToolRegistry) -> Callable[[GraphState], dict]:
    def tools_node(state: GraphState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[BaseMessage] = []
        recent = list(state.get("recent_tools", []))
        force_finalize = False
        for call in last.tool_calls:
            name = call["name"]
            recent.append(name)
            observation = registry.call(name, call["args"])
            logger.debug("[tools] call=%s args=%s observation=%s", name, call["args"], observation)
            if len(recent) >= MAX_CONSECUTIVE_SAME_TOOL and len(set(recent[-MAX_CONSECUTIVE_SAME_TOOL:])) == 1:
                force_finalize = True
                observation += (
                    f"\n[Cảnh báo] Đã gọi '{name}' {MAX_CONSECUTIVE_SAME_TOOL} lần liên tiếp bất kể tham số. "
                    "Dừng lại, chốt câu trả lời từ những gì đã có."
                )
            tool_messages.append(ToolMessage(content=observation, tool_call_id=call["id"]))
        if force_finalize:
            logger.debug("[tools] force_finalize=True (lặp tool quá %d lần)", MAX_CONSECUTIVE_SAME_TOOL)
        return {"messages": tool_messages, "recent_tools": recent, "force_finalize": force_finalize}

    return tools_node


def route_after_tools(state: GraphState) -> str:
    return "finalize" if state.get("force_finalize") else "agent"


__all__ = ["MAX_CONSECUTIVE_SAME_TOOL", "make_tools_node", "route_after_tools"]
