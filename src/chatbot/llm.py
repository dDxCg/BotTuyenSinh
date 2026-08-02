"""Gọi LLM tập trung cho agent — dùng chung bởi các node trong `graph/` và các tool
sẽ thêm sau này (không phải client HTTP thô, đó là `src/llm_client.py`)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .chatbot import Chatbot

try:
    from ..tools.attach_source_link import ATTACH_SOURCE_LINK_SCHEMA
    from ..tools.contact_support import CONTACT_SUPPORT_SCHEMA
except ImportError:
    from tools.attach_source_link import ATTACH_SOURCE_LINK_SCHEMA
    from tools.contact_support import CONTACT_SUPPORT_SCHEMA

LLMCall = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]
"""Nhận (messages OpenAI-format, tools OpenAI-format) -> object có `.content`/`.tool_calls`
(shape của `openai`'s `ChatCompletionMessage`) — tiêm fake vào đây để test không gọi API thật."""


def to_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        for schema in (ATTACH_SOURCE_LINK_SCHEMA, CONTACT_SUPPORT_SCHEMA)
    ]


def raw_tool_calls_to_lc(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tool_call in getattr(raw, "tool_calls", None) or []:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": tool_call.function.name, "args": args, "id": tool_call.id})
    return calls


def default_llm_call(bot: Chatbot) -> LLMCall:
    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        return bot.complete_with_tools(messages, tools)

    return call


__all__ = ["LLMCall", "default_llm_call", "raw_tool_calls_to_lc", "to_openai_tools"]
