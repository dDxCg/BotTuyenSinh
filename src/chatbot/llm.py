"""Gọi LLM tập trung cho agent — dùng chung bởi các node trong `graph/` và các tool
sẽ thêm sau này. `AgentLLM` bọc `src/llm_client.py` (client OpenAI-compatible thô) thành
1 `LLMCall` gọi được thẳng (`agent_llm(messages, tools)`), không cần hàm bọc thêm."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from src.llm_client import chat_client

from .config import Settings

try:
    from ..tools.attach_source_link import ATTACH_SOURCE_LINK_SCHEMA
    from ..tools.contact_support import CONTACT_SUPPORT_SCHEMA
except ImportError:
    from tools.attach_source_link import ATTACH_SOURCE_LINK_SCHEMA
    from tools.contact_support import CONTACT_SUPPORT_SCHEMA

LLMCall = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]
"""Nhận (messages OpenAI-format, tools OpenAI-format) -> object có `.content`/`.tool_calls`
(shape của `openai`'s `ChatCompletionMessage`) — tiêm fake vào đây để test không gọi API thật."""


class AgentLLM:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.client = chat_client(
            self.settings.api_key,
            self.settings.base_url,
            self.settings.timeout_seconds,
            self.settings.max_retries,
        )

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            tools=tools or None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        return response.choices[0].message


class JudgeLLM:
    """Model rẻ/nhỏ riêng cho phân loại (planner, faithfulness check...) — tách khỏi
    `AgentLLM` (model chính, đắt hơn, dùng để sinh câu trả lời)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chat_client(
            settings.judge_api_key,
            settings.judge_base_url,
            settings.judge_timeout_seconds,
            settings.judge_max_retries,
        )

    def classify_techniques(self, question: str, available: tuple[str, ...]) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Phân tích câu hỏi tuyển sinh sau, chọn 0 hoặc nhiều kỹ thuật cần dùng "
                    f"trong danh sách: {', '.join(available)}.\n"
                    "- 'query_split': câu hỏi gộp nhiều ý/chủ đề khác nhau trong 1 câu.\n"
                    "- 'hyde': câu hỏi mơ hồ/ngắn, khó match trực tiếp với văn phong tài liệu.\n"
                    "Không chắc hoặc câu đơn giản 1 ý rõ ràng: trả mảng rỗng []. Trả đúng 1 mảng "
                    "JSON các chuỗi tên kỹ thuật, không thêm chữ nào khác."
                ),
            },
            {"role": "user", "content": question},
        ]
        response = self.client.chat.completions.create(
            model=self.settings.judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=64,
        )
        raw = response.choices[0].message.content or "[]"
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(plan, list):
            return []
        return [item for item in plan if item in available]


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


def generate_hypothetical_document(llm: AgentLLM, question: str) -> str:
    """HyDE: sinh 1 đoạn văn giả định trả lời câu hỏi, dùng để embed thay câu hỏi gốc khi
    retrieve — không cần tool schema nên gọi thẳng client, không qua `AgentLLM.__call__`."""

    messages = [
        {
            "role": "system",
            "content": (
                "Viết 1 đoạn văn ngắn (3-5 câu) trả lời câu hỏi sau như thể trích từ tài liệu "
                "chính thức, dù không chắc đúng. Không thêm lời dẫn."
            ),
        },
        {"role": "user", "content": question},
    ]
    response = llm.client.chat.completions.create(
        model=llm.settings.model,
        messages=messages,
        temperature=llm.settings.temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


def rewrite_query_candidates(llm: AgentLLM, question: str, candidates: list[str]) -> list[str]:
    """Câu hỏi gộp bị regex tách thành nhiều candidate — candidate quá ngắn/cụt (vd "email")
    thiếu ngữ cảnh khiến so sánh embedding sai lệch. Viết lại mỗi candidate thành 1 câu hỏi
    độc lập, đầy đủ ngữ cảnh, trước khi embed để tách theo semantic."""

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    messages = [
        {
            "role": "system",
            "content": (
                "Câu hỏi gốc bị tách thành nhiều phần theo dấu câu. Viết lại MỖI phần thành 1 "
                "câu hỏi độc lập, đầy đủ ngữ cảnh (không viết tắt, không cụt lủn 1-2 từ), giữ "
                "đúng ý gốc. Trả về đúng 1 mảng JSON các chuỗi, cùng thứ tự, không thêm chữ nào khác."
            ),
        },
        {"role": "user", "content": f"Câu hỏi gốc: {question}\n\nCác phần:\n{numbered}"},
    ]
    response = llm.client.chat.completions.create(
        model=llm.settings.model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content or "[]"
    try:
        rewritten = json.loads(raw)
    except json.JSONDecodeError:
        return candidates
    if isinstance(rewritten, list) and len(rewritten) == len(candidates):
        return [str(item).strip() for item in rewritten]
    return candidates


__all__ = [
    "AgentLLM",
    "JudgeLLM",
    "LLMCall",
    "generate_hypothetical_document",
    "raw_tool_calls_to_lc",
    "rewrite_query_candidates",
    "to_openai_tools",
]
