"""Agent tuyển sinh dựng trên LangGraph — hợp nhất 2 luồng cũ thành 1 graph.

Thay cho:
- `admission_agent.build_admission_agent()` + `react.py`: ReAct vòng lặp tự viết,
  regex-parse `Thought:`/`Action:`/`Final Answer:`.
- `demo_service.DemoService.chat()`: regex-gate + 1 lượt gọi LLM không tool-calling,
  dùng thật ở production nhưng khác hoàn toàn luồng ReAct.

Graph:
    câu hỏi -> guardrail -> retrieve -> grounding_decision -> agent (native tool-calling)
                  |                           |                  <-> tools
                  v                           v                  -> finalize
            respond_restricted          respond_no_grounding

Xem docs/plan/langgraph-langfuse-migration.md mục 3.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .agent_tools import build_registry
from .chatbot import Chatbot
from .config import Settings
from .guardrail import UNRELATED_REPLY, classify_restricted
from .postprocess import (
    DEFAULT_SUGGESTIONS,  # noqa: F401 - re-export tiện cho caller (giữ nguyên gợi ý mặc định)
    _attachments,
    _cited_chunks,
    _clean_answer,
    _contact_markdown,
    _is_refusal_answer,
    _prioritize_sources,
)
from .prompt import render_admission_policy, render_system_prompt
from .rag_bridge import NO_GROUNDING_THRESHOLD, ChromaRetriever
from .types import Chunk, Retriever, ToolRegistry

try:  # chạy dạng `src.chatbot` (test) hoặc `chatbot` với src trên sys.path (team)
    from ..tools.attach_source_link import ATTACH_SOURCE_LINK_SCHEMA
    from ..tools.contact_support import CONTACT_SUPPORT_SCHEMA
except ImportError:  # pragma: no cover - phụ thuộc cách nạp package
    from tools.attach_source_link import ATTACH_SOURCE_LINK_SCHEMA  # type: ignore[no-redef]
    from tools.contact_support import CONTACT_SUPPORT_SCHEMA  # type: ignore[no-redef]

# Anti-loop guard: gọi cùng 1 tool >= N lần liên tiếp (bất kể args) coi là bế tắc —
# thay guard cũ ở react.py:127-134 vốn chỉ chặn khi args giống hệt nhau, nên model
# đổi nhẹ tham số là lách được (bug M07/N09/M04 trong eval/report-agent.md).
MAX_CONSECUTIVE_SAME_TOOL = 3

LLMCall = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]
"""Nhận (messages OpenAI-format, tools OpenAI-format) -> object có `.content`/`.tool_calls`
(shape của `openai`'s `ChatCompletionMessage`) — tiêm fake vào đây để test không gọi API thật."""


class GraphState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    restricted_reason: str | None
    retrieved: list[Chunk]
    grounded: bool
    best_score: float
    final_answer: str
    sources: list[dict]
    path: str
    recent_tools: list[str]


def _to_openai_tools() -> list[dict[str, Any]]:
    """Đổi schema kiểu Anthropic (`input_schema`) đã có sẵn trong `src/tools/*.py`
    sang format `tools=` của OpenAI — chưa từng được dùng thật trước graph này."""
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


def _raw_tool_calls_to_lc(raw: Any) -> list[dict[str, Any]]:
    """Đổi `.tool_calls` của OpenAI SDK response thành format `AIMessage.tool_calls`."""
    calls: list[dict[str, Any]] = []
    for tool_call in getattr(raw, "tool_calls", None) or []:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": tool_call.function.name, "args": args, "id": tool_call.id})
    return calls


def _default_llm_call(bot: Chatbot) -> LLMCall:
    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        return bot.complete_with_tools(messages, tools)

    return call


def make_guardrail_node() -> Callable[[GraphState], dict]:
    def guardrail(state: GraphState) -> dict:
        return {"restricted_reason": classify_restricted(state["question"])}

    return guardrail


def route_after_guardrail(state: GraphState) -> str:
    return "respond_restricted" if state["restricted_reason"] else "retrieve"


def make_respond_restricted_node() -> Callable[[GraphState], dict]:
    def respond_restricted(state: GraphState) -> dict:
        reason = state["restricted_reason"]
        if reason == "unrelated":
            return {"final_answer": UNRELATED_REPLY, "sources": [], "path": "out_of_scope"}
        answer = _contact_markdown(reason, state["question"])
        return {"final_answer": answer, "sources": [], "path": "contact_support"}

    return respond_restricted


def make_retrieve_node(retriever: Retriever, top_k: int = 5) -> Callable[[GraphState], dict]:
    def retrieve(state: GraphState) -> dict:
        chunks = _prioritize_sources(retriever.retrieve(state["question"], k=top_k))
        best_score = max((c.score for c in chunks), default=0.0)
        return {"retrieved": chunks, "best_score": best_score}

    return retrieve


def make_grounding_node(threshold: float = NO_GROUNDING_THRESHOLD) -> Callable[[GraphState], dict]:
    def grounding_decision(state: GraphState) -> dict:
        grounded = bool(state["retrieved"]) and state["best_score"] >= threshold
        return {"grounded": grounded}

    return grounding_decision


def route_after_grounding(state: GraphState) -> str:
    return "agent" if state["grounded"] else "respond_no_grounding"


def make_respond_no_grounding_node() -> Callable[[GraphState], dict]:
    def respond_no_grounding(state: GraphState) -> dict:
        answer = _contact_markdown("no_grounding", state["question"])
        return {"final_answer": answer, "sources": [], "path": "contact_support"}

    return respond_no_grounding


def make_agent_node(
    llm_call: LLMCall, tools: list[dict[str, Any]], policy_text: str, threshold: float
) -> Callable[[GraphState], dict]:
    def agent(state: GraphState) -> dict:
        if not state["messages"]:
            # Lượt đầu: dựng system prompt, chunk RAG đổ sẵn (không qua tool tìm kiếm).
            # tool_signatures=[] vì schema thật đi qua `tools=` — nhét thêm text mô tả
            # tool vào prompt chỉ gây nhiễu/xung đột với native tool-calling.
            system = render_system_prompt(
                tool_signatures=[],
                retrieved=state["retrieved"],
                context=policy_text,
                react=False,
                threshold=threshold,
            )
            messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=state["question"]),
            ]
        else:
            messages = state["messages"]

        raw = llm_call(convert_to_openai_messages(messages), tools)
        ai_message = AIMessage(content=raw.content or "", tool_calls=_raw_tool_calls_to_lc(raw))
        return {"messages": [ai_message]}

    return agent


def route_after_agent(state: GraphState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "finalize"


def make_tools_node(registry: ToolRegistry) -> Callable[[GraphState], dict]:
    def tools_node(state: GraphState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[BaseMessage] = []
        recent = list(state.get("recent_tools", []))
        for call in last.tool_calls:
            name = call["name"]
            recent.append(name)
            observation = registry.call(name, call["args"])
            if len(recent) >= MAX_CONSECUTIVE_SAME_TOOL and len(set(recent[-MAX_CONSECUTIVE_SAME_TOOL:])) == 1:
                observation += (
                    f"\n[Cảnh báo] Đã gọi '{name}' {MAX_CONSECUTIVE_SAME_TOOL} lần liên tiếp bất kể tham số. "
                    "Dừng lại, chốt câu trả lời từ những gì đã có."
                )
            tool_messages.append(ToolMessage(content=observation, tool_call_id=call["id"]))
        return {"messages": tool_messages, "recent_tools": recent}

    return tools_node


def make_finalize_node() -> Callable[[GraphState], dict]:
    def finalize(state: GraphState) -> dict:
        answer = _clean_answer(state["messages"][-1].content or "")
        if _is_refusal_answer(answer):
            return {"final_answer": answer, "sources": [], "path": "out_of_scope"}
        cited = _cited_chunks(answer, state["retrieved"])
        return {"final_answer": answer, "sources": _attachments(cited), "path": "agent+tool_calling"}

    return finalize


def build_graph(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    llm_call: LLMCall | None = None,
    top_k: int = 5,
    threshold: float = NO_GROUNDING_THRESHOLD,
    checkpointer: Any = None,
) -> Any:
    """Graph hợp nhất, thay cho `build_admission_agent()` (ReAct) và
    `DemoService.chat()` (regex-gate + single-shot).

    `llm_call`/`retriever` cho phép tiêm fake trong test — không cần OpenAI/Chroma thật.
    """
    retriever = retriever or ChromaRetriever()
    registry = build_registry(retriever)
    tools = _to_openai_tools()
    call = llm_call or _default_llm_call(Chatbot(settings=settings, retriever=retriever))
    policy_text = render_admission_policy(threshold=threshold)

    graph = StateGraph(GraphState)
    graph.add_node("guardrail", make_guardrail_node())
    graph.add_node("respond_restricted", make_respond_restricted_node())
    graph.add_node("retrieve", make_retrieve_node(retriever, top_k=top_k))
    graph.add_node("grounding_decision", make_grounding_node(threshold=threshold))
    graph.add_node("respond_no_grounding", make_respond_no_grounding_node())
    graph.add_node("agent", make_agent_node(call, tools, policy_text, threshold))
    graph.add_node("tools", make_tools_node(registry))
    graph.add_node("finalize", make_finalize_node())

    graph.add_edge("__start__", "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"respond_restricted": "respond_restricted", "retrieve": "retrieve"},
    )
    graph.add_edge("respond_restricted", END)
    graph.add_edge("retrieve", "grounding_decision")
    graph.add_conditional_edges(
        "grounding_decision",
        route_after_grounding,
        {"agent": "agent", "respond_no_grounding": "respond_no_grounding"},
    )
    graph.add_edge("respond_no_grounding", END)
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "messages": [],
        "restricted_reason": None,
        "retrieved": [],
        "grounded": False,
        "best_score": 0.0,
        "final_answer": "",
        "sources": [],
        "path": "",
        "recent_tools": [],
    }


def run_graph(graph: Any, question: str, thread_id: str = "default", recursion_limit: int = 25) -> GraphState:
    """Chạy 1 lượt hỏi-đáp, trả state cuối (`final_answer`/`sources`/`path`)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return graph.invoke(initial_state(question), config=config)


__all__ = ["GraphState", "build_graph", "initial_state", "run_graph"]
