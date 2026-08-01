"""Graph LangGraph mới (`src/chatbot/graph.py`) — toàn bộ offline: LLM và retriever
đều là fake, không gọi OpenRouter/Chroma/E5 thật (không cần OPENAI_API, không nạp
model embedding). Khớp quy ước `pytest -m 'not live and not e2e'` trong pyproject.toml.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from conftest import (
    FakeMessage,
    make_fake_payload,
    make_fake_retriever,
    make_tool_call,
    repeating_llm_call,
    scripted_llm_call,
)

from src.chatbot.agent_tools import build_registry
from src.chatbot.graph import build_graph, make_tools_node, run_graph
from src.chatbot.rag_bridge import NO_GROUNDING_THRESHOLD


def _never_call(*_args, **_kwargs):
    raise AssertionError("không được gọi trong nhánh này")


def test_guardrail_short_circuits_before_retrieval_and_llm() -> None:
    """Câu hỏi bị guardrail chặn -> không được đụng tới retriever/LLM."""
    graph = build_graph(retriever=make_fake_retriever(_never_call), llm_call=_never_call)
    state = run_graph(graph, "Em có nên nộp hồ sơ không ạ?")
    assert state["path"] == "contact_support"
    assert "Hotline" in state["final_answer"] or "hotline" in state["final_answer"].lower()


def test_unrelated_question_returns_canned_reply() -> None:
    graph = build_graph(retriever=make_fake_retriever(_never_call), llm_call=_never_call)
    state = run_graph(graph, "Con gà có trước hay quả trứng có trước?")
    assert state["path"] == "out_of_scope"
    assert "AI Thực Chiến" in state["final_answer"]


def test_no_grounding_short_circuits_before_llm() -> None:
    """Score thấp hơn ngưỡng -> contact_support, không gọi LLM (đúng thiết kế:
    grounding_decision là node duy nhất quyết định, thay 3 chỗ rải rác cũ)."""
    low_score_results = [make_fake_payload("c1", "Nội dung không liên quan lắm", score=0.3)]
    graph = build_graph(retriever=make_fake_retriever(low_score_results), llm_call=_never_call)
    state = run_graph(graph, "Điều kiện dự tuyển là gì?")
    assert state["path"] == "contact_support"
    assert state["sources"] == []


def test_grounded_question_runs_tool_calling_loop_and_attaches_sources() -> None:
    """Đủ căn cứ -> agent gọi tool `attach_source_link` (native tool-calling, không
    còn regex Thought/Action) rồi chốt Final Answer -> sources được đính kèm."""
    results = [make_fake_payload("c1", "Điều kiện dự tuyển gồm...", score=0.92)]
    retriever = make_fake_retriever(results)

    llm_call = scripted_llm_call(
        [
            FakeMessage(tool_calls=[make_tool_call("call_1", "attach_source_link", {"chunk_ids": ["c1"]})]),
            FakeMessage(content="Điều kiện dự tuyển gồm tốt nghiệp THPT và đủ 18 tuổi."),
        ]
    )
    graph = build_graph(retriever=retriever, llm_call=llm_call)
    state = run_graph(graph, "Điều kiện dự tuyển là gì?")

    assert state["path"] == "agent+tool_calling"
    assert "18 tuổi" in state["final_answer"]
    assert len(state["sources"]) == 1
    assert state["sources"][0]["source_link"].startswith("https://vinuni.edu.vn")


def test_grounded_question_without_tool_call_finalizes_directly() -> None:
    """Model có thể chốt Final Answer ngay lượt đầu, không nhất thiết gọi tool nào."""
    results = [make_fake_payload("c1", "Học phí là...", score=0.95)]
    llm_call = scripted_llm_call([FakeMessage(content="Học phí là 20 triệu/kỳ.")])
    graph = build_graph(retriever=make_fake_retriever(results), llm_call=llm_call)
    state = run_graph(graph, "Học phí bao nhiêu?")
    assert state["path"] == "agent+tool_calling"
    assert state["final_answer"] == "Học phí là 20 triệu/kỳ."


def test_anti_loop_guard_warns_after_repeated_same_tool_calls() -> None:
    """Thay guard cũ (react.py) chỉ chặn khi args giống hệt: guard mới chặn theo TÊN
    tool lặp liên tiếp, bất kể args có đổi hay không (bug M07/N09/M04 cũ)."""
    results = [make_fake_payload("c1", "...", score=0.9), make_fake_payload("c2", "...", score=0.9)]
    retriever = make_fake_retriever(results)
    retriever.retrieve("điều kiện dự tuyển", k=5)  # nạp chunk_by_id để attach_source_link tra được nguồn
    registry = build_registry(retriever)
    tools_node = make_tools_node(registry)

    state = {"recent_tools": []}
    for i, chunk_id in enumerate(["c1", "c2", "c1"]):
        call = make_tool_call(f"call_{i}", "attach_source_link", {"chunk_ids": [chunk_id]})
        state["messages"] = [AIMessage(content="", tool_calls=[{"name": call.function.name, "args": {"chunk_ids": [chunk_id]}, "id": call.id}])]
        result = tools_node(state)
        state["recent_tools"] = result["recent_tools"]
        last_observation = result["messages"][-1].content

    assert "Cảnh báo" in last_observation
    assert state["recent_tools"] == ["attach_source_link"] * 3


def test_runaway_tool_calling_hits_recursion_limit() -> None:
    """Guard mềm (cảnh báo trong Observation) không bắt buộc model dừng — recursion_limit
    của LangGraph là lưới an toàn cứng cuối cùng khi model bỏ qua cảnh báo."""
    results = [make_fake_payload("c1", "...", score=0.9)]
    llm_call = repeating_llm_call(
        FakeMessage(tool_calls=[make_tool_call("call_x", "attach_source_link", {"chunk_ids": ["c1"]})])
    )
    graph = build_graph(retriever=make_fake_retriever(results), llm_call=llm_call)
    with pytest.raises(GraphRecursionError):
        run_graph(graph, "Điều kiện dự tuyển là gì?", recursion_limit=6)
