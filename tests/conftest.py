"""Fixtures dùng chung: fake retriever + fake LLM, không gọi Postgres/OpenAI thật."""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from src.chatbot.rag_bridge import PgVectorRetriever


def make_fake_payload(
    chunk_id: str,
    content: str,
    score: float,
    source_link: str = "https://vinuni.edu.vn/vi/thong-tin-tuyen-sinh-chuong-trinh-dao-tao-nhan-tai-ai-thuc-chien-khoa-co-ban/",
    loai_nguon: str = "web",
) -> dict[str, Any]:
    """Payload đúng shape `src/rag/pg_retrieval.py` trả về (docs/rag-system.md §8.2)."""
    return {
        "id": chunk_id,
        "content": content,
        "cosine_similarity": score,
        "metadata": {
            "loai_nguon": loai_nguon,
            "source_link": source_link,
            "ten_tai_lieu": "Thông tin tuyển sinh",
            "muc_lon": "Điều kiện",
            "muc_nho": "",
        },
    }


def make_fake_retriever(results: list[dict[str, Any]]) -> PgVectorRetriever:
    """`PgVectorRetriever` thật, chỉ đổi nguồn dữ liệu qua `retrieve_fn` (đã hỗ trợ sẵn
    trong `src/chatbot/rag_bridge.py` để test không cần nạp Postgres/E5 local)."""

    def fake_retrieve_fn(query: str, top_k: int = 5) -> dict[str, Any]:
        return {"results": results[:top_k]}

    return PgVectorRetriever(retrieve_fn=fake_retrieve_fn)


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunctionCall


@dataclass
class FakeMessage:
    """Mô phỏng `ChatCompletionMessage` của OpenAI SDK: chỉ cần `.content`/`.tool_calls`."""

    content: str = ""
    tool_calls: list[FakeToolCall] = field(default_factory=list)


def make_tool_call(call_id: str, name: str, args: dict[str, Any]) -> FakeToolCall:
    return FakeToolCall(id=call_id, function=FakeFunctionCall(name=name, arguments=json.dumps(args)))


def scripted_llm_call(responses: list[FakeMessage]) -> Callable[[list[dict], list[dict]], FakeMessage]:
    """LLM giả: trả lần lượt từng response trong `responses` mỗi lần được gọi.

    Gọi quá số response đã soạn -> lỗi rõ ràng thay vì IndexError khó hiểu, để test
    sai kịch bản (agent lặp nhiều vòng hơn dự kiến) báo lỗi đúng chỗ.
    """
    calls: list[int] = [0]

    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> FakeMessage:
        index = calls[0]
        calls[0] += 1
        if index >= len(responses):
            raise AssertionError(f"llm_call bị gọi lần thứ {index + 1}, chỉ soạn {len(responses)} response")
        return responses[index]

    return call


def repeating_llm_call(response: FakeMessage) -> Callable[[list[dict], list[dict]], FakeMessage]:
    """LLM giả không bao giờ dừng — dùng để test anti-loop guard / recursion_limit."""

    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> FakeMessage:
        return response

    return call


@pytest.fixture
def fake_payload():
    return make_fake_payload
