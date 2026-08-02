from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.rag.types import Chunk


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
    turn_active: bool
    force_finalize: bool
    hyde_document: str
    query_fragments: list[str]


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
        "turn_active": False,
        "force_finalize": False,
        "hyde_document": "",
        "query_fragments": [question],
    }


__all__ = ["GraphState", "initial_state"]
