from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.chatbot.config import Settings
from src.chatbot.graph import build_graph, run_graph
from src.chatbot.graph.runner import get_final_state, stream_graph_steps
from src.chatbot.agent_guardrail import DEFAULT_SUGGESTIONS
from src.rag.rag_bridge import PgVectorRetriever
from src.rag.types import Retriever


@dataclass(frozen=True)
class Reply:
    answer: str
    sources: list[dict]
    suggestions: list[str]
    grounded: bool
    top_score: float | None
    path: str


class Service:
    def __init__(
        self,
        retriever: Retriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever or PgVectorRetriever()
        self.settings = settings or Settings.from_env()
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("src.chatbot.types", "Chunk")])
        self.checkpointer = InMemorySaver(serde=serde)
        self.graph = build_graph(
            settings=self.settings,
            retriever=self.retriever,
            checkpointer=self.checkpointer,
        )
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock

    def chat(self, session_id: str, question: str) -> Reply:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống")

        with self._lock_for(session_id):
            state = run_graph(self.graph, question, thread_id=session_id)

        return Reply(
            answer=state["final_answer"],
            sources=state["sources"],
            suggestions=DEFAULT_SUGGESTIONS,
            grounded=state["grounded"],
            top_score=state["best_score"] if state["retrieved"] else None,
            path=state["path"],
        )

    def chat_stream(self, session_id: str, question: str) -> Iterator[str | Reply]:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống")

        with self._lock_for(session_id):
            for node_name in stream_graph_steps(self.graph, question, thread_id=session_id):
                yield node_name
            state = get_final_state(self.graph, thread_id=session_id)

        yield Reply(
            answer=state["final_answer"],
            sources=state["sources"],
            suggestions=DEFAULT_SUGGESTIONS,
            grounded=state["grounded"],
            top_score=state["best_score"] if state["retrieved"] else None,
            path=state["path"],
        )

    def reset(self, session_id: str) -> None:
        with self._guard:
            self._locks.pop(session_id, None)
        self.checkpointer.delete_thread(session_id)


def reply_dict(reply: Reply) -> dict:
    return asdict(reply)


__all__ = ["Reply", "Service", "reply_dict"]
