"""Service nối chatbot, RAG và hai tool production cho web demo."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Callable

from src.chatbot.chatbot import Chatbot
from src.chatbot.config import Settings
from src.chatbot.guardrail import UNRELATED_REPLY, classify_restricted
from src.chatbot.postprocess import (
    DEFAULT_SUGGESTIONS,
    _attachments,
    _cited_chunks,
    _clean_answer,
    _contact_markdown,
    _is_refusal_answer,
    _prioritize_sources,
)
from src.chatbot.rag_bridge import PgVectorRetriever
from src.chatbot.types import Retriever
from src.tools.contact_support import NO_GROUNDING_THRESHOLD


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
        bot_factory: Callable[[], Chatbot] | None = None,
    ) -> None:
        self.retriever = retriever or PgVectorRetriever()
        self.settings = settings
        self.bot_factory = bot_factory
        self._sessions: dict[str, Chatbot] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _session(self, session_id: str) -> tuple[Chatbot, threading.Lock]:
        with self._guard:
            bot = self._sessions.get(session_id)
            if bot is None:
                bot = self.bot_factory() if self.bot_factory else Chatbot(
                    settings=self.settings,
                    retriever=self.retriever,
                    top_k=5,
                )
                self._sessions[session_id] = bot
                self._locks[session_id] = threading.Lock()
            return bot, self._locks[session_id]

    def chat(self, session_id: str, question: str) -> Reply:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống")

        bot, lock = self._session(session_id)
        with lock:
            restricted_reason = classify_restricted(question)
            if restricted_reason:
                if restricted_reason == "unrelated":
                    bot.remember(question, UNRELATED_REPLY)
                    return Reply(
                        UNRELATED_REPLY,
                        [],
                        DEFAULT_SUGGESTIONS,
                        False,
                        None,
                        "out_of_scope",
                    )
                answer = _contact_markdown(restricted_reason, question)
                bot.remember(question, answer)
                return Reply(
                    answer, [], DEFAULT_SUGGESTIONS, False, None, "contact_support"
                )

            retrieved = self.retriever.retrieve(question, k=5)
            top_score = max((chunk.score for chunk in retrieved), default=None)
            chunks = _prioritize_sources(retrieved)
            if not chunks or top_score is None or top_score < NO_GROUNDING_THRESHOLD:
                answer = _contact_markdown("no_grounding", question)
                bot.remember(question, answer)
                return Reply(
                    answer, [], DEFAULT_SUGGESTIONS, False, top_score, "contact_support"
                )

            answer = bot.chat_with_retrieved(question, chunks)
            answer = _clean_answer(answer)
            if _is_refusal_answer(answer):
                return Reply(
                    answer=answer,
                    sources=[],
                    suggestions=DEFAULT_SUGGESTIONS,
                    grounded=False,
                    top_score=top_score,
                    path="out_of_scope",
                )
            cited_chunks = _cited_chunks(answer, chunks)
            return Reply(
                answer=answer,
                sources=_attachments(cited_chunks),
                suggestions=DEFAULT_SUGGESTIONS,
                grounded=True,
                top_score=top_score,
                path="rag+attach_source_link",
            )

    def reset(self, session_id: str) -> None:
        with self._guard:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)


def reply_dict(reply: Reply) -> dict:
    return asdict(reply)


__all__ = ["Reply", "Service", "classify_restricted", "reply_dict"]
