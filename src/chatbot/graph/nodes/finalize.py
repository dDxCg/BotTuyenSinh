from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage

from src.logger import logger

from ...agent_guardrail import (
    _attachments,
    _cited_chunks,
    _clean_answer,
    _contact_markdown,
    _is_refusal_answer,
)
from ..state import GraphState


def _last_ai_content(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    return ""


def make_finalize_node() -> Callable[[GraphState], dict]:
    def finalize(state: GraphState) -> dict:
        if state.get("force_finalize"):
            answer = _clean_answer(_last_ai_content(state["messages"]))
            if not answer:
                answer = _contact_markdown("no_grounding", state["question"])
                logger.debug("[finalize] force_finalize không có AI content, fallback contact_support")
                return {"final_answer": answer, "sources": [], "path": "contact_support"}
        else:
            answer = _clean_answer(state["messages"][-1].content or "")

        if _is_refusal_answer(answer):
            logger.debug("[finalize] refusal answer=%r", answer)
            return {"final_answer": answer, "sources": [], "path": "out_of_scope"}
        cited = _cited_chunks(answer, state["retrieved"])
        logger.debug(
            "[finalize] answer=%r cited_chunks=%s",
            answer,
            [c.metadata.get("chunk_id", c.source) for c in cited],
        )
        return {"final_answer": answer, "sources": _attachments(cited), "path": "agent+tool_calling"}

    return finalize


__all__ = ["make_finalize_node"]
