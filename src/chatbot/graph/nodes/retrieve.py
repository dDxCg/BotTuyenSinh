from __future__ import annotations

from collections.abc import Callable

from src.logger import logger

from src.rag.types import Retriever

from ...agent_guardrail import _prioritize_sources
from ..state import GraphState


def make_retrieve_node(retriever: Retriever, top_k: int = 5) -> Callable[[GraphState], dict]:
    def retrieve(state: GraphState) -> dict:
        chunks = _prioritize_sources(retriever.retrieve(state["question"], k=top_k))
        best_score = max((c.score for c in chunks), default=0.0)
        logger.debug(
            "[retrieve] question=%r top_k=%d best_score=%.3f chunks=%s",
            state["question"],
            top_k,
            best_score,
            [(c.metadata.get("chunk_id", c.source), round(c.score, 3)) for c in chunks],
        )
        for chunk in chunks:
            logger.debug("[retrieve] chunk %s: %s", chunk.metadata.get("chunk_id", chunk.source), chunk.text)
        return {"retrieved": chunks, "best_score": best_score, "turn_active": True}

    return retrieve


__all__ = ["make_retrieve_node"]
