from __future__ import annotations

from collections.abc import Callable

from src.logger import logger
from src.rag.types import Chunk, Retriever

from ...agent_guardrail import _prioritize_sources
from ..state import GraphState


def _chunk_key(chunk: Chunk) -> str:
    chunk_id = chunk.metadata.get("chunk_id")
    return str(chunk_id) if chunk_id else f"{chunk.source}:{chunk.text}"


def _merge_chunks(chunk_lists: list[list[Chunk]], cap: int) -> list[Chunk]:
    best_by_key: dict[str, Chunk] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            key = _chunk_key(chunk)
            if key not in best_by_key or chunk.score > best_by_key[key].score:
                best_by_key[key] = chunk
    merged = sorted(best_by_key.values(), key=lambda c: c.score, reverse=True)
    return merged[:cap]


def make_retrieve_node(retriever: Retriever, top_k: int = 5) -> Callable[[GraphState], dict]:
    def retrieve(state: GraphState) -> dict:
        queries = list(state.get("query_fragments") or [state["question"]])
        hyde_document = state.get("hyde_document") or ""
        if hyde_document:
            queries.append(hyde_document)

        if len(queries) > 1:
            logger.debug("[retrieve] retrieve theo %d query: %s", len(queries), queries)
            per_query = [retriever.retrieve(query, k=top_k) for query in queries]
            chunks = _merge_chunks(per_query, cap=top_k * len(queries))
        else:
            chunks = retriever.retrieve(queries[0], k=top_k)

        chunks = _prioritize_sources(chunks)
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
