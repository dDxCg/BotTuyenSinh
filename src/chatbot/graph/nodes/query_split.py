from __future__ import annotations

from collections.abc import Callable

from src.logger import logger

from ...llm import AgentLLM
from ...query_split import split_subquestions
from ..state import GraphState


def make_query_split_node(llm: AgentLLM) -> Callable[[GraphState], dict]:
    def query_split(state: GraphState) -> dict:
        fragments = split_subquestions(state["question"], llm)
        logger.debug("[query_split] question=%r fragments=%s", state["question"], fragments)
        return {"query_fragments": fragments}

    return query_split


__all__ = ["make_query_split_node"]
