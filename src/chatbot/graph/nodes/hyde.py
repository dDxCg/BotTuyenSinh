from __future__ import annotations

from collections.abc import Callable

from src.logger import logger

from ...llm import AgentLLM, generate_hypothetical_document
from ..state import GraphState


def make_hyde_node(llm: AgentLLM) -> Callable[[GraphState], dict]:
    def hyde(state: GraphState) -> dict:
        if "hyde" not in state["plan"]:
            return {}
        document = generate_hypothetical_document(llm, state["question"])
        logger.debug("[hyde] question=%r hypothetical=%r", state["question"], document)
        return {"hyde_document": document}

    return hyde


__all__ = ["make_hyde_node"]
