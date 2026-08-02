from __future__ import annotations

from collections.abc import Callable

from src.logger import logger
from src.rag.rag_bridge import NO_GROUNDING_THRESHOLD

from ...agent_guardrail import _contact_markdown
from ..state import GraphState


def make_grounding_node(threshold: float = NO_GROUNDING_THRESHOLD) -> Callable[[GraphState], dict]:
    def grounding_decision(state: GraphState) -> dict:
        grounded = bool(state["retrieved"]) and state["best_score"] >= threshold
        logger.debug(
            "[grounding_decision] best_score=%.3f threshold=%.3f grounded=%s",
            state["best_score"],
            threshold,
            grounded,
        )
        return {"grounded": grounded}

    return grounding_decision


def route_after_grounding(state: GraphState) -> str:
    return "agent" if state["grounded"] else "respond_no_grounding"


def make_respond_no_grounding_node() -> Callable[[GraphState], dict]:
    def respond_no_grounding(state: GraphState) -> dict:
        logger.debug("[respond_no_grounding] best_score=%.3f", state["best_score"])
        answer = _contact_markdown("no_grounding", state["question"])
        return {"final_answer": answer, "sources": [], "path": "contact_support"}

    return respond_no_grounding


__all__ = ["make_grounding_node", "make_respond_no_grounding_node", "route_after_grounding"]
