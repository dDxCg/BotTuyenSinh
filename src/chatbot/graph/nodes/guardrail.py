from __future__ import annotations

from collections.abc import Callable

from src.logger import logger

from ...agent_guardrail import UNRELATED_REPLY, _contact_markdown, classify_restricted
from ..state import GraphState


def make_guardrail_node() -> Callable[[GraphState], dict]:
    def guardrail(state: GraphState) -> dict:
        reason = classify_restricted(state["question"])
        logger.debug("[guardrail] question=%r restricted_reason=%r", state["question"], reason)
        return {"restricted_reason": reason}

    return guardrail


def route_after_guardrail(state: GraphState) -> str:
    return "respond_restricted" if state["restricted_reason"] else "retrieve"


def make_respond_restricted_node() -> Callable[[GraphState], dict]:
    def respond_restricted(state: GraphState) -> dict:
        reason = state["restricted_reason"]
        logger.debug("[respond_restricted] reason=%r", reason)
        if reason == "unrelated":
            return {"final_answer": UNRELATED_REPLY, "sources": [], "path": "out_of_scope"}
        answer = _contact_markdown(reason, state["question"])
        return {"final_answer": answer, "sources": [], "path": "contact_support"}

    return respond_restricted


__all__ = ["make_guardrail_node", "make_respond_restricted_node", "route_after_guardrail"]
