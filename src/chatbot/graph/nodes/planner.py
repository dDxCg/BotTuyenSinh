from __future__ import annotations

from collections.abc import Callable

from src.logger import logger

from ..state import GraphState

AVAILABLE_TECHNIQUES = ("query_split", "hyde")

PlannerCall = Callable[[str, tuple[str, ...]], list[str]]
"""Nhận (question, available_techniques) -> list tên kỹ thuật được chọn — tiêm fake vào đây
để test không gọi API thật, giống pattern `LLMCall`."""


def make_planner_node(classify: PlannerCall) -> Callable[[GraphState], dict]:
    def planner(state: GraphState) -> dict:
        plan = classify(state["question"], AVAILABLE_TECHNIQUES)
        logger.debug("[planner] question=%r plan=%s", state["question"], plan)
        return {"plan": plan}

    return planner


__all__ = ["AVAILABLE_TECHNIQUES", "PlannerCall", "make_planner_node"]
