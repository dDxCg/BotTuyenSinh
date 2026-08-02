from __future__ import annotations

from typing import Any

from .state import GraphState, initial_state


def run_graph(graph: Any, question: str, thread_id: str = "default", recursion_limit: int = 25) -> GraphState:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return graph.invoke(initial_state(question), config=config)


__all__ = ["run_graph"]
