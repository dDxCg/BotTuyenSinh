from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .state import GraphState, initial_state


def run_graph(graph: Any, question: str, thread_id: str = "default", recursion_limit: int = 25) -> GraphState:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return graph.invoke(initial_state(question), config=config)


def stream_graph_steps(
    graph: Any, question: str, thread_id: str = "default", recursion_limit: int = 25
) -> Iterator[str]:
    """Yield tên node ngay khi chạy xong — dùng để bắn progress event qua SSE.
    `GraphState` cuối cùng lấy qua `graph.get_state(config)` sau khi stream cạn,
    vì `stream_mode="updates"` chỉ trả phần state mỗi node ghi thêm, không phải state đầy đủ."""

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    for update in graph.stream(initial_state(question), config=config, stream_mode="updates"):
        for node_name in update:
            yield node_name


def get_final_state(graph: Any, thread_id: str = "default") -> GraphState:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.get_state(config).values


__all__ = ["run_graph", "stream_graph_steps", "get_final_state"]
