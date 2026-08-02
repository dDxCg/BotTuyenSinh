from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.rag.rag_bridge import NO_GROUNDING_THRESHOLD, PgVectorRetriever
from src.rag.types import Retriever

from ..config import Settings
from ..llm import AgentLLM, JudgeLLM, LLMCall, to_openai_tools
from ..prompts import render_admission_policy
from .nodes.agent import make_agent_node, route_after_agent
from .nodes.finalize import make_finalize_node
from .nodes.grounding import (
    make_grounding_node,
    make_respond_no_grounding_node,
    route_after_grounding,
)
from .nodes.guardrail import (
    make_guardrail_node,
    make_respond_restricted_node,
    route_after_guardrail,
)
from .nodes.hyde import make_hyde_node
from .nodes.planner import PlannerCall, make_planner_node
from .nodes.query_split import make_query_split_node
from .nodes.retrieve import make_retrieve_node
from .nodes.tools import make_tools_node, route_after_tools
from .state import GraphState
from ..agent_tools import build_registry


def build_graph(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    llm_call: LLMCall | None = None,
    judge_call: PlannerCall | None = None,
    top_k: int | None = None,
    threshold: float = NO_GROUNDING_THRESHOLD,
    checkpointer: Any = None,
) -> Any:
    settings = settings or Settings.from_env()
    top_k = top_k if top_k is not None else settings.top_k
    retriever = retriever or PgVectorRetriever()
    registry = build_registry(retriever)
    tools = to_openai_tools()
    call = llm_call or AgentLLM(settings=settings)
    classify = judge_call or JudgeLLM(settings).classify_techniques
    policy_text = render_admission_policy(threshold=threshold)

    graph = StateGraph(GraphState)
    graph.add_node("guardrail", make_guardrail_node())
    graph.add_node("respond_restricted", make_respond_restricted_node())
    graph.add_node("planner", make_planner_node(classify))
    graph.add_node("query_split", make_query_split_node(AgentLLM(settings=settings)))
    graph.add_node("hyde", make_hyde_node(AgentLLM(settings=settings)))
    graph.add_node("retrieve", make_retrieve_node(retriever, top_k=top_k))
    graph.add_node("grounding_decision", make_grounding_node(threshold=threshold))
    graph.add_node("respond_no_grounding", make_respond_no_grounding_node())
    graph.add_node("agent", make_agent_node(call, tools, policy_text, threshold))
    graph.add_node("tools", make_tools_node(registry))
    graph.add_node("finalize", make_finalize_node())

    # Graph shape cố định — planner quyết định qua state["plan"], không qua topology.
    # query_split/hyde tự no-op (return {}) nếu tên kỹ thuật của mình không có trong plan.
    graph.add_edge("__start__", "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"respond_restricted": "respond_restricted", "retrieve": "planner"},
    )
    graph.add_edge("respond_restricted", END)
    graph.add_edge("planner", "query_split")
    graph.add_edge("query_split", "hyde")
    graph.add_edge("hyde", "retrieve")
    graph.add_edge("retrieve", "grounding_decision")
    graph.add_conditional_edges(
        "grounding_decision",
        route_after_grounding,
        {"agent": "agent", "respond_no_grounding": "respond_no_grounding"},
    )
    graph.add_edge("respond_no_grounding", END)
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


__all__ = ["build_graph"]
