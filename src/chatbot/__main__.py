"""CLI chat với agent tư vấn tuyển sinh — LangGraph, RAG thật + 2 tool thật (native tool-calling).

    uv run python -m src.chatbot [--trace] [--max-steps N] [--top-k K]

Cần: `.env` có OPENAI_API, DATABASE_URL (Postgres/Neon + pgvector) đã embedding
(`python -m src.rag.pg_store`).
"""

import argparse
import sys
import time
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .agent_tools import build_registry
from .config import Settings
from .graph import GraphState, build_graph, run_graph
from .rag_bridge import PgVectorRetriever

HELP = "Gõ 'exit' để thoát, 'reset' để xoá lịch sử, 'stats' để xem số lượt embedding."


def print_trace(state: GraphState, elapsed: float, retriever: PgVectorRetriever) -> None:
    cache = ""
    if hasattr(retriever, "cache_hits"):
        cache = f" · embedding {retriever.cache_misses} miss / {retriever.cache_hits} hit"
    print(f"  {elapsed:.1f}s · {len(state['retrieved'])} chunk{cache}")

    for chunk in state["retrieved"]:
        source_type = chunk.metadata.get("source_type", "?")
        print(f"      [{chunk.score:.3f}] {chunk.source} ({source_type})")

    name_by_call_id: dict[str, str] = {}
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                name_by_call_id[call["id"]] = call["name"]
                print(f"      Action: {call['name']} {call['args']}")
        elif isinstance(message, ToolMessage):
            name = name_by_call_id.get(message.tool_call_id, "?")
            print(f"      Observation [{name}]: {message.content[:300]}")


def main() -> None:
    # Console Windows mặc định cp1252, không in được tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="chatbot")
    parser.add_argument("--trace", action="store_true", help="in tool call/observation")
    parser.add_argument("--max-steps", type=int, default=6, help="giới hạn số vòng agent<->tools")
    parser.add_argument("--top-k", type=int, default=5, help="số chunk mỗi lần truy xuất")
    args = parser.parse_args()

    try:
        settings = Settings.from_env()
    except RuntimeError as exc:  # thiếu cấu hình .env
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    retriever = PgVectorRetriever()
    graph = build_graph(settings=settings, retriever=retriever, top_k=args.top_k, checkpointer=InMemorySaver())
    tool_names = ", ".join(build_registry(retriever).names())

    print(f"{settings.model} @ {settings.base_url}")
    print(f"Tool: {tool_names}")
    print(f"{HELP}\n")

    thread_id = str(uuid.uuid4())

    while True:
        try:
            user = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"exit", "quit"}:
            break
        if user == "reset":
            thread_id = str(uuid.uuid4())  # thread mới = checkpoint cũ không còn được tham chiếu tới
            print("Đã xoá lịch sử.\n")
            continue
        if user == "stats":
            print(
                f"Embedding: {retriever.cache_misses} lượt encode, "
                f"{retriever.cache_hits} lượt dùng cache.\n"
            )
            continue

        started = time.perf_counter()
        try:
            state = run_graph(graph, user, thread_id=thread_id, recursion_limit=args.max_steps * 2 + 8)
        except Exception as exc:  # lỗi mạng/DB/vượt giới hạn vòng lặp không được làm chết phiên chat
            print(f"Lỗi: {type(exc).__name__}: {exc}")
            print("Kiểm tra: pgvector đã build chưa (`python -m src.rag.pg_store`).\n")
            continue
        elapsed = time.perf_counter() - started

        if args.trace:
            print_trace(state, elapsed, retriever)
        print(f"Bot: {state['final_answer']}\n")


if __name__ == "__main__":
    main()
