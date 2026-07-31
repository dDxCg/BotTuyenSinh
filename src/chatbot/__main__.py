"""CLI chat với agent tư vấn tuyển sinh — RAG thật + 2 tool thật.

    uv run python -m src.chatbot [--trace] [--max-steps N] [--top-k K]

Cần: `.env` có OPENAI_API, model E5-large local, và ChromaDB đã embedding
(`python src/rag/embedding.py`).
"""

import argparse
import sys
import time

from .admission_agent import build_admission_agent
from .react import ReActAgent, ReActResult

HELP = "Gõ 'exit' để thoát, 'reset' để xoá lịch sử, 'stats' để xem số lượt embedding."


def print_trace(agent: ReActAgent, result: ReActResult, elapsed: float) -> None:
    retriever = agent.bot.retriever
    cache = ""
    if hasattr(retriever, "cache_hits"):
        cache = f" · embedding {retriever.cache_misses} miss / {retriever.cache_hits} hit"
    print(f"  {elapsed:.1f}s · {len(result.retrieved)} chunk{cache}")

    for chunk in result.retrieved:
        source_type = chunk.metadata.get("source_type", "?")
        print(f"      [{chunk.score:.3f}] {chunk.source} ({source_type})")

    for index, step in enumerate(result.steps, 1):
        print(f"  [{index}] Thought: {step.thought}")
        if step.action:
            print(f"      Action: {step.action} {step.action_input}")
            print(f"      Observation: {step.observation[:300]}")


def main() -> None:
    # Console Windows mặc định cp1252, không in được tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="chatbot")
    parser.add_argument("--trace", action="store_true", help="in Thought/Action/Observation")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5, help="số chunk mỗi lần truy xuất")
    args = parser.parse_args()

    try:
        agent = build_admission_agent(max_steps=args.max_steps, top_k=args.top_k)
    except RuntimeError as exc:  # thiếu cấu hình .env
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    bot = agent.bot
    print(f"{bot.settings.model} @ {bot.settings.base_url}")
    print(f"Tool: {', '.join(agent.registry.names())}")
    print(f"{HELP}\n")

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
            agent.reset()
            print("Đã xoá lịch sử.\n")
            continue
        if user == "stats":
            retriever = agent.bot.retriever
            print(
                f"Embedding local: {retriever.cache_misses} lượt encode, "
                f"{retriever.cache_hits} lượt dùng cache.\n"
            )
            continue

        started = time.perf_counter()
        try:
            result = agent.run(user)
        except Exception as exc:  # lỗi mạng/ChromaDB không được làm chết phiên chat
            print(f"Lỗi: {type(exc).__name__}: {exc}")
            print(
                "Kiểm tra: model local đã tải chưa (`python -m src.rag.download_model`) "
                "và ChromaDB đã build chưa (`python src/rag/embedding.py`).\n"
            )
            continue
        elapsed = time.perf_counter() - started

        if args.trace:
            print_trace(agent, result, elapsed)
        if result.stopped_early:
            print("  (hết số bước cho phép)")
        print(f"Bot: {result.answer}\n")


if __name__ == "__main__":
    main()
