"""Top-k chunks bằng pgvector (Postgres/Neon) — thay ChromaDB.

Trả JSON cùng shape với bản Chroma cũ (docs/rag-system.md §8.2) nên
`rag_bridge.payload_to_chunks()` không cần đổi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from . import embedding
except ImportError:  # pragma: no cover - chạy trực tiếp không qua package
    import embedding  # type: ignore[no-redef]

from src.db_client import DBHandler, DbConfigError, vector_literal

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_TOP_K = 5


class RetrievalError(RuntimeError):
    """Dữ liệu hoặc cấu hình retrieval không hợp lệ."""


def clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if v not in (None, "", [], {})}


def query_embedding(question: str, config: "embedding.EmbeddingConfig") -> list[float]:
    return embedding.embed_query(question, config=config)


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    env_file: Path = DEFAULT_ENV_FILE,
    table_name: str | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise RetrievalError("Câu hỏi không được để trống")
    if top_k < 1:
        raise RetrievalError("top_k phải lớn hơn 0")

    embedding.load_env_file(env_file)
    config = embedding.load_config(env_file)
    table = table_name or os.getenv("PG_TABLE", "").strip() or config.table_name

    vector = query_embedding(question, config)
    with DBHandler() as db:
        stored_models = {row[0] for row in db.execute(f"SELECT DISTINCT embedding_model FROM {table}")}
        if not stored_models:
            raise RetrievalError(f"Bảng '{table}' chưa có record")
        if stored_models != {config.model}:
            raise RetrievalError(
                f"Bảng '{table}' dùng model {stored_models}, nhưng cấu hình hiện "
                f"tại là '{config.model}'"
            )

        rows = db.execute(
            f"""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(vector), vector_literal(vector), top_k),
        )

    results = [
        {
            "rank": rank,
            "id": chunk_id,
            "cosine_similarity": round(float(cosine_similarity), 6),
            "content": content,
            "metadata": clean_metadata(metadata),
        }
        for rank, (chunk_id, content, metadata, cosine_similarity) in enumerate(rows, start=1)
    ]
    return {
        "question": question,
        "embedding_model": config.model,
        "returned_results": len(results),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tìm top chunks qua pgvector.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--table", default=None)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    question = args.question if args.question is not None else input("Nhập câu hỏi: ")
    payload = retrieve(
        question=question, top_k=args.top_k, env_file=args.env_file, table_name=args.table
    )
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2))


if __name__ == "__main__":
    try:
        main()
    except (RetrievalError, DbConfigError, embedding.ConfigurationError, embedding.EmbeddingError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
