from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from . import embedding
except ImportError:
    import embedding

from src.db_client import DBHandler, DbConfigError, vector_literal

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class PgStoreError(RuntimeError):
    pass


def ensure_schema(db: DBHandler, table: str, dimension: int, recreate: bool) -> None:
    db.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if recreate:
        db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding VECTOR({dimension}) NOT NULL
        )
        """
    )
    db.commit()


def build_pgvector_database(
    chunks: Sequence[dict[str, Any]],
    config: "embedding.EmbeddingConfig",
    recreate: bool = False,
) -> int:
    with DBHandler() as db:
        ensure_schema(db, config.table_name, config.dimension, recreate=recreate)

        rows = db.execute(f"SELECT DISTINCT embedding_model FROM {config.table_name} LIMIT 2")
        stored_models = {row[0] for row in rows}
        if stored_models and stored_models != {config.model}:
            raise PgStoreError(
                f"Bảng '{config.table_name}' đang chứa vector model {stored_models}, "
                f"khác model đang dùng '{config.model}'. Chạy lại với --recreate."
            )

        total_batches = (len(chunks) + config.batch_size - 1) // config.batch_size
        for batch_index, batch in enumerate(embedding.batches(chunks, config.batch_size), start=1):
            documents = [str(chunk["content"]) for chunk in batch]
            vectors = embedding.embed_documents(documents, config=config)
            for chunk, vector in zip(batch, vectors):
                db.execute(
                    f"""
                    INSERT INTO {config.table_name} (id, content, metadata, embedding_model, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        str(chunk["id"]),
                        str(chunk["content"]),
                        json.dumps(chunk["metadata"], ensure_ascii=False),
                        config.model,
                        vector_literal(vector),
                    ),
                )
            db.commit()
            print(f"Batch {batch_index}/{total_batches}: stored {len(batch)} chunks")

        return db.execute(f"SELECT COUNT(*) FROM {config.table_name}")[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding chunks.json vào Postgres/pgvector.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--chunks-file", type=Path, default=None)
    parser.add_argument(
        "--recreate", action="store_true", help="Xoá bảng cũ trước khi embedding lại."
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    embedding.load_env_file(args.env_file)
    chunks_file = embedding.resolve_project_path(
        args.chunks_file or os.getenv("CHUNKS_FILE", ""), embedding.DEFAULT_CHUNKS_FILE
    )
    chunks = embedding.load_chunks(chunks_file)
    config = embedding.load_config(args.env_file)
    stored_count = build_pgvector_database(chunks, config, recreate=args.recreate)
    print(f"Hoàn tất: bảng '{config.table_name}' có {stored_count} records")


if __name__ == "__main__":
    try:
        main()
    except (
        PgStoreError,
        DbConfigError,
        embedding.ConfigurationError,
        embedding.EmbeddingError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
