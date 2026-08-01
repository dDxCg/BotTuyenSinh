from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.db_client import DbConfigError

from . import chunking, embedding
from .pg_store import PgStoreError, build_pgvector_database

DEFAULT_ENV_FILE = embedding.DEFAULT_ENV_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk + embed + lưu pgvector, 1 lệnh duy nhất.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--web-dir", type=Path, default=chunking.DEFAULT_WEB_DIR)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument(
        "--strategy",
        choices=chunking.STRATEGIES,
        default="structure",
        help="Cách chunk: structure (mặc định), sentence, hoặc semantic",
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Xoá bảng cũ trước khi embedding lại."
    )
    return parser.parse_args()


def run(env_file: Path, web_dir: Path, max_chars: int, strategy: str, recreate: bool) -> None:
    embedding.load_env_file(env_file)

    chunks = chunking.build_chunks(web_dir=web_dir, max_chars=max_chars, strategy=strategy)
    chunks_file = embedding.resolve_project_path(
        os.getenv("CHUNKS_FILE", ""), embedding.DEFAULT_CHUNKS_FILE
    )
    chunking.save_chunks(chunks, chunks_file)
    print(f"Đã chunk {len(chunks)} chunks, ghi vào {chunks_file.resolve()}")

    config = embedding.load_config(env_file)
    stored_count = build_pgvector_database(chunks, config, recreate=recreate)
    print(f"Hoàn tất: bảng '{config.table_name}' có {stored_count} records")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    run(
        env_file=args.env_file,
        web_dir=args.web_dir,
        max_chars=args.max_chars,
        strategy=args.strategy,
        recreate=args.recreate,
    )


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
