"""1 nơi duy nhất kết nối Postgres (Neon/pgvector) — pg_store.py và pg_retrieval.py
dùng chung, tránh mỗi module tự đọc DATABASE_URL/tự connect riêng.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class DbConfigError(RuntimeError):
    """Thiếu DATABASE_URL trong .env."""


def connection_string() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DbConfigError("Thiếu DATABASE_URL trong .env (Postgres/Neon + pgvector)")
    return url


def connect() -> Any:
    import psycopg

    return psycopg.connect(connection_string())


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


__all__ = ["DbConfigError", "connect", "connection_string", "vector_literal"]
