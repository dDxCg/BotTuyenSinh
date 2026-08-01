from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class DbConfigError(RuntimeError):
    pass


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


class DBHandler:
    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or connection_string()
        self._conn: Any = None

    def connect(self) -> "DBHandler":
        import psycopg

        self._conn = psycopg.connect(self._dsn)
        return self

    def execute(self, query: str, params: Sequence[Any] | None = None) -> list[tuple] | None:
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description is None:
                return None
            return cur.fetchall()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> "DBHandler":
        return self.connect()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()


def test_connection() -> bool:

    try:
        with DBHandler() as db:
            row = db.execute("SELECT 1")
        print(f"OK: kết nối Postgres thành công, SELECT 1 -> {row[0][0]}")
        return True
    except Exception as exc:
        print(f"Lỗi kết nối Postgres: {type(exc).__name__}: {exc}")
        return False


__all__ = [
    "DBHandler",
    "DbConfigError",
    "connect",
    "connection_string",
    "test_connection",
    "vector_literal",
]


if __name__ == "__main__":
    raise SystemExit(0 if test_connection() else 1)
