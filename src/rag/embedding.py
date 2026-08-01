from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from src.llm_client import embedding_client


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_CHUNKS_FILE = RAG_DIR / "chunks.json"
DEFAULT_TABLE_NAME = "ai_thuc_chien_chunks"
MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


class EmbeddingError(RuntimeError):
    pass


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimension: int
    batch_size: int
    table_name: str


def _validate_vectors(
    result: list[list[float]], expected_count: int, dimension: int, model: str
) -> None:
    if len(result) != expected_count:
        raise EmbeddingError(f"Model trả {len(result)} vector, cần {expected_count} vector")
    invalid_dimensions = {len(vector) for vector in result if len(vector) != dimension}
    if invalid_dimensions:
        raise EmbeddingError(
            f"Vector có số chiều {sorted(invalid_dimensions)}; cần {dimension} chiều cho {model}"
        )


def embed_texts(
    texts: Sequence[str],
    *,
    config: "EmbeddingConfig | None" = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    config = config or load_config()
    batch_size = config.batch_size if batch_size is None else batch_size
    if batch_size < 1:
        raise EmbeddingError("batch_size phải lớn hơn 0")

    cleaned = [str(text).strip() for text in texts]
    for text in cleaned:
        if not text:
            raise EmbeddingError("Nội dung embedding không được để trống")

    client = embedding_client()
    result: list[list[float]] = []
    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start : start + batch_size]
        try:
            response = client.embeddings.create(model=config.model, input=batch)
        except Exception as exc:
            raise EmbeddingError(f"Không tạo được embedding qua OpenAI API: {exc}") from exc
        result.extend([list(item.embedding) for item in response.data])
    _validate_vectors(result, len(cleaned), config.dimension, config.model)
    return result


def embed_documents(
    documents: Sequence[str], *, config: "EmbeddingConfig | None" = None
) -> list[list[float]]:
    return embed_texts(documents, config=config)


def embed_query(question: str, *, config: "EmbeddingConfig | None" = None) -> list[float]:
    return embed_texts([question], config=config)[0]


def load_env_file(env_file: Path = DEFAULT_ENV_FILE) -> None:

    load_dotenv(env_file, override=False)


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} phải là số nguyên") from exc
    if value < 1:
        raise ConfigurationError(f"{name} phải lớn hơn 0")
    return value


def load_config(env_file: Path = DEFAULT_ENV_FILE) -> EmbeddingConfig:
    load_env_file(env_file)
    model = os.getenv("EMBEDDING_MODEL", MODEL_ID).strip() or MODEL_ID
    dimension = positive_int_env("EMBEDDING_DIMENSION", EMBEDDING_DIMENSION)
    return EmbeddingConfig(
        model=model,
        dimension=dimension,
        batch_size=positive_int_env("EMBEDDING_BATCH_SIZE", 8),
        table_name=os.getenv("PG_TABLE", DEFAULT_TABLE_NAME).strip() or DEFAULT_TABLE_NAME,
    )


def resolve_project_path(value: str | Path, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_chunks(chunks_file: Path) -> list[dict[str, Any]]:

    if not chunks_file.exists():
        raise FileNotFoundError(f"Không tìm thấy chunks JSON: {chunks_file}")
    try:
        payload = json.loads(chunks_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON không hợp lệ: {chunks_file}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise ValueError("chunks.json phải có object gốc và field 'chunks' dạng list")

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_chunk in enumerate(payload["chunks"]):
        if not isinstance(raw_chunk, dict):
            raise ValueError(f"Chunk {index} phải là object")
        chunk_id = raw_chunk.get("id")
        content = raw_chunk.get("content")
        metadata = raw_chunk.get("metadata")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk {index} thiếu id")
        if chunk_id in seen_ids:
            raise ValueError(f"Chunk id bị trùng: {chunk_id}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Chunk {chunk_id} thiếu content")
        if not isinstance(metadata, dict):
            raise ValueError(f"Chunk {chunk_id} thiếu metadata")
        seen_ids.add(chunk_id)
        chunks.append(raw_chunk)
    if not chunks:
        raise ValueError("chunks.json không có chunk nào")
    return chunks


def batches(items: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra chunks.json.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--chunks-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    load_env_file(args.env_file)
    chunks_file = resolve_project_path(
        args.chunks_file or os.getenv("CHUNKS_FILE", ""), DEFAULT_CHUNKS_FILE
    )
    chunks = load_chunks(chunks_file)
    print(f"Hợp lệ: {len(chunks)} chunks trong {chunks_file.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigurationError, EmbeddingError, FileNotFoundError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
