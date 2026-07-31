"""Tạo embedding local cho ``chunks.json`` và lưu vào ChromaDB.

Toàn bộ vector được sinh bởi ``intfloat/multilingual-e5-large`` từ model đã
tải vào workspace. Module này không có client HTTP và không đọc API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_CHUNKS_FILE = RAG_DIR / "chunks.json"
DEFAULT_CHROMA_DIR = RAG_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "ai_thuc_chien_chunks"
MODEL_ID = "intfloat/multilingual-e5-large"
EMBEDDING_DIMENSION = 1024
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "intfloat-multilingual-e5-large"
DEFAULT_QUERY_PREFIX = "query:"
DEFAULT_DOCUMENT_PREFIX = "passage:"
_ENCODE_LOCK = threading.Lock()


class LocalEmbeddingError(RuntimeError):
    """Model local thiếu hoặc không tạo được embedding hợp lệ."""


def resolve_model_path() -> Path:
    raw = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "").strip()
    if not raw:
        return DEFAULT_MODEL_PATH
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def load_local_model() -> Any:
    """Nạp một model từ ổ đĩa; không tải model qua mạng khi RAG chạy."""

    model_path = resolve_model_path()
    if not model_path.is_dir():
        raise LocalEmbeddingError(
            f"Chưa có model local tại {model_path}. "
            "Chạy `python -m src.rag.download_model` trước khi dùng RAG."
        )
    device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu").strip() or "cpu"
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            str(model_path),
            device=device,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise LocalEmbeddingError(f"Không nạp được model local: {exc}") from exc


def _prefixed(text: str, prefix: str) -> str:
    clean_text = str(text).strip()
    if not clean_text:
        raise LocalEmbeddingError("Nội dung embedding không được để trống")
    return f"{prefix.rstrip()} {clean_text}" if prefix else clean_text


def embed_texts(
    texts: Sequence[str],
    *,
    prefix: str,
    batch_size: int = 8,
) -> list[list[float]]:
    """Encode batch bằng E5 local và trả vector cosine đã chuẩn hóa."""

    if batch_size < 1:
        raise LocalEmbeddingError("batch_size phải lớn hơn 0")
    if not texts:
        return []
    inputs = [_prefixed(text, prefix) for text in texts]
    model = load_local_model()
    try:
        with _ENCODE_LOCK:
            vectors = model.encode(
                inputs,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
    except Exception as exc:
        raise LocalEmbeddingError(f"Không tạo được embedding local: {exc}") from exc

    result = [[float(value) for value in vector.tolist()] for vector in vectors]
    if len(result) != len(inputs):
        raise LocalEmbeddingError(
            f"Model trả {len(result)} vector, cần {len(inputs)} vector"
        )
    invalid_dimensions = {
        len(vector) for vector in result if len(vector) != EMBEDDING_DIMENSION
    }
    if invalid_dimensions:
        raise LocalEmbeddingError(
            f"Vector local có số chiều {sorted(invalid_dimensions)}; "
            f"cần {EMBEDDING_DIMENSION} chiều cho {MODEL_ID}"
        )
    return result


def embed_documents(
    documents: Sequence[str],
    *,
    prefix: str = DEFAULT_DOCUMENT_PREFIX,
    batch_size: int = 8,
) -> list[list[float]]:
    return embed_texts(documents, prefix=prefix, batch_size=batch_size)


def embed_query(question: str, prefix: str = DEFAULT_QUERY_PREFIX) -> list[float]:
    return embed_texts([question], prefix=prefix, batch_size=1)[0]


def warmup_local_model() -> None:
    """Nạp weights một lần lúc app khởi động."""

    load_local_model()


class ConfigurationError(ValueError):
    """Cấu hình local thiếu hoặc không hợp lệ."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    batch_size: int
    document_prefix: str
    collection_name: str


def load_env_file(env_file: Path = DEFAULT_ENV_FILE) -> None:
    """Đọc `.env` nếu có, không ghi đè biến môi trường đã được thiết lập."""

    if not env_file.exists():
        return
    for line_number, raw_line in enumerate(
        env_file.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(
                f"Dòng {line_number} trong {env_file} không có dấu '='"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigurationError(
                f"Dòng {line_number} trong {env_file} thiếu tên biến"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


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
    return EmbeddingConfig(
        model=MODEL_ID,
        batch_size=positive_int_env("EMBEDDING_BATCH_SIZE", 8),
        document_prefix=os.getenv(
            "EMBEDDING_DOCUMENT_PREFIX", DEFAULT_DOCUMENT_PREFIX
        ).strip(),
        collection_name=os.getenv(
            "CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME
        ).strip()
        or DEFAULT_COLLECTION_NAME,
    )


def resolve_project_path(value: str | Path, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    """Đọc và kiểm tra schema do chunking.py tạo."""

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


def batches(
    items: Sequence[dict[str, Any]], size: int
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def chroma_metadata(metadata: dict[str, Any], model: str) -> dict[str, Any]:
    """Đổi metadata JSON thành các kiểu Chroma hỗ trợ."""

    result: dict[str, Any] = {"embedding_model": model}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            result[str(key)] = value
        else:
            result[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return result


def open_collection(chroma_dir: Path, config: EmbeddingConfig, recreate: bool):
    """Mở collection persistent và chặn vector từ model khác."""

    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    if recreate:
        try:
            client.delete_collection(config.collection_name)
        except NotFoundError:
            pass
    try:
        collection = client.get_collection(
            name=config.collection_name,
            embedding_function=None,
        )
    except NotFoundError:
        collection = client.create_collection(
            name=config.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": config.model},
            embedding_function=None,
        )

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != config.model:
        raise ConfigurationError(
            f"Collection dùng model '{stored_model}', nhưng RAG local cố định dùng "
            f"'{config.model}'. Chạy với --recreate hoặc đổi CHROMA_COLLECTION."
        )
    return collection


def build_chroma_database(
    chunks: Sequence[dict[str, Any]],
    chroma_dir: Path,
    config: EmbeddingConfig,
    recreate: bool = False,
) -> int:
    """Encode document bằng E5 local rồi upsert vào ChromaDB."""

    collection = open_collection(chroma_dir, config, recreate=recreate)
    total_batches = (len(chunks) + config.batch_size - 1) // config.batch_size
    for batch_index, batch in enumerate(batches(chunks, config.batch_size), start=1):
        documents = [str(chunk["content"]) for chunk in batch]
        vectors = embed_documents(
            documents,
            prefix=config.document_prefix,
            batch_size=config.batch_size,
        )
        collection.upsert(
            ids=[str(chunk["id"]) for chunk in batch],
            documents=documents,
            embeddings=vectors,
            metadatas=[
                chroma_metadata(chunk["metadata"], config.model) for chunk in batch
            ],
        )
        print(f"Batch {batch_index}/{total_batches}: stored {len(batch)} chunks")
    return collection.count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding local chunks.json và lưu vào ChromaDB."
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--chunks-file", type=Path, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa collection cũ cùng tên trước khi embedding lại.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Chỉ kiểm tra chunks.json; không nạp model và không tạo ChromaDB.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    load_env_file(args.env_file)
    chunks_file = resolve_project_path(
        args.chunks_file or os.getenv("CHUNKS_FILE", ""), DEFAULT_CHUNKS_FILE
    )
    chroma_dir = resolve_project_path(
        args.chroma_dir or os.getenv("CHROMA_DIR", ""), DEFAULT_CHROMA_DIR
    )
    chunks = load_chunks(chunks_file)
    if args.validate_only:
        print(f"Hợp lệ: {len(chunks)} chunks trong {chunks_file.resolve()}")
        return

    config = load_config(args.env_file)
    stored_count = build_chroma_database(
        chunks=chunks,
        chroma_dir=chroma_dir,
        config=config,
        recreate=args.recreate,
    )
    print(
        f"Hoàn tất: collection '{config.collection_name}' có {stored_count} records "
        f"tại {chroma_dir.resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ConfigurationError,
        LocalEmbeddingError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
