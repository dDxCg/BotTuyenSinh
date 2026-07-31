"""Tìm top chunks bằng cosine similarity và E5-large local."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from . import embedding
except ImportError:  # Cho phép chạy trực tiếp: python src/rag/retrieval.py
    import embedding  # type: ignore[no-redef]


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_CHROMA_DIR = RAG_DIR / "chroma_db"
DEFAULT_TOP_K = 5


class RetrievalError(RuntimeError):
    """Dữ liệu hoặc cấu hình retrieval không hợp lệ."""


def clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def query_embedding(question: str) -> list[float]:
    """Embedding câu hỏi bằng model local duy nhất của hệ thống."""

    prefix = os.getenv(
        "EMBEDDING_QUERY_PREFIX", embedding.DEFAULT_QUERY_PREFIX
    ).strip()
    return embedding.embed_query(question, prefix=prefix)


def open_existing_collection(
    chroma_dir: Path,
    collection_name: str,
    expected_model: str,
):
    """Mở collection đã embedding; không tự tạo collection rỗng."""

    if not chroma_dir.exists():
        raise RetrievalError(f"Không tìm thấy ChromaDB: {chroma_dir}")

    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except NotFoundError as exc:
        raise RetrievalError(
            f"Không tìm thấy collection '{collection_name}' trong {chroma_dir}"
        ) from exc

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != expected_model:
        raise RetrievalError(
            f"Collection dùng model '{stored_model}', nhưng RAG local cố định dùng "
            f"'{expected_model}'"
        )
    hnsw_config = (collection.configuration or {}).get("hnsw") or {}
    if hnsw_config.get("space") != "cosine":
        raise RetrievalError("Collection không được cấu hình cosine distance")
    if collection.count() == 0:
        raise RetrievalError("Collection chưa có record")
    return collection


def format_results(
    question: str,
    model: str,
    raw_results: dict[str, Any],
) -> dict[str, Any]:
    ids: Sequence[str] = (raw_results.get("ids") or [[]])[0]
    documents: Sequence[str] = (raw_results.get("documents") or [[]])[0]
    metadatas: Sequence[dict[str, Any] | None] = (
        raw_results.get("metadatas") or [[]]
    )[0]
    distances: Sequence[float] = (raw_results.get("distances") or [[]])[0]
    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise RetrievalError("Chroma trả số lượng field không đồng nhất")

    results: list[dict[str, Any]] = []
    for rank, (chunk_id, content, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        results.append(
            {
                "rank": rank,
                "id": chunk_id,
                "cosine_similarity": round(1.0 - float(distance), 6),
                "content": content,
                "metadata": clean_metadata(metadata),
            }
        )
    return {
        "question": question,
        "embedding_model": model,
        "returned_results": len(results),
        "results": results,
    }


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    env_file: Path = DEFAULT_ENV_FILE,
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Embedding local câu hỏi và lấy top_k chunks theo cosine similarity."""

    question = question.strip()
    if not question:
        raise RetrievalError("Câu hỏi không được để trống")
    if top_k < 1:
        raise RetrievalError("top_k phải lớn hơn 0")

    embedding.load_env_file(env_file)
    config = embedding.load_config(env_file)
    resolved_chroma_dir = embedding.resolve_project_path(
        chroma_dir or os.getenv("CHROMA_DIR", ""), DEFAULT_CHROMA_DIR
    )
    resolved_collection_name = (
        collection_name
        or os.getenv("CHROMA_COLLECTION", "").strip()
        or config.collection_name
    )
    collection = open_existing_collection(
        chroma_dir=resolved_chroma_dir,
        collection_name=resolved_collection_name,
        expected_model=config.model,
    )

    result_count = min(top_k, collection.count())
    print("Đang tạo embedding local cho câu hỏi...", file=sys.stderr, flush=True)
    vector = query_embedding(question)
    print("Đang tìm chunks gần nhất...", file=sys.stderr, flush=True)
    raw_results = collection.query(
        query_embeddings=[vector],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    return format_results(question, config.model, raw_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tìm top chunks bằng multilingual-e5-large local."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Câu hỏi người dùng. Nếu bỏ trống, nhập trực tiếp trên terminal.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="In JSON một dòng thay vì định dạng dễ đọc.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    question = args.question if args.question is not None else input("Nhập câu hỏi: ")
    payload = retrieve(
        question=question,
        top_k=args.top_k,
        env_file=args.env_file,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (
        RetrievalError,
        embedding.ConfigurationError,
        embedding.LocalEmbeddingError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
