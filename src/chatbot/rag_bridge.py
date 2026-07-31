"""Nối `src/rag/retrieval.py` (ChromaDB) vào contract `Retriever` của chatbot.

`retrieval.retrieve()` trả JSON theo docs/rag-system.md §8.2. Lớp này đổi nó
thành `Chunk`, giữ nguyên `chunk_id` + `source_link` + `loai_nguon` trong
metadata để tool `attach_source_link` truy ngược được nguồn.
"""

import re
import unicodedata
from collections import OrderedDict
from typing import Any, Callable

from .types import Chunk

# Ngưỡng chốt trong docs/design-agent-tools.md §4: dưới mức này coi là không có căn cứ.
NO_GROUNDING_THRESHOLD = 0.7

# docs/rag-system.md §5.1 dùng `loai_nguon`; tool dùng `source_type`.
SOURCE_TYPE_BY_LOAI_NGUON = {
    "facebook": "community_facebook",
    "web": "official_web",
}


def _load_retrieve() -> Callable[..., dict[str, Any]]:
    """Import muộn để chỉ nạp Chroma và model local khi thật sự retrieval."""

    from src.rag.retrieval import retrieve

    return retrieve


def payload_to_chunks(payload: dict[str, Any]) -> list[Chunk]:
    """Đổi JSON retrieval thành Chunk, giữ đủ metadata để trích nguồn."""
    chunks: list[Chunk] = []
    for item in payload.get("results", []):
        metadata = dict(item.get("metadata") or {})
        loai_nguon = metadata.get("loai_nguon", "")
        metadata["chunk_id"] = item.get("id", "")
        metadata["source_type"] = SOURCE_TYPE_BY_LOAI_NGUON.get(loai_nguon, loai_nguon)
        chunks.append(
            Chunk(
                text=item.get("content", ""),
                source=metadata.get("ten_tai_lieu") or metadata.get("source_file", "unknown"),
                score=float(item.get("cosine_similarity", 0.0)),
                metadata=metadata,
            )
        )
    return chunks


def cache_key(query: str, k: int) -> tuple[str, int]:
    """Chuẩn hoá query để câu hỏi chỉ khác dấu câu/hoa thường vẫn dùng chung vector."""
    text = unicodedata.normalize("NFC", query).casefold().strip()
    return re.sub(r"[\s\W_]+", " ", text).strip(), k


class ChromaRetriever:
    """Retriever thật, đọc ChromaDB đã embedding sẵn.

    Nhớ kết quả lượt gần nhất (`chunk_by_id`) để `attach_source_link` đổi
    `chunk_ids` do model đưa thành nguồn — model chỉ biết id, không biết URL.

    Có cache LRU để tránh encode lại câu hỏi lặp và giảm tải CPU/GPU local.
    """

    def __init__(
        self,
        retrieve_fn: Callable[..., dict[str, Any]] | None = None,
        cache_size: int = 128,
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self._cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_payload: dict[str, Any] = {}
        self.last_chunks: list[Chunk] = []
        self.chunk_by_id: dict[str, Chunk] = {}

    def _fetch(self, query: str, k: int) -> dict[str, Any]:
        key = cache_key(query, k)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.cache_misses += 1
        if self._retrieve_fn is None:
            self._retrieve_fn = _load_retrieve()
        payload = self._retrieve_fn(query, top_k=k)
        if self._cache_size > 0:
            self._cache[key] = payload
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return payload

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        self.last_payload = self._fetch(query, k)
        chunks = payload_to_chunks(self.last_payload)
        self.last_chunks = chunks
        # Tích luỹ qua nhiều lượt: model có thể trích chunk từ lần tìm trước.
        self.chunk_by_id.update({c.metadata["chunk_id"]: c for c in chunks if c.metadata.get("chunk_id")})
        return chunks

    def best_score(self) -> float:
        results = self.last_payload.get("results") or []
        return float(results[0].get("cosine_similarity", 0.0)) if results else 0.0

    def has_grounding(self) -> bool:
        """Đủ căn cứ để agent tự trả lời, theo ngưỡng đã chốt."""
        return self.best_score() >= NO_GROUNDING_THRESHOLD
