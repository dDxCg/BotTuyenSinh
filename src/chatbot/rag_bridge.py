import re
import unicodedata
from collections import OrderedDict
from typing import Any, Callable

from src.rag.settings import RagSettings

from .types import Chunk


NO_GROUNDING_THRESHOLD = RagSettings.from_env().no_grounding_threshold


SOURCE_TYPE_BY_KIND = {
    "web": "official_web",
}


def _load_retrieve() -> Callable[..., dict[str, Any]]:

    from src.rag.pg_retrieval import retrieve

    return retrieve


def payload_to_chunks(payload: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for item in payload.get("results", []):
        metadata = dict(item.get("metadata") or {})
        source_kind = metadata.get("source_kind", "")
        metadata["chunk_id"] = item.get("id", "")
        metadata["source_type"] = SOURCE_TYPE_BY_KIND.get(source_kind, source_kind)
        chunks.append(
            Chunk(
                text=item.get("content", ""),
                source=metadata.get("document_name") or metadata.get("source_file", "unknown"),
                score=float(item.get("cosine_similarity", 0.0)),
                metadata=metadata,
            )
        )
    return chunks


def cache_key(query: str, k: int) -> tuple[str, int]:
    text = unicodedata.normalize("NFC", query).casefold().strip()
    return re.sub(r"[\s\W_]+", " ", text).strip(), k


class PgVectorRetriever:

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

        self.chunk_by_id.update({c.metadata["chunk_id"]: c for c in chunks if c.metadata.get("chunk_id")})
        return chunks

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunk_by_id.get(chunk_id)

    def best_score(self) -> float:
        results = self.last_payload.get("results") or []
        return float(results[0].get("cosine_similarity", 0.0)) if results else 0.0

    def has_grounding(self) -> bool:
        return self.best_score() >= NO_GROUNDING_THRESHOLD
