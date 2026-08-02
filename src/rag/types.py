from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str = "unknown"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...


class NullRetriever:
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return []

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return None


__all__ = ["Chunk", "NullRetriever", "Retriever"]
