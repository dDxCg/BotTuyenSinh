"""Chunk theo semantic — dùng embedding model hiện có để đo độ tương đồng giữa
câu liền kề, cắt chunk tại điểm ngữ nghĩa đổi mạnh (embedding-based, không cần LLM).
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    from . import embedding
    from .chunking import Section, pack_units_to_chunks, section_prefix, split_by_words, split_into_sentences
except ImportError:  # pragma: no cover - chạy trực tiếp không qua package
    import embedding  # type: ignore[no-redef]
    from chunking import (  # type: ignore[no-redef]
        Section,
        pack_units_to_chunks,
        section_prefix,
        split_by_words,
        split_into_sentences,
    )

DEFAULT_PERCENTILE = 95.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile_value(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (percentile / 100) * (len(ordered) - 1)
    low, high = int(math.floor(rank)), int(math.ceil(rank))
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def semantic_breakpoints(
    sentences: list[str],
    config: "embedding.EmbeddingConfig | None" = None,
    percentile: float = DEFAULT_PERCENTILE,
) -> list[int]:
    """Trả danh sách index (1-based, vị trí cắt trước câu đó) nơi độ tương đồng
    ngữ nghĩa với câu liền trước giảm mạnh (percentile-based breakpoint)."""

    if len(sentences) < 2:
        return []
    vectors = embedding.embed_documents(sentences, config=config)
    distances = [
        1 - cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]
    cutoff = percentile_value(distances, percentile)
    return [i + 1 for i, d in enumerate(distances) if d > cutoff]


def group_by_breakpoints(sentences: list[str], breakpoints: list[int]) -> list[str]:
    groups: list[str] = []
    start = 0
    for bp in sorted(breakpoints):
        if bp > start:
            groups.append(" ".join(sentences[start:bp]))
            start = bp
    if start < len(sentences):
        groups.append(" ".join(sentences[start:]))
    return groups


def split_section_semantic(
    section: Section,
    max_chars: int,
    config: "embedding.EmbeddingConfig | None" = None,
    percentile: float = DEFAULT_PERCENTILE,
) -> list[str]:
    prefix = section_prefix(section)
    separator = "\n\n" if prefix else ""
    body_limit = max(200, max_chars - len(prefix) - len(separator))

    text = "\n".join(line for line in section.lines if line.strip())
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    breakpoints = semantic_breakpoints(sentences, config=config, percentile=percentile)
    groups = group_by_breakpoints(sentences, breakpoints)

    units: list[str] = []
    for group in groups:
        if len(group) <= body_limit:
            units.append(group)
        else:
            units.extend(split_by_words(group, body_limit))
    return pack_units_to_chunks(units, prefix, separator, body_limit)


__all__ = [
    "cosine_similarity",
    "group_by_breakpoints",
    "percentile_value",
    "semantic_breakpoints",
    "split_section_semantic",
]
