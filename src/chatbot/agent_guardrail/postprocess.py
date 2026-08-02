import re

from src.rag.types import Chunk

from .guardrail import _plain

try:
    from ...tools.attach_source_link import ChunkRef, attach_source_link
    from ...tools.contact_support import contact_support
except ImportError:
    from tools.attach_source_link import ChunkRef, attach_source_link
    from tools.contact_support import contact_support

DEFAULT_SUGGESTIONS = [
    "Lịch học thế nào?",
    "Điều kiện dự tuyển?",
    "Hồ sơ gồm những gì?",
    "Địa điểm học ở đâu?",
]


OFFICIAL_SOURCE_MARGIN = 0.03


def _contact_markdown(reason: str, question: str) -> str:
    result = contact_support(reason, question)
    channels = result.contact_channels
    return (
        f"{result.message}\n\n"
        "**Kênh tuyển sinh chính thức**\n\n"
        f"- Hotline: {channels['hotline']}\n"
        f"- Tuyển sinh: {channels['tuyen_sinh']}\n"
        f"- Email: [{channels['email']}](mailto:{channels['email']})\n\n"
        f"**Câu hỏi soạn sẵn:** {result.suggested_question}"
    )


def _source_type(chunk: Chunk) -> str:
    return str(chunk.metadata.get("source_type") or "official_web")


def _prioritize_sources(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return []
    best_score = max(chunk.score for chunk in chunks)
    return sorted(
        chunks,
        key=lambda chunk: (
            0
            if _source_type(chunk) == "official_web"
            and best_score - chunk.score <= OFFICIAL_SOURCE_MARGIN
            else 1,
            -chunk.score,
        ),
    )


def _attachments(chunks: list[Chunk]) -> list[dict]:
    refs: list[ChunkRef] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        source_url = str(chunk.metadata.get("source_link", "")).strip()
        chunk_id = str(chunk.metadata.get("chunk_id", chunk.source))
        if not source_url or (chunk_id, source_url) in seen:
            continue
        seen.add((chunk_id, source_url))
        refs.append(ChunkRef(chunk_id, _source_type(chunk), source_url))
    attached = {
        item["chunk_id"]: item for item in attach_source_link(refs)
    }
    sources: list[dict] = []
    seen2: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        chunk_id = str(chunk.metadata.get("chunk_id", chunk.source))
        attachment = attached.get(chunk_id)
        if not attachment:
            continue
        source = {
            "section_major": str(chunk.metadata.get("section_major") or ""),
            "section_minor": str(chunk.metadata.get("section_minor") or ""),
            "source_link": attachment["source_url"],
            "source_type": attachment["source_type"],
            "label_hien_thi": attachment["label_hien_thi"],
            "warning": attachment.get("warning", ""),
        }
        key = (source["section_major"], source["section_minor"], source["source_link"])
        if key not in seen2:
            seen2.add(key)
            sources.append(source)
    return sources


def _cited_chunks(answer: str, chunks: list[Chunk]) -> list[Chunk]:
    cited = [chunk for chunk in chunks if f"[{chunk.source}]" in answer]
    return cited or [chunks[0]]


def _clean_answer(answer: str) -> str:
    answer = re.sub(r"\[(?:source|chunk_[A-Za-z0-9_-]+)\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?im)^\s*Nguồn\s*:\s*$", "", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r" {2,}", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def _is_refusal_answer(answer: str) -> bool:
    text = _plain(answer)
    refusal_patterns = (
        r"ngoai (chu de|pham vi)",
        r"khong the (ho tro|tra loi).{0,100}(cau hoi|noi dung)",
        r"chi (co the|ho tro).{0,80}(ai thuc chien|chuong trinh|tuyen sinh)",
    )
    return any(re.search(pattern, text) for pattern in refusal_patterns)


__all__ = [
    "DEFAULT_SUGGESTIONS",
    "OFFICIAL_SOURCE_MARGIN",
    "_attachments",
    "_cited_chunks",
    "_clean_answer",
    "_contact_markdown",
    "_is_refusal_answer",
    "_prioritize_sources",
    "_source_type",
]
