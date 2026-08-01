"""Xử lý sau khi có câu trả lời thô từ LLM: ưu tiên nguồn, đính link, dọn câu trả lời.

Tách khỏi `service.py` để dùng chung giữa luồng service cũ và graph mới
(`src/chatbot/graph.py`) — một chỗ duy nhất, không nhân bản logic lần 3.
"""

import re

from .guardrail import _plain
from .types import Chunk

try:  # chạy dạng `src.chatbot` (test) hoặc `chatbot` với src trên sys.path (team)
    from ..tools.attach_source_link import ChunkRef, attach_source_link
    from ..tools.contact_support import contact_support
except ImportError:  # pragma: no cover - phụ thuộc cách nạp package
    from tools.attach_source_link import ChunkRef, attach_source_link  # type: ignore[no-redef]
    from tools.contact_support import contact_support  # type: ignore[no-redef]

DEFAULT_SUGGESTIONS = [
    "Lịch học thế nào?",
    "Điều kiện dự tuyển?",
    "Hồ sơ gồm những gì?",
    "Địa điểm học ở đâu?",
]

# Nguồn chính thức được ưu tiên khi độ phù hợp gần tương đương nguồn cộng đồng.
# Margin nhỏ giữ nguyên nguồn cộng đồng khi nó thực sự khớp tốt hơn rõ rệt.
OFFICIAL_SOURCE_MARGIN = 0.03


def _contact_markdown(reason: str, question: str) -> str:
    result = contact_support(reason, question)  # type: ignore[arg-type]
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
    """Đọc lại `source_type` đã tính 1 lần duy nhất ở `rag_bridge.payload_to_chunks()`
    (map `loai_nguon` -> `source_type`) — không tính lại ở đây để tránh 2 nơi định
    nghĩa cùng 1 mapping (từng là bug: `rag_bridge.SOURCE_TYPE_BY_LOAI_NGUON` và
    ternary riêng ở đây có thể lệch nhau khi thêm loại nguồn mới)."""
    return str(chunk.metadata.get("source_type") or "official_web")


def _prioritize_sources(chunks: list[Chunk]) -> list[Chunk]:
    """Ưu tiên nguồn chính thức nếu score nằm sát kết quả tốt nhất."""
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
            "muc_lon": str(chunk.metadata.get("muc_lon") or ""),
            "muc_nho": str(chunk.metadata.get("muc_nho") or ""),
            "source_link": attachment["source_url"],
            "source_type": attachment["source_type"],
            "label_hien_thi": attachment["label_hien_thi"],
            "warning": attachment.get("warning", ""),
        }
        key = (source["muc_lon"], source["muc_nho"], source["source_link"])
        if key not in seen2:
            seen2.add(key)
            sources.append(source)
    return sources


def _cited_chunks(answer: str, chunks: list[Chunk]) -> list[Chunk]:
    cited = [chunk for chunk in chunks if f"[{chunk.source}]" in answer]
    return cited or [chunks[0]]


def _clean_answer(answer: str) -> str:
    """Citation nằm trong JSON sources, không rò mã nội bộ vào nội dung user."""
    answer = re.sub(r"\[(?:source|chunk_[A-Za-z0-9_-]+)\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?im)^\s*Nguồn\s*:\s*$", "", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r" {2,}", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def _is_refusal_answer(answer: str) -> bool:
    """Nhận diện lời từ chối của LLM để không gắn nguồn RAG không liên quan."""

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
