"""Service nối chatbot, RAG và hai tool production cho web demo."""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from typing import Callable

from src.chatbot.chatbot import Chatbot
from src.chatbot.config import Settings
from src.chatbot.rag_bridge import ChromaRetriever
from src.chatbot.types import Chunk, Retriever
from src.tools.attach_source_link import ChunkRef, attach_source_link
from src.tools.contact_support import NO_GROUNDING_THRESHOLD, contact_support


DEFAULT_SUGGESTIONS = [
    "Lịch học thế nào?",
    "Điều kiện dự tuyển?",
    "Hồ sơ gồm những gì?",
    "Địa điểm học ở đâu?",
]

# Nguồn chính thức được ưu tiên khi độ phù hợp gần tương đương nguồn cộng đồng.
# Margin nhỏ giữ nguyên nguồn cộng đồng khi nó thực sự khớp tốt hơn rõ rệt.
OFFICIAL_SOURCE_MARGIN = 0.03

# Gate bảo thủ: chỉ chặn câu chắc chắn không liên quan. Mọi câu còn mơ hồ hoặc
# có khả năng liên quan đều phải đi qua retrieval; RAG/LLM quyết định có đủ căn
# cứ để trả lời hay không.
UNRELATED_PATTERNS = (
    r"^\s*\d+(?:\s*[+\-*/x:]\s*\d+)+\s*\??$",
    r"con ga.{0,30}qua trung|qua trung.{0,30}con ga",
    r"thu do.{0,20}(phap|duc|anh|my|nhat)",
    r"thoi tiet|bong da|nau an|ke chuyen|tu vi|xem boi",
    r"sua (dieu hoa|xe|dien|nuoc|may)|tho (dien|nuoc|sua xe)",
)

UNRELATED_REPLY = (
    "Cảm ơn bạn đã đặt câu hỏi! Mình là trợ lý tư vấn Chương trình AI Thực Chiến "
    "của VinUni, nên mình xin phép không trả lời các nội dung ngoài phạm vi này để "
    "tránh cung cấp thông tin không phù hợp.\n\n"
    "Mình có thể hỗ trợ bạn về điều kiện dự tuyển, hồ sơ, lịch học, học phí, "
    "nội dung chương trình hoặc địa điểm học."
)


@dataclass(frozen=True)
class DemoReply:
    answer: str
    sources: list[dict]
    suggestions: list[str]
    grounded: bool
    top_score: float | None
    path: str


def _plain(value: str) -> str:
    text = value.lower().replace("đ", "d")
    import unicodedata

    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def classify_restricted(question: str) -> str | None:
    """Chỉ chặn ranh giới chính sách và câu chắc chắn không liên quan."""
    text = _plain(question)
    personal = (
        r"(trang thai|ket qua|diem).{0,20}(ho so|cua (toi|em|minh))",
        r"(ma ho so|email).{0,30}(tra|kiem|xem)",
    )
    out_of_scope = (
        r"(co nen|nen hay khong|co dang).{0,20}(nop|dang ky|hoc)",
        r"(co dau|se dau|kha nang dau)",
        r"(cam ket dau ra|thu nhap|muc luong|luong sau)",
    )
    if any(re.search(pattern, text) for pattern in personal):
        return "personal_data_request"
    if any(re.search(pattern, text) for pattern in out_of_scope):
        return "out_of_scope"
    if any(re.search(pattern, text) for pattern in UNRELATED_PATTERNS):
        return "unrelated"
    return None


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
    return "community_facebook" if chunk.metadata.get("loai_nguon") == "facebook" else "official_web"


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
    seen: set[tuple[str, str, str]] = set()
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
        if key not in seen:
            seen.add(key)
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


class DemoService:
    def __init__(
        self,
        retriever: Retriever | None = None,
        settings: Settings | None = None,
        bot_factory: Callable[[], Chatbot] | None = None,
    ) -> None:
        self.retriever = retriever or ChromaRetriever()
        self.settings = settings
        self.bot_factory = bot_factory
        self._sessions: dict[str, Chatbot] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _session(self, session_id: str) -> tuple[Chatbot, threading.Lock]:
        with self._guard:
            bot = self._sessions.get(session_id)
            if bot is None:
                bot = self.bot_factory() if self.bot_factory else Chatbot(
                    settings=self.settings,
                    retriever=self.retriever,
                    top_k=5,
                )
                self._sessions[session_id] = bot
                self._locks[session_id] = threading.Lock()
            return bot, self._locks[session_id]

    def chat(self, session_id: str, question: str) -> DemoReply:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống")

        bot, lock = self._session(session_id)
        with lock:
            restricted_reason = classify_restricted(question)
            if restricted_reason:
                if restricted_reason == "unrelated":
                    bot.remember(question, UNRELATED_REPLY)
                    return DemoReply(
                        UNRELATED_REPLY,
                        [],
                        DEFAULT_SUGGESTIONS,
                        False,
                        None,
                        "out_of_scope",
                    )
                answer = _contact_markdown(restricted_reason, question)
                bot.remember(question, answer)
                return DemoReply(
                    answer, [], DEFAULT_SUGGESTIONS, False, None, "contact_support"
                )

            retrieved = self.retriever.retrieve(question, k=5)
            top_score = max((chunk.score for chunk in retrieved), default=None)
            chunks = _prioritize_sources(retrieved)
            if not chunks or top_score is None or top_score < NO_GROUNDING_THRESHOLD:
                answer = _contact_markdown("no_grounding", question)
                bot.remember(question, answer)
                return DemoReply(
                    answer, [], DEFAULT_SUGGESTIONS, False, top_score, "contact_support"
                )

            answer = bot.chat_with_retrieved(question, chunks)
            answer = _clean_answer(answer)
            if _is_refusal_answer(answer):
                return DemoReply(
                    answer=answer,
                    sources=[],
                    suggestions=DEFAULT_SUGGESTIONS,
                    grounded=False,
                    top_score=top_score,
                    path="out_of_scope",
                )
            cited_chunks = _cited_chunks(answer, chunks)
            return DemoReply(
                answer=answer,
                sources=_attachments(cited_chunks),
                suggestions=DEFAULT_SUGGESTIONS,
                grounded=True,
                top_score=top_score,
                path="rag+attach_source_link",
            )

    def reset(self, session_id: str) -> None:
        with self._guard:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)


def reply_dict(reply: DemoReply) -> dict:
    return asdict(reply)


__all__ = ["DemoReply", "DemoService", "classify_restricted", "reply_dict"]
