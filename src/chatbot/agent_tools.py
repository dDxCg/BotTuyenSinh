"""Bọc 2 tool thật ở `src/tools/` thành `Tool` cho vòng ReAct.

Khe hở phải vá ở đây: `attach_source_link()` nhận `ChunkRef` (có sẵn source_url),
nhưng schema trong docs/design-agent-tools.md §3 hứa với model là
`chunk_ids: list[str]`. Model chỉ biết id, nên tầng này tra ngược id -> nguồn từ
kết quả retrieval gần nhất.

Chunk đến từ RAG đổ sẵn vào system prompt (`prefetch_rag`), không qua tool.
"""

from .rag_bridge import ChromaRetriever
from .types import Tool, ToolRegistry

try:  # chạy dạng `src.chatbot` (test) hoặc `chatbot` với src trên sys.path (team)
    from ..tools.attach_source_link import ChunkRef, attach_source_link
    from ..tools.contact_support import contact_support
except ImportError:  # pragma: no cover - phụ thuộc cách nạp package
    from tools.attach_source_link import ChunkRef, attach_source_link  # type: ignore[no-redef]
    from tools.contact_support import contact_support  # type: ignore[no-redef]


def make_attach_source_link(retriever: ChromaRetriever) -> Tool:
    def attach_source_link_tool(chunk_ids: list[str] | str) -> str:
        if isinstance(chunk_ids, str):  # model hay truyền một id trần
            chunk_ids = [chunk_ids]

        refs: list[ChunkRef] = []
        unknown: list[str] = []
        for chunk_id in chunk_ids:
            chunk = retriever.chunk_by_id.get(chunk_id)
            source_url = (chunk.metadata.get("source_link") or "") if chunk else ""
            source_type = (chunk.metadata.get("source_type") or "") if chunk else ""
            if not chunk or not source_url or not source_type:
                unknown.append(chunk_id)
                continue
            refs.append(ChunkRef(chunk_id=chunk_id, source_type=source_type, source_url=source_url))

        if not refs:
            return (
                f"Lỗi: không tra được nguồn cho id {unknown}. Chỉ dùng id xuất hiện trong "
                "mục Ngữ cảnh truy xuất."
            )

        lines = []
        for attachment in attach_source_link(refs):
            line = f"- {attachment['label_hien_thi']}: {attachment['source_url']}"
            if "warning" in attachment:
                line += f"\n  ⚠ {attachment['warning']}"
            lines.append(line)
        if unknown:
            lines.append(f"(bỏ qua id không tra được: {unknown})")
        return "\n".join(lines)

    return Tool(
        name="attach_source_link",
        description=(
            "Lấy link nguồn của các chunk đã dùng để trả lời. Gọi sau khi đã chọn được dữ kiện, "
            "trước khi chốt Final Answer. Chunk từ Facebook luôn kèm cảnh báo không phải nguồn chính thức."
        ),
        signature="attach_source_link(chunk_ids: list[str]) -> str",
        func=attach_source_link_tool,
    )


def make_contact_support() -> Tool:
    def contact_support_tool(
        reason: str,
        user_question: str,
        partial_context: str | None = None,
    ) -> str:
        result = contact_support(reason, user_question, partial_context)  # type: ignore[arg-type]
        channels = "\n".join(f"- {key}: {value}" for key, value in result.contact_channels.items())
        parts = [result.message, "Kênh liên hệ tuyển sinh:", channels]
        if result.conflicting_facts:
            facts = "\n".join(f"- {fact}" for fact in result.conflicting_facts)
            parts.insert(1, f"Hai dữ kiện đang khác nhau (không tự chọn giúp):\n{facts}")
        parts.append(f"Câu hỏi soạn sẵn để gửi nhân viên: \"{result.suggested_question}\"")
        parts.append(">>> Đưa nguyên nội dung này cho người dùng trong Final Answer.")
        return "\n\n".join(parts)

    return Tool(
        name="contact_support",
        description=(
            "Chuyển câu hỏi cho nhân viên tuyển sinh khi không đủ căn cứ hoặc câu hỏi ngoài phạm vi. "
            "reason nhận: no_grounding | out_of_scope | conflicting_sources | personal_data_request."
        ),
        signature=(
            "contact_support(reason: str, user_question: str, partial_context: str = None) -> str"
        ),
        func=contact_support_tool,
    )


def build_registry(retriever: ChromaRetriever) -> ToolRegistry:
    """Bộ tool đầy đủ của agent tư vấn tuyển sinh."""
    return ToolRegistry(
        [
            make_attach_source_link(retriever),
            make_contact_support(),
        ]
    )


__all__ = [
    "build_registry",
    "make_attach_source_link",
    "make_contact_support",
]
