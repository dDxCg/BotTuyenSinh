from src.rag.types import Retriever

from ..types import Tool

try:
    from ...tools.attach_source_link import ChunkRef, attach_source_link
except ImportError:
    from tools.attach_source_link import ChunkRef, attach_source_link


def make_attach_source_link(retriever: Retriever) -> Tool:
    def attach_source_link_tool(chunk_ids: list[str] | str) -> str:
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]

        refs: list[ChunkRef] = []
        unknown: list[str] = []
        for chunk_id in chunk_ids:
            chunk = retriever.get_chunk(chunk_id)
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
            "trước khi chốt Final Answer."
        ),
        signature="attach_source_link(chunk_ids: list[str]) -> str",
        func=attach_source_link_tool,
    )


__all__ = ["make_attach_source_link"]
