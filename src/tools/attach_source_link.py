"""Tool 2 — đính kèm link nguồn gốc của (các) chunk đã dùng để trả lời.

Thiết kế: docs/design-agent-tools.md §3. Nguồn phải là metadata cứng gắn lúc
ingest (source_type/source_url theo từng file gốc) — không suy ra ở tầng model.
"""

from dataclasses import dataclass
from typing import Literal

SourceType = Literal["official_web"]

# Nhãn hiển thị cho user theo URL nguồn — cập nhật khi có bản mới từ VinUni/Vingroup.
DISPLAY_LABELS: dict[str, str] = {
    "https://vinuni.edu.vn/vi/thong-tin-tuyen-sinh-chuong-trinh-dao-tao-nhan-tai-ai-thuc-chien-khoa-co-ban/": (
        "Thông tin tuyển sinh chính thức — VinUni"
    ),
    "https://vinuni.edu.vn/wp-content/uploads/2025/04/20K-AI-Handbook_final.pdf": "20K AI Handbook — VinUni",
    "https://vinuni.edu.vn/vi/vingroup-tang-toc-dao-tao-20-000-nhan-tai-ai-thuc-chien/": (
        "Vingroup tăng tốc đào tạo AI Thực chiến"
    ),
}


def _display_label(source_url: str) -> str:
    """Chấp nhận metadata cũ có/không có dấu gạch chéo cuối URL."""
    return (
        DISPLAY_LABELS.get(source_url)
        or DISPLAY_LABELS.get(source_url.rstrip("/"))
        or DISPLAY_LABELS.get(f"{source_url.rstrip('/')}/")
        or source_url
    )


@dataclass
class ChunkRef:
    """Chunk đã có metadata nguồn gắn sẵn từ lúc ingest."""

    chunk_id: str
    source_type: SourceType
    source_url: str


def attach_source_link(chunk_refs: list[ChunkRef]) -> list[dict]:
    """Format nguồn cho từng chunk đã dùng để trả lời."""
    attachments = []
    for chunk in chunk_refs:
        label = _display_label(chunk.source_url)
        attachments.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "source_url": chunk.source_url,
                "label_hien_thi": label,
            }
        )
    return attachments


ATTACH_SOURCE_LINK_SCHEMA = {
    "name": "attach_source_link",
    "description": (
        "Lấy link nguồn gốc của (các) chunk đã dùng để trả lời, dựa trên nhãn source_type đã gắn khi "
        "ingest. Gọi sau khi đã chọn được chunk trả lời, trước khi trả kết quả cuối cho user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ID của các chunk đã dùng để tạo câu trả lời",
            },
        },
        "required": ["chunk_ids"],
    },
}
