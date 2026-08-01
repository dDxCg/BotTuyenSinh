from dataclasses import dataclass
from typing import Literal

SourceType = Literal["official_web"]


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
    return (
        DISPLAY_LABELS.get(source_url)
        or DISPLAY_LABELS.get(source_url.rstrip("/"))
        or DISPLAY_LABELS.get(f"{source_url.rstrip('/')}/")
        or source_url
    )


@dataclass
class ChunkRef:

    chunk_id: str
    source_type: SourceType
    source_url: str


def attach_source_link(chunk_refs: list[ChunkRef]) -> list[dict]:
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
