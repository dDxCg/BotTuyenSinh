"""Tool 2 — đính kèm link nguồn gốc của (các) chunk đã dùng để trả lời.

Thiết kế: docs/design-agent-tools.md §3. Nguồn phải là metadata cứng gắn lúc
ingest (source_type/source_url theo từng file gốc) — không suy ra ở tầng model.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceType = Literal["official_web", "community_facebook"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_CLEAN_DIR = PROJECT_ROOT / "data" / "web" / "_clean"
FACEBOOK_CLEAN_DIR = PROJECT_ROOT / "data" / "Data_FaceBook_ckean"

SOURCE_COMMENT_RE = re.compile(r"<!--\s*source:\s*(\S+)\s*-->")

# Nhãn hiển thị cho user theo URL nguồn — cập nhật khi có bản mới từ VinUni/Vingroup.
DISPLAY_LABELS: dict[str, str] = {
    "https://vinuni.edu.vn/vi/thong-tin-tuyen-sinh-chuong-trinh-dao-tao-nhan-tai-ai-thuc-chien-khoa-co-ban/": (
        "Thông tin tuyển sinh chính thức — VinUni"
    ),
    "https://vinuni.edu.vn/wp-content/uploads/2025/04/20K-AI-Handbook_final.pdf": "20K AI Handbook — VinUni",
    "https://vinuni.edu.vn/vi/vingroup-tang-toc-dao-tao-20-000-nhan-tai-ai-thuc-chien/": (
        "Vingroup tăng tốc đào tạo AI Thực chiến"
    ),
    "https://www.facebook.com/groups/2125430681651241": (
        "Chia sẻ cộng đồng — nhóm Facebook (không phải nguồn chính thức)"
    ),
}

COMMUNITY_WARNING = (
    "Đây là chia sẻ cộng đồng, không phải nguồn chính thức. "
    "Khi khác biệt, ưu tiên thông báo tuyển sinh/email/sổ tay từ VinUni."
)


def _display_label(source_url: str) -> str:
    """Chấp nhận metadata cũ có/không có dấu gạch chéo cuối URL."""
    return (
        DISPLAY_LABELS.get(source_url)
        or DISPLAY_LABELS.get(source_url.rstrip("/"))
        or DISPLAY_LABELS.get(f"{source_url.rstrip('/')}/")
        or source_url
    )


@dataclass
class SourceMeta:
    source_type: SourceType
    source_url: str
    label: str


@dataclass
class ChunkRef:
    """Chunk đã có metadata nguồn gắn sẵn từ lúc ingest."""

    chunk_id: str
    source_type: SourceType
    source_url: str


def _extract_source_url(file_path: Path) -> str | None:
    first_line = file_path.read_text(encoding="utf-8").splitlines()[0]
    match = SOURCE_COMMENT_RE.search(first_line)
    return match.group(1) if match else None


def build_source_registry() -> dict[str, SourceMeta]:
    """Đọc dòng `<!-- source: URL -->` đầu mỗi file trong data/web/_clean và
    data/Data_FaceBook_ckean, trả về map file_slug -> SourceMeta. Dùng ở bước
    ingest để gắn source_type/source_url vào từng chunk theo file gốc."""
    registry: dict[str, SourceMeta] = {}
    for clean_dir, source_type in (
        (WEB_CLEAN_DIR, "official_web"),
        (FACEBOOK_CLEAN_DIR, "community_facebook"),
    ):
        for file_path in clean_dir.glob("*.md"):
            source_url = _extract_source_url(file_path)
            if source_url is None:
                raise ValueError(f"{file_path} thiếu dòng '<!-- source: URL -->' ở đầu file")
            registry[file_path.stem] = SourceMeta(
                source_type=source_type,
                source_url=source_url,
                label=_display_label(source_url),
            )
    return registry


def attach_source_link(chunk_refs: list[ChunkRef]) -> list[dict]:
    """Format nguồn cho từng chunk đã dùng để trả lời — không gộp link khác source_type lại với nhau."""
    attachments = []
    for chunk in chunk_refs:
        label = _display_label(chunk.source_url)
        attachment = {
            "chunk_id": chunk.chunk_id,
            "source_type": chunk.source_type,
            "source_url": chunk.source_url,
            "label_hien_thi": label,
        }
        if chunk.source_type == "community_facebook":
            attachment["warning"] = COMMUNITY_WARNING
        attachments.append(attachment)
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
