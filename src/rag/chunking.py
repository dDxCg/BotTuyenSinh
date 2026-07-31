"""Chunk tài liệu Markdown theo cấu trúc để dùng cho hệ thống RAG.

Mặc định script đọc:

- ``data/Data_FaceBook_ckean/**/*.md``
- ``data/web/_clean/**/*.md``

và ghi kết quả dễ kiểm tra vào ``src/rag/chunks.json``.

Chạy từ thư mục gốc dự án:

    python src/rag/chunking.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACEBOOK_DIR = PROJECT_ROOT / "data" / "Data_FaceBook_ckean"
DEFAULT_WEB_DIR = PROJECT_ROOT / "data" / "web" / "_clean"
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "rag" / "chunks.json"
FACEBOOK_DOCUMENT_NAME = "feedback_nguoi_dung_tren_Facebook"

SOURCE_RE = re.compile(
    r"^\s*<!--\s*source\s*:\s*(?P<link>https?://\S+?)\s*-->\s*$",
    re.IGNORECASE,
)
ATX_HEADING_RE = re.compile(
    r"^\s*(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$"
)
NUMBERED_TITLE_RE = re.compile(
    r"^(?P<label>(?:[IVXLCDM]+|\d+(?:\.\d+)+|\d+))"
    r"(?:(?:[.)])(?=\s)|(?=\s))\s+(?P<title>.+)$",
    re.IGNORECASE,
)
EMPHASIZED_NUMBERED_HEADING_RE = re.compile(
    r"^\s*_?\*\*"
    r"(?P<label>(?:[IVXLCDM]+|\d+(?:\.\d+)+|\d+))"
    r"(?:(?:[.)])(?=\s)|(?=\s))\s*"
    r"(?P<title>.*?)"
    r"\*\*_?\s*(?P<body>.*)$",
    re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Heading:
    """Một tiêu đề đã được chuẩn hóa về cấp cấu trúc."""

    level: int
    title: str
    inline_body: str = ""


@dataclass
class Section:
    """Nội dung thuộc một nhánh trong cây đề mục."""

    major: str | None = None
    minor: str | None = None
    subsection: str | None = None
    lines: list[str] = field(default_factory=list)


def extract_source_link(lines: Sequence[str], source_file: Path) -> str:
    """Lấy source từ phần đầu tài liệu và báo lỗi nếu tài liệu thiếu source."""

    for line in lines[:20]:
        match = SOURCE_RE.match(line)
        if match:
            return match.group("link")
        if line.strip() and not line.lstrip().startswith("<!--"):
            break
    raise ValueError(
        f"Không tìm thấy '<!-- source: https://... -->' ở đầu file: {source_file}"
    )


def clean_heading_title(value: str) -> str:
    """Bỏ Markdown trang trí nhưng giữ nguyên nội dung tiêu đề."""

    value = value.strip()
    for opening, closing in (("_**", "**_"), ("**", "**"), ("__", "__")):
        if value.startswith(opening) and value.endswith(closing):
            value = value[len(opening) : -len(closing)].strip()
            break
    return value.rstrip(":：").strip()


def numbered_heading(line: str) -> tuple[str, str, str] | None:
    """Đọc tiêu đề dạng ``**I. ...**`` hoặc ``**1. ...:** nội dung``."""

    match = EMPHASIZED_NUMBERED_HEADING_RE.match(line)
    if not match:
        return None

    label = match.group("label")
    title = clean_heading_title(match.group("title"))
    body = match.group("body").strip()
    full_title = f"{label}. {title}" if title else f"{label}."
    return label, full_title, body


def contains_roman_sections(lines: Sequence[str]) -> bool:
    """Xác định tài liệu dùng phân cấp I → 1 → 1.1."""

    for line in lines:
        parsed = numbered_heading(line)
        if parsed and re.fullmatch(r"[IVXLCDM]+", parsed[0], re.IGNORECASE):
            return True
    return False


def parse_heading(line: str, roman_schema: bool) -> Heading | None:
    """Chuyển một dòng tiêu đề thành cấp 1 (lớn), 2 (nhỏ), hoặc 3 (con)."""

    logical = numbered_heading(line)
    if logical:
        label, title, body = logical
        if re.fullmatch(r"[IVXLCDM]+", label, re.IGNORECASE):
            level = 1
        else:
            level = 2 if "." not in label else 3
        return Heading(level=level, title=title, inline_body=body)

    markdown = ATX_HEADING_RE.match(line)
    if not markdown:
        return None

    markdown_level = len(markdown.group("marks"))
    raw_title = clean_heading_title(markdown.group("title"))

    # H1 là tên tài liệu, không phải đề mục nội dung.
    if markdown_level == 1:
        return Heading(level=0, title=raw_title)

    # Khi tài liệu có I/II/III, ưu tiên số thứ tự ngữ nghĩa.
    numbered = NUMBERED_TITLE_RE.match(raw_title)
    if roman_schema and numbered:
        label = numbered.group("label")
        if re.fullmatch(r"[IVXLCDM]+", label, re.IGNORECASE):
            level = 1
        else:
            level = 2 if "." not in label else 3
        return Heading(level=level, title=raw_title)

    # Với FAQ và handbook, cấp Markdown phản ánh đúng cấu trúc tài liệu.
    return Heading(level=min(markdown_level - 1, 3), title=raw_title)


def section_has_content(section: Section) -> bool:
    return any(line.strip() for line in section.lines)


def parse_sections(lines: Sequence[str]) -> list[Section]:
    """Tách tài liệu thành các section lá, giữ context của đề mục cha."""

    roman_schema = contains_roman_sections(lines)
    sections: list[Section] = []
    current = Section()

    def flush() -> None:
        nonlocal current
        if section_has_content(current):
            sections.append(current)
        current = Section(
            major=current.major,
            minor=current.minor,
            subsection=current.subsection,
        )

    for line in lines:
        if SOURCE_RE.match(line):
            continue

        heading = parse_heading(line, roman_schema)
        if heading is None:
            current.lines.append(line.rstrip())
            continue

        if heading.level == 0:
            # Tên tài liệu đã nằm trong metadata; tránh lặp ở mọi chunk.
            continue

        flush()
        if heading.level == 1:
            current.major = heading.title
            current.minor = None
            current.subsection = None
        elif heading.level == 2:
            current.minor = heading.title
            current.subsection = None
        else:
            current.subsection = heading.title

        if heading.inline_body:
            current.lines.append(heading.inline_body)

    flush()
    return sections


def lines_to_blocks(lines: Sequence[str]) -> list[str]:
    """Gom đoạn văn; giữ từng gạch đầu dòng thành đơn vị không bị cắt."""

    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        if LIST_ITEM_RE.match(line):
            flush()
            blocks.append(line.strip())
            continue
        current.append(line.strip())

    flush()
    return blocks


def split_by_words(text: str, limit: int) -> list[str]:
    """Phương án cuối cho đoạn đơn lẻ dài hơn giới hạn."""

    words = text.split()
    if not words:
        return []

    parts: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= limit:
            current = candidate
        else:
            parts.append(current)
            current = word
    parts.append(current)
    return parts


def split_oversized_block(block: str, limit: int) -> list[str]:
    """Cắt block lớn theo dòng, câu, rồi mới tới từ."""

    if len(block) <= limit:
        return [block]

    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) > 1:
        parts: list[str] = []
        for line in lines:
            parts.extend(split_oversized_block(line, limit))
        return parts

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_RE.split(block)
        if sentence.strip()
    ]
    if len(sentences) > 1:
        parts = []
        current = sentences[0]
        for sentence in sentences[1:]:
            candidate = f"{current} {sentence}"
            if len(candidate) <= limit:
                current = candidate
            else:
                parts.extend(split_by_words(current, limit))
                current = sentence
        parts.extend(split_by_words(current, limit))
        return parts

    return split_by_words(block, limit)


def section_prefix(section: Section) -> str:
    """Tạo breadcrumb ngắn để embedding vẫn mang ngữ cảnh đề mục."""

    headings = [
        (1, section.major),
        (2, section.minor),
        (3, section.subsection),
    ]
    return "\n".join(
        f"{'#' * level} {title}" for level, title in headings if title
    )


def split_section(section: Section, max_chars: int) -> list[str]:
    """Tách section theo block, không vượt max_chars khi có thể."""

    prefix = section_prefix(section)
    separator = "\n\n" if prefix else ""
    body_limit = max(200, max_chars - len(prefix) - len(separator))
    blocks: list[str] = []
    for block in lines_to_blocks(section.lines):
        blocks.extend(split_oversized_block(block, body_limit))

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for block in blocks:
        added_length = len(block) + (2 if current else 0)
        if current and current_length + added_length > body_limit:
            body = "\n\n".join(current)
            chunks.append(f"{prefix}{separator}{body}".strip())
            current = [block]
            current_length = len(block)
        else:
            current.append(block)
            current_length += added_length

    if current:
        body = "\n\n".join(current)
        chunks.append(f"{prefix}{separator}{body}".strip())
    return chunks


def relative_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_chunk_id(
    source_path: str,
    section: Section,
    content: str,
    part_index: int,
) -> str:
    raw_id = "\0".join(
        [
            source_path,
            section.major or "",
            section.minor or "",
            section.subsection or "",
            str(part_index),
            content,
        ]
    )
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def chunk_document(
    source_file: Path,
    source_type: str,
    max_chars: int,
) -> list[dict[str, object]]:
    """Chunk một file và thêm metadata truy xuất nguồn."""

    text = source_file.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    source_link = extract_source_link(lines, source_file)
    source_path = relative_source_path(source_file)
    document_name = (
        FACEBOOK_DOCUMENT_NAME if source_type == "facebook" else source_file.name
    )

    chunks: list[dict[str, object]] = []
    document_chunk_index = 0
    for section in parse_sections(lines):
        for part_index, content in enumerate(
            split_section(section, max_chars=max_chars), start=1
        ):
            metadata = {
                "muc_lon": section.major,
                "muc_nho": section.minor,
                "muc_con": section.subsection,
                "ten_tai_lieu": document_name,
                "source_link": source_link,
                "loai_nguon": source_type,
                "source_file": source_path,
                "chunk_index": document_chunk_index,
                "part_index": part_index,
            }
            chunks.append(
                {
                    "id": make_chunk_id(
                        source_path=source_path,
                        section=section,
                        content=content,
                        part_index=part_index,
                    ),
                    "content": content,
                    "metadata": metadata,
                }
            )
            document_chunk_index += 1
    return chunks


def markdown_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {directory}")
    yield from sorted(path for path in directory.rglob("*.md") if path.is_file())


def build_chunks(
    facebook_dir: Path = DEFAULT_FACEBOOK_DIR,
    web_dir: Path = DEFAULT_WEB_DIR,
    max_chars: int = 1800,
) -> list[dict[str, object]]:
    """Chunk toàn bộ Facebook và Web clean."""

    if max_chars < 400:
        raise ValueError("max_chars phải từ 400 trở lên")

    chunks: list[dict[str, object]] = []
    for source_file in markdown_files(facebook_dir):
        chunks.extend(chunk_document(source_file, "facebook", max_chars))
    for source_file in markdown_files(web_dir):
        chunks.extend(chunk_document(source_file, "web", max_chars))

    for global_index, chunk in enumerate(chunks):
        metadata = chunk["metadata"]
        if isinstance(metadata, dict):
            metadata["global_chunk_index"] = global_index
    return chunks


def save_chunks(chunks: Sequence[dict[str, object]], output_file: Path) -> None:
    """Ghi JSON UTF-8, tiếng Việt không bị escape."""

    document_names = sorted(
        {
            str(chunk["metadata"]["ten_tai_lieu"])
            for chunk in chunks
            if isinstance(chunk.get("metadata"), dict)
        }
    )
    payload = {
        "total_documents": len(document_names),
        "total_chunks": len(chunks),
        "chunks": list(chunks),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk dữ liệu Facebook/Web theo cấu trúc tài liệu."
    )
    parser.add_argument(
        "--facebook-dir",
        type=Path,
        default=DEFAULT_FACEBOOK_DIR,
        help=f"Thư mục Facebook clean (mặc định: {DEFAULT_FACEBOOK_DIR})",
    )
    parser.add_argument(
        "--web-dir",
        type=Path,
        default=DEFAULT_WEB_DIR,
        help=f"Thư mục Web clean (mặc định: {DEFAULT_WEB_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File JSON đầu ra (mặc định: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Số ký tự tối đa gần đúng cho mỗi chunk (mặc định: 1800)",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    chunks = build_chunks(
        facebook_dir=args.facebook_dir,
        web_dir=args.web_dir,
        max_chars=args.max_chars,
    )
    save_chunks(chunks, args.output)
    print(f"Đã ghi {len(chunks)} chunks vào {args.output.resolve()}")


if __name__ == "__main__":
    main()
