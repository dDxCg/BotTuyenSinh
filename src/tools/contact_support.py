"""Tool 1 — chuyển câu hỏi cho nhân viên tuyển sinh khi agent không có căn cứ để trả lời.

Thiết kế: docs/design-agent-tools.md §2.
"""

from dataclasses import dataclass, field
from typing import Literal

Reason = Literal["no_grounding", "out_of_scope", "conflicting_sources", "personal_data_request"]

# Similarity score dưới ngưỡng này coi là "không có căn cứ" — chốt cùng team 2026-07-30.
NO_GROUNDING_THRESHOLD = 0.7

# Nguồn: data/web/_clean/thong-tin-tuyen-sinh-chuong-trinh-dao-tao-nhan-tai-ai-thuc-chien-khoa-co-ban.md
CONTACT_CHANNELS = {
    "hotline": "0979.489.846",
    "tuyen_sinh": "Ms. Phương Thảo — 0388.339.478",
    "email": "AIthucchien@vinuni.edu.vn",
}

_REASON_MESSAGES: dict[Reason, str] = {
    "no_grounding": "Mình chưa tìm thấy dữ kiện đủ căn cứ trong tài liệu tuyển sinh chính thức và chia sẻ cộng đồng để trả lời câu này chính xác.",
    "out_of_scope": "Câu này nằm ngoài phạm vi mình được hỗ trợ (vd. tra cứu hồ sơ cá nhân, tư vấn nên nộp hay không) — mình không thể tự trả lời thay.",
    "conflicting_sources": "Mình tìm thấy hai dữ kiện khác nhau về việc này và không có căn cứ để chọn bên nào đúng hơn.",
    "personal_data_request": "Mình không xử lý thông tin hồ sơ/cá nhân qua kênh chat này.",
}


@dataclass
class ContactSupportResult:
    message: str
    contact_channels: dict[str, str]
    suggested_question: str
    conflicting_facts: list[str] = field(default_factory=list)


def contact_support(
    reason: Reason,
    user_question: str,
    partial_context: str | None = None,
) -> ContactSupportResult:
    """Không đoán, không tự trả lời — luôn trả kênh liên hệ + câu hỏi soạn sẵn cho user chỉnh trước khi gửi."""
    conflicting_facts: list[str] = []
    if reason == "conflicting_sources" and partial_context:
        conflicting_facts = [line.strip() for line in partial_context.splitlines() if line.strip()]

    return ContactSupportResult(
        message=_REASON_MESSAGES[reason],
        contact_channels=dict(CONTACT_CHANNELS),
        suggested_question=user_question.strip(),
        conflicting_facts=conflicting_facts,
    )


CONTACT_SUPPORT_SCHEMA = {
    "name": "contact_support",
    "description": (
        "Chuyển câu hỏi của người dùng cho nhân viên tuyển sinh khi agent không có căn cứ trong tài liệu "
        "chính thức/cộng đồng để trả lời, hoặc câu hỏi ngoài phạm vi agent được phép trả lời "
        "(vd. tra hồ sơ cá nhân, xin lời khuyên nên nộp hay không)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": ["no_grounding", "out_of_scope", "conflicting_sources", "personal_data_request"],
                "description": "Vì sao agent không tự trả lời được",
            },
            "user_question": {
                "type": "string",
                "description": "Câu hỏi nguyên văn của user, dùng để soạn sẵn câu hỏi chuyển cho nhân viên",
            },
            "partial_context": {
                "type": "string",
                "description": (
                    "Optional — dữ kiện liên quan gần nhất agent tìm được dù chưa đủ căn cứ trả lời trực tiếp "
                    "(vd. hai chunk mâu thuẫn, mỗi dòng một dữ kiện)"
                ),
            },
        },
        "required": ["reason", "user_question"],
    },
}
