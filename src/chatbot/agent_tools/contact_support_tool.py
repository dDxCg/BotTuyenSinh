from ..types import Tool

try:
    from ...tools.contact_support import contact_support
except ImportError:
    from tools.contact_support import contact_support


def make_contact_support() -> Tool:
    def contact_support_tool(
        reason: str,
        user_question: str,
        partial_context: str | None = None,
    ) -> str:
        result = contact_support(reason, user_question, partial_context)
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


__all__ = ["make_contact_support"]
