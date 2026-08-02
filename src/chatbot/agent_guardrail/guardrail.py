import re
import unicodedata

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


def _plain(value: str) -> str:
    text = value.lower().replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def classify_restricted(question: str) -> str | None:
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


__all__ = ["UNRELATED_PATTERNS", "UNRELATED_REPLY", "classify_restricted"]
