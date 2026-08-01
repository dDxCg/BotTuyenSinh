"""Regex guardrail — dùng chung giữa Service (cũ) và graph (mới)."""

import pytest

from src.chatbot.guardrail import classify_restricted


@pytest.mark.parametrize(
    "question",
    [
        "Trạng thái hồ sơ của em thế nào rồi?",
        "Cho em xin kết quả hồ sơ ạ, mã hồ sơ là AB123",
        "Em gửi email rồi, cho em kiểm tra kết quả với",
    ],
)
def test_personal_data_request(question: str) -> None:
    assert classify_restricted(question) == "personal_data_request"


@pytest.mark.parametrize(
    "question",
    [
        "Em có nên nộp hồ sơ không ạ?",
        "Liệu em có đậu không?",
        "Chương trình có cam kết đầu ra không?",
    ],
)
def test_out_of_scope(question: str) -> None:
    assert classify_restricted(question) == "out_of_scope"


@pytest.mark.parametrize(
    "question",
    [
        "2 + 2 ?",
        "Con gà có trước hay quả trứng có trước?",
        "Thời tiết hôm nay thế nào?",
    ],
)
def test_unrelated(question: str) -> None:
    assert classify_restricted(question) == "unrelated"


@pytest.mark.parametrize(
    "question",
    [
        "Điều kiện dự tuyển chương trình AI Thực Chiến là gì?",
        "Học phí bao nhiêu?",
        "Lịch học diễn ra khi nào?",
    ],
)
def test_in_scope_passes_through(question: str) -> None:
    assert classify_restricted(question) is None
