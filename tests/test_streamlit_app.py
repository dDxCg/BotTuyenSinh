"""Streamlit UI (`ui/streamlit_app/`) — offline qua `AppTest` (chạy script thật,
không cần trình duyệt). Patch `ApiClient.chat`/`.reset` để không gọi backend
FastAPI thật qua HTTP."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP_DIR = PROJECT_ROOT / "src" / "streamlit_app"
APP_FILE = STREAMLIT_APP_DIR / "app.py"

sys.path.insert(0, str(STREAMLIT_APP_DIR))

from api_client import ChatReply  # noqa: E402


@pytest.fixture
def at() -> AppTest:
    with patch(
        "api_client.ApiClient.chat",
        return_value=ChatReply(
            answer="Điều kiện dự tuyển gồm tốt nghiệp THPT.",
            sources=[{"label_hien_thi": "Thông tin tuyển sinh", "source_link": "https://vinuni.edu.vn/x"}],
            suggestions=[],
            grounded=True,
            top_score=0.9,
            path="agent+tool_calling",
        ),
    ), patch("api_client.ApiClient.reset", return_value=None):
        test = AppTest.from_file(str(APP_FILE))
        test.run()
        yield test


def test_landing_renders_without_error(at: AppTest) -> None:
    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "KHÔNG phải website chính thức" in rendered  # disclaimer bắt buộc trước deploy
    assert "vn-banner" in rendered  # banner ảnh mượn từ WP, phủ full màn hình
    assert "Tuyển sinh" in rendered  # nav item mượn từ WP (không có trang đích thật)


def test_chat_bubble_starts_closed(at: AppTest) -> None:
    assert at.session_state["chat_open"] is False
    assert at.button(key="chat_bubble_toggle") is not None
    assert not at.chat_input  # panel chưa render khi đóng


def test_bubble_click_opens_panel(at: AppTest) -> None:
    at.button(key="chat_bubble_toggle").click().run()
    assert at.session_state["chat_open"] is True
    assert at.chat_input(key="chat_prompt") is not None


def test_resize_toggles_panel_size(at: AppTest) -> None:
    at.button(key="chat_bubble_toggle").click().run()
    assert at.session_state["chat_panel_size"] == "compact"
    at.button(key="chat_resize").click().run()
    assert at.session_state["chat_panel_size"] == "large"
    at.button(key="chat_resize").click().run()
    assert at.session_state["chat_panel_size"] == "compact"


def test_send_message_calls_api_and_renders_reply(at: AppTest) -> None:
    at.button(key="chat_bubble_toggle").click().run()
    at.chat_input(key="chat_prompt").set_value("Điều kiện dự tuyển là gì?").run()

    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "Điều kiện dự tuyển gồm tốt nghiệp THPT" in rendered
    assert at.session_state["chat_messages"][-1]["content"] == "Điều kiện dự tuyển gồm tốt nghiệp THPT."


def test_reset_clears_messages(at: AppTest) -> None:
    at.button(key="chat_bubble_toggle").click().run()
    at.chat_input(key="chat_prompt").set_value("Học phí bao nhiêu?").run()
    assert at.session_state["chat_messages"]

    at.button(key="chat_reset_btn").click().run()
    assert at.session_state["chat_messages"] == []
