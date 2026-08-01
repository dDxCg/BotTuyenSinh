"""Entry point Streamlit — landing page full-screen (header + banner VinUni) + chat bubble nổi.

Chạy:
    uv run streamlit run src/streamlit_app/app.py

Cần backend FastAPI chạy trước (`uv run python -m src.app`), mặc định gọi tới
http://127.0.0.1:8000 — đổi qua biến môi trường API_BASE_URL nếu backend ở nơi khác.
"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClient
from chat_widget import render_chat_widget
from header_banner import render_header_banner

st.set_page_config(page_title="AI Thực Chiến — VinUni", page_icon="🎓", layout="wide")

render_header_banner()
render_chat_widget(ApiClient())
