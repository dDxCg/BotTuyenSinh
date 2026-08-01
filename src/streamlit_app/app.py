from __future__ import annotations

import streamlit as st

from api_client import ApiClient
from chat_widget import render_chat_widget
from header_banner import render_header_banner

st.set_page_config(page_title="VinUni", layout="wide")

render_header_banner()
render_chat_widget(ApiClient())
