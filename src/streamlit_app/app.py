from __future__ import annotations

import threading

import streamlit as st

from api_client import ApiClient
from chat_widget import render_chat_widget
from header_banner import render_header_banner

st.set_page_config(page_title="VinUni", layout="wide")

client = ApiClient()

if "wake_up_started" not in st.session_state:
    st.session_state.wake_up_started = True
    threading.Thread(target=client.wake_up, daemon=True).start()

render_header_banner()
render_chat_widget(client)
