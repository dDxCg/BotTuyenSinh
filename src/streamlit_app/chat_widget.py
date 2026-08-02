from __future__ import annotations

import uuid

import streamlit as st

from api_client import ApiClient, ApiError, ChatReply

STEP_LABELS = {
    "guardrail": "Kiểm tra câu hỏi...",
    "respond_restricted": "Chuẩn bị câu trả lời...",
    "planner": "Lên kế hoạch truy vấn...",
    "query_split": "Tách ý câu hỏi...",
    "hyde": "Suy luận ngữ cảnh...",
    "retrieve": "Tìm tài liệu liên quan...",
    "grounding_decision": "Đánh giá độ liên quan...",
    "respond_no_grounding": "Chuẩn bị câu trả lời...",
    "agent": "Soạn câu trả lời...",
    "tools": "Tra cứu thêm thông tin...",
    "finalize": "Hoàn tất...",
}


WP_NAVY = "#134d8b"
WP_RED = "#c83538"
WP_FONT = "'Montserrat','Segoe UI',sans-serif"

PANEL_SIZES = {
    "compact": {"width": 340, "height": 440},
    "large": {"width": 460, "height": 620},
}


def _inject_css(panel_size: str) -> None:
    dims = PANEL_SIZES[panel_size]
    st.markdown(
        f"""
        <style>
        /* Streamlit ép width:100% lên mọi stVerticalBlock — không ghi đè thì
        `right:24px` + width full-viewport đẩy div ra ngoài trái màn hình
        (left = right_edge - width = âm). fit-content bắt nó co về đúng kích
        thước nội dung trước khi neo theo `right`. */
        .st-key-chat_bubble {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 1000;
            width: fit-content !important;
        }}
        .st-key-chat_bubble button {{
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 1.4rem;
            background: {WP_NAVY};
            color: #ffffff;
            border: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        .st-key-chat_bubble button:hover {{ background: #0f3d6e; color: #ffffff; }}

        .st-key-chat_panel {{
            position: fixed;
            bottom: 92px;
            right: 24px;
            z-index: 1000;
            width: {dims["width"]}px !important;
            height: {dims["height"]}px;
            max-height: 80vh;
            background: #ffffff;
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 16px;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.22);
            padding: 0.75rem;
            overflow-y: auto;
            font-family: {WP_FONT};
        }}

        .vn-chat-header {{
            background: {WP_NAVY};
            color: #ffffff;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 10px 14px;
            margin: -0.75rem -0.75rem 0.6rem -0.75rem;
            border-radius: 16px 16px 0 0;
        }}

        .st-key-chat_panel [data-testid="stChatMessage"] {{
            background: #f3f4f6;
            border-radius: 10px;
        }}
        .st-key-chat_resize button {{
            border-radius: 6px;
            border-color: {WP_NAVY};
            color: {WP_NAVY};
        }}
        .st-key-chat_reset_btn button {{
            border-radius: 6px;
            border-color: {WP_RED};
            color: {WP_RED};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    st.session_state.setdefault("chat_open", False)
    st.session_state.setdefault("chat_panel_size", "compact")
    st.session_state.setdefault("chat_session_id", str(uuid.uuid4()))
    st.session_state.setdefault("chat_messages", [])


def render_chat_widget(client: ApiClient) -> None:
    _init_state()
    _inject_css(st.session_state.chat_panel_size)

    with st.container(key="chat_bubble"):
        icon = "✕" if st.session_state.chat_open else "💬"
        if st.button(icon, key="chat_bubble_toggle"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

    if not st.session_state.chat_open:
        return

    with st.container(key="chat_panel"):
        st.markdown('<div class="vn-chat-header">Trợ lý tuyển sinh AI Thực Chiến</div>', unsafe_allow_html=True)

        _, resize_col, reset_col = st.columns([3, 1, 1])
        with resize_col:
            if st.button("⤢", key="chat_resize", help="Đổi kích thước cửa sổ chat"):
                st.session_state.chat_panel_size = (
                    "large" if st.session_state.chat_panel_size == "compact" else "compact"
                )
                st.rerun()
        with reset_col:
            if st.button("🗑", key="chat_reset_btn", help="Xoá hội thoại"):
                try:
                    client.reset(st.session_state.chat_session_id)
                except ApiError as exc:
                    st.error(str(exc))
                st.session_state.chat_messages = []
                st.rerun()

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Hỏi về tuyển sinh...", key="chat_prompt")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.status("Đang xử lý...", expanded=True) as status:
                        if not st.session_state.get("backend_awake"):
                            status.update(label="Đánh thức backend (có thể mất ~1 phút lần đầu)...")
                            client.wake_up()
                            st.session_state.backend_awake = True

                        reply: ChatReply | None = None
                        for item in client.chat_stream(st.session_state.chat_session_id, prompt):
                            if isinstance(item, ChatReply):
                                reply = item
                            else:
                                status.write(STEP_LABELS.get(item, item))
                        status.update(label="Xong", state="complete", expanded=False)
                    st.write(reply.answer)
                    if reply.sources:
                        with st.expander("Nguồn tham khảo"):
                            for source in reply.sources:
                                label = source.get("label_hien_thi", source.get("source_link", ""))
                                link = source.get("source_link", "")
                                st.write(f"- [{label}]({link})" if link else f"- {label}")
                    answer_text = reply.answer
                except ApiError as exc:
                    st.error(str(exc))
                    answer_text = f"(lỗi kết nối backend: {exc})"
            st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})


__all__ = ["render_chat_widget"]
