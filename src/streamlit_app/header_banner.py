"""Header + banner mượn từ bản scrape WP (`prototype.html`) — toàn bộ landing page.

Chỉ lấy: logo VinUni (trích base64 từ `prototype.html`'s `--logo-src`, lưu thành
file thật) + nav menu (nhãn giống WP, không có page đích thật — `href="#"` chấp
nhận được vì chỉ để trông giống bản gốc, không cam kết điều hướng) + ảnh banner
phủ kín màn hình.

Asset ảnh (`vinuni_banner.webp`, `vinuni_logo_white.webp`) nằm trong
`streamlit/assets/` — tự chứa trong thư mục app, không phụ thuộc cây thư mục
nào khác. Base64-encode ngay tại đây, không phụ thuộc FastAPI static server
đang chạy hay không.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_FILE = ASSETS_DIR / "vinuni_logo_white.webp"
BANNER_FILE = ASSETS_DIR / "vinuni_banner.webp"

DISCLAIMER_TEXT = "⚠️ Đây là bản demo minh hoạ — KHÔNG phải website chính thức của VinUni."

# Nhãn y hệt nav chính của prototype.html (dòng ~1350-1355) — không có trang đích
# thật, giữ lại làm cảnh nhìn giống bản gốc; click không điều hướng đi đâu cả.
NAV_ITEMS = ["Đào tạo", "Nghiên cứu", "Hợp tác", "Tuyển sinh", "Đời sống sinh viên", "Về VinUni"]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def render_header_banner() -> None:
    logo_b64 = _b64(LOGO_FILE) if LOGO_FILE.exists() else None
    banner_b64 = _b64(BANNER_FILE) if BANNER_FILE.exists() else None

    st.markdown(
        """
        <style>
        /* Ẩn chrome mặc định của Streamlit để banner nằm sát top, phủ hết màn hình */
        [data-testid="stHeader"] { display: none; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        body { overflow-x: hidden; }

        .vn-disclaimer {
            width: 100%;
            background: #c72127;
            color: #ffffff;
            text-align: center;
            font-weight: 700;
            padding: 10px 16px;
            font-size: 0.95rem;
        }
        .vn-header {
            width: 100%;
            background: #134d8b;
            padding: 14px 24px;
            display: flex;
            align-items: center;
            gap: 32px;
            flex-wrap: wrap;
        }
        .vn-header img { height: 34px; }
        .vn-nav { display: flex; gap: 22px; flex-wrap: wrap; margin: 0; padding: 0; list-style: none; }
        .vn-nav a {
            color: #ffffff;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 600;
            opacity: 0.9;
        }
        .vn-nav a:hover { opacity: 1; text-decoration: underline; }

        .vn-banner {
            position: relative;
            width: 100%;
            height: calc(100vh - 100px);
            background-size: cover;
            background-position: center center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="vn-disclaimer">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)

    logo_html = f'<img src="data:image/webp;base64,{logo_b64}" alt="VinUniversity">' if logo_b64 else ""
    nav_html = "".join(f'<li><a href="#">{label}</a></li>' for label in NAV_ITEMS)
    st.markdown(
        f'<div class="vn-header">{logo_html}<ul class="vn-nav">{nav_html}</ul></div>',
        unsafe_allow_html=True,
    )

    if banner_b64:
        st.markdown(
            f'<div class="vn-banner" style="background-image:url(data:image/webp;base64,{banner_b64});" '
            'role="img" aria-label="Không gian khuôn viên Trường Đại học VinUni"></div>',
            unsafe_allow_html=True,
        )


__all__ = ["render_header_banner"]
