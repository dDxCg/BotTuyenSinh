"""1 nơi duy nhất tạo OpenAI client (OpenRouter) — chatbot và embedding dùng chung,
tránh mỗi module tự đọc key/base_url/.env riêng.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class LlmConfigError(RuntimeError):
    """Thiếu API key trong .env."""


def chat_client(
    api_key: str, base_url: str, timeout: float = 60.0, max_retries: int = 2
) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)


@lru_cache(maxsize=1)
def embedding_client() -> OpenAI:
    # Trống thì dùng chung key/endpoint OpenRouter với chat (OPENAI_API/OPENAI_BASE_URL).
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("OPENAI_API", "").strip()
    if not api_key:
        raise LlmConfigError("Thiếu EMBEDDING_API_KEY hoặc OPENAI_API trong .env")
    base_url = (
        os.getenv("EMBEDDING_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or None
    )
    return OpenAI(api_key=api_key, base_url=base_url)


__all__ = ["LlmConfigError", "chat_client", "embedding_client"]
