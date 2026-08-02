from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import requests


TIMEOUT_SECONDS = 90
HEALTH_TIMEOUT_SECONDS = 100


@dataclass
class ChatReply:
    answer: str
    sources: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    grounded: bool = False
    top_score: float | None = None
    path: str = ""


class ApiError(RuntimeError):
    pass


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    detail = payload.get("detail")
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or f"HTTP {response.status_code}")


class ApiClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}{path}", json=payload, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise ApiError(f"Không kết nối được backend ({self.base_url}): {exc}") from exc
        if response.status_code != 200:
            raise ApiError(_error_detail(response))
        return response.json()

    def chat(self, session_id: str, message: str) -> ChatReply:
        data = self._post("/api/chat", {"session_id": session_id, "message": message})
        return ChatReply(**data)

    def chat_stream(self, session_id: str, message: str) -> Iterator[str | ChatReply]:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/stream",
                json={"session_id": session_id, "message": message},
                timeout=TIMEOUT_SECONDS,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Không kết nối được backend ({self.base_url}): {exc}") from exc

        if response.status_code != 200:
            raise ApiError(_error_detail(response))

        event = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None or raw_line == "":
                continue
            if raw_line.startswith("event:"):
                event = raw_line[len("event:"):].strip()
            elif raw_line.startswith("data:"):
                data = json.loads(raw_line[len("data:"):].strip())
                if event == "step":
                    yield data["node"]
                elif event == "result":
                    yield ChatReply(**data)
                elif event == "error":
                    raise ApiError(data.get("detail", "Lỗi backend"))

    def reset(self, session_id: str) -> None:
        self._post("/api/reset", {"session_id": session_id})

    def wake_up(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=HEALTH_TIMEOUT_SECONDS)
        except requests.RequestException:
            return False
        return response.status_code == 200


__all__ = ["ApiClient", "ApiError", "ChatReply"]
