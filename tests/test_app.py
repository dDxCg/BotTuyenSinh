"""FastAPI app (`src/app.py`) — offline: override Service bằng fake, không nạp
E5 local, không gọi OpenRouter thật. `get_service` được override qua
`app.dependency_overrides` (cơ chế DI chuẩn của FastAPI), tránh chạy `lifespan`
(vốn gọi `warmup_local_model()` nạp model thật, chậm và cần model đã tải)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from src.app import app, get_service
from src.service import Reply


@dataclass
class FakeDemoService:
    """Thay `Service` thật — trả `Reply` cố định, ghi lại lời gọi để assert."""

    last_call: tuple[str, str] | None = None
    reset_calls: list[str] | None = None

    def __post_init__(self) -> None:
        self.reset_calls = []

    def chat(self, session_id: str, question: str) -> Reply:
        self.last_call = (session_id, question)
        return Reply(
            answer="Điều kiện dự tuyển gồm...",
            sources=[],
            suggestions=["Học phí bao nhiêu?"],
            grounded=True,
            top_score=0.9,
            path="agent+tool_calling",
        )

    def reset(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


@pytest.fixture
def client():
    # Không dùng `with TestClient(app)`: context manager chạy lifespan, vốn gọi
    # warmup_local_model() nạp E5 thật — override get_service() đã đủ, khỏi cần lifespan.
    fake = FakeDemoService()
    app.dependency_overrides[get_service] = lambda: fake
    test_client = TestClient(app)
    test_client.fake_service = fake  # type: ignore[attr-defined]
    yield test_client
    app.dependency_overrides.pop(get_service, None)


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_demo_reply(client: TestClient) -> None:
    response = client.post("/api/chat", json={"session_id": "s1", "message": "Điều kiện dự tuyển?"})
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "agent+tool_calling"
    assert body["grounded"] is True
    assert client.fake_service.last_call == ("s1", "Điều kiện dự tuyển?")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "", "message": "hi"},
        {"session_id": "x" * 129, "message": "hi"},
        {"session_id": "s1", "message": ""},
        {"session_id": "s1", "message": "x" * 2001},
    ],
)
def test_chat_rejects_invalid_payload(client: TestClient, payload: dict) -> None:
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_reset_calls_service(client: TestClient) -> None:
    response = client.post("/api/reset", json={"session_id": "s1"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert client.fake_service.reset_calls == ["s1"]  # type: ignore[attr-defined]
