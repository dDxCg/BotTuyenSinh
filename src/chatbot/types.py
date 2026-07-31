"""Contract giữa chatbot và hai tầng ngoài (RAG, tools).

Đổi retriever hay tool chỉ cần thoả contract ở đây.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    """Một đoạn ngữ cảnh trả về từ retriever."""

    text: str
    source: str = "unknown"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...


class NullRetriever:
    """Mặc định khi Chatbot không được cấp retriever — prompt tự bỏ mục Ngữ cảnh."""

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return []


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    signature: str
    func: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools or []}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, signature: str) -> Callable:
        """Decorator: đăng ký hàm thành tool."""

        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name, description, signature, func))
            return func

        return wrap

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def signatures(self) -> list[Tool]:
        """Danh sách đổ vào system prompt (có .name/.description/.signature)."""
        return list(self._tools.values())

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Lỗi: không có tool tên '{name}'. Tool khả dụng: {', '.join(self.names()) or 'không có'}."
        try:
            return str(tool(**args))
        except Exception as exc:  # tool lỗi -> feedback lại cho model tự sửa
            return f"Lỗi khi chạy '{name}': {type(exc).__name__}: {exc}"

    def __len__(self) -> int:
        return len(self._tools)
