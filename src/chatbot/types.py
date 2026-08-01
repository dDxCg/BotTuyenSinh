from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:


    text: str
    source: str = "unknown"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:

        ...


class NullRetriever:


    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return []

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return None


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


        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name, description, signature, func))
            return func

        return wrap

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def signatures(self) -> list[Tool]:

        return list(self._tools.values())

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Lỗi: không có tool tên '{name}'. Tool khả dụng: {', '.join(self.names()) or 'không có'}."
        try:
            return str(tool(**args))
        except Exception as exc:
            return f"Lỗi khi chạy '{name}': {type(exc).__name__}: {exc}"

    def __len__(self) -> int:
        return len(self._tools)
