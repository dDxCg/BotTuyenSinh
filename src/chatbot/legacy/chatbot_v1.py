"""Bản Chatbot cũ (trước khi chuyển hẳn sang LangGraph agent trong `graph/`).

Không còn được gọi ở đâu trong code sống (`graph/llm.py` chỉ dùng `Chatbot.complete_with_tools`
ở `chatbot.py` rút gọn) — giữ lại đây để tham khảo khi cần đối chiếu hành vi cũ, không import
vào code chạy thật.
"""

from collections.abc import Iterator, Sequence
from typing import Any

from src.llm_client import chat_client
from src.rag.types import Chunk, NullRetriever, Retriever

from ..config import Settings
from ..prompts import ToolSignature, render_system_prompt


class ChatbotV1:
    def __init__(
        self,
        settings: Settings | None = None,
        tool_signatures: Sequence[Any] | None = None,
        retriever: Retriever | None = None,
        context: str = "",
        top_k: int | None = None,
        grounding_threshold: float = 0.7,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.tool_signatures: list[Any] = list(tool_signatures or [])
        self.retriever: Retriever = retriever or NullRetriever()
        self.context = context
        self.top_k = top_k if top_k is not None else self.settings.top_k
        self.grounding_threshold = grounding_threshold
        self.client = chat_client(
            self.settings.api_key,
            self.settings.base_url,
            self.settings.timeout_seconds,
            self.settings.max_retries,
        )
        self.history: list[dict[str, str]] = []
        self.last_retrieved: list[Chunk] = []

    def retrieve(self, query: str) -> list[Chunk]:
        self.last_retrieved = self.retriever.retrieve(query, k=self.top_k)
        return self.last_retrieved

    def system_prompt(
        self,
        query: str,
        react: bool = False,
        max_steps: int = 6,
        prefetch: bool = True,
    ) -> str:
        return render_system_prompt(
            tool_signatures=self.tool_signatures,
            retrieved=self.retrieve(query) if prefetch else [],
            context=self.context,
            react=react,
            max_steps=max_steps,
            threshold=self.grounding_threshold,
        )

    def _messages(self, user_message: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt(user_message)},
            *self.history,
            {"role": "user", "content": user_message},
        ]

    def complete(
        self,
        messages: list[dict[str, str]],
        stop: list[str] | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content or ""

    def chat(self, user_message: str) -> str:
        reply = self.complete(self._messages(user_message))
        self.remember(user_message, reply)
        return reply

    def chat_with_retrieved(self, user_message: str, retrieved: Sequence[Chunk]) -> str:
        self.last_retrieved = list(retrieved)
        system = render_system_prompt(
            tool_signatures=self.tool_signatures,
            retrieved=self.last_retrieved,
            context=self.context,
        )
        messages = [
            {"role": "system", "content": system},
            *self.history,
            {"role": "user", "content": user_message},
        ]
        reply = self.complete(messages)
        self.remember(user_message, reply)
        return reply

    def stream(self, user_message: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.settings.model,
            messages=self._messages(user_message),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stream=True,
        )
        chunks: list[str] = []
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                chunks.append(delta)
                yield delta
        self.remember(user_message, "".join(chunks))

    def remember(self, user_message: str, reply: str) -> None:
        self.history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]

    def reset(self) -> None:
        self.history.clear()
        self.last_retrieved = []


__all__ = ["ChatbotV1", "ToolSignature"]
