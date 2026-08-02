from typing import Any

from src.llm_client import chat_client

from .config import Settings


class Chatbot:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.client = chat_client(
            self.settings.api_key,
            self.settings.base_url,
            self.settings.timeout_seconds,
            self.settings.max_retries,
        )

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            tools=tools or None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        return response.choices[0].message


__all__ = ["Chatbot"]
