"""Render system prompt từ template Jinja."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class ToolSignature:
    """Chữ ký tool đổ vào system prompt."""

    name: str
    description: str
    signature: str


def render_system_prompt(
    tool_signatures: Sequence[Any] | None = None,
    retrieved: Sequence[Any] | None = None,
    context: str = "",
    react: bool = False,
    max_steps: int = 6,
    threshold: float = 0.7,
    template: str = "system.j2",
) -> str:
    """Đổ tool signature + chunk RAG + giao thức ReAct vào system prompt.

    `tool_signatures`: object có .name/.description/.signature (ToolSignature hoặc Tool).
    `retrieved`: object có .text/.source/.score/.metadata (chatbot.types.Chunk).
    Kết luận đủ/không đủ căn cứ tính ngay tại đây theo `threshold`, để model
    không phải tự đoán — chunk đứng đầu đã là chunk điểm cao nhất.
    """
    chunks = list(retrieved or [])
    best_score = max((c.score for c in chunks), default=0.0)
    return _env.get_template(template).render(
        tool_signatures=list(tool_signatures or []),
        retrieved=chunks,
        context=context,
        react=react,
        max_steps=max_steps,
        best_score=best_score,
        threshold=threshold,
        grounded=best_score >= threshold,
    )
