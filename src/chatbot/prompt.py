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


    name: str
    description: str
    signature: str


def render_admission_policy(threshold: float = 0.7) -> str:

    return _env.get_template("admission_policy.md").render(threshold=threshold)


def render_system_prompt(
    tool_signatures: Sequence[Any] | None = None,
    retrieved: Sequence[Any] | None = None,
    context: str = "",
    react: bool = False,
    max_steps: int = 6,
    threshold: float = 0.7,
    template: str = "system.j2",
) -> str:

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
