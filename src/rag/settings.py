import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class RagSettings:
    no_grounding_threshold: float

    @classmethod
    def from_env(cls) -> "RagSettings":
        return cls(
            no_grounding_threshold=float(os.getenv("GROUNDING_THRESHOLD", "0.7")),
        )


__all__ = ["RagSettings"]
