from src.rag.types import Retriever

from ..types import ToolRegistry
from .attach_source_link_tool import make_attach_source_link
from .contact_support_tool import make_contact_support


def build_registry(retriever: Retriever) -> ToolRegistry:
    return ToolRegistry(
        [
            make_attach_source_link(retriever),
            make_contact_support(),
        ]
    )


__all__ = ["build_registry", "make_attach_source_link", "make_contact_support"]
