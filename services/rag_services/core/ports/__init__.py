"""Core ports (interfaces) for GraphRAG"""

from .graph_repository import (
    GraphRepository,
    GraphRepositoryError,
    NodeNotFoundError,
    DuplicateNodeError,
    InvalidQueryError,
    ConnectionError,
)

__all__ = [
    "GraphRepository",
    "GraphRepositoryError",
    "NodeNotFoundError",
    "DuplicateNodeError",
    "InvalidQueryError",
    "ConnectionError",
]
