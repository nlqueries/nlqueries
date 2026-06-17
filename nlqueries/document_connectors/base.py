"""
nlqueries.document_connectors.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines the public document connector interface for nlqueries-core.

Every document integration (PDF, Word, Excel, Notion, Confluence, …) implements
``DocumentConnector``. This is parallel to ``DatabaseConnector`` — both hierarchies
are independent; neither imports from the other.

This module is part of the public OSS API and has no dependency on the enterprise layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    """A single text chunk extracted from a document source.

    ``chunk_id`` is deterministic: sha256(f"{source_id}:{page_number}:{chunk_index}")[:16]
    so the same document always produces the same IDs and re-ingestion is idempotent.
    """

    chunk_id: str
    source_id: str
    source_name: str
    page_number: int | None
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentConnector(ABC):
    """Abstract base class for all document connectors.

    Concrete subclasses implement ``ingest()`` and ``supports()``.  The rest of
    nlqueries-core (embeddings, CLI, orchestrator) is written against this interface
    so it works with any document source without knowing the underlying library.
    """

    @abstractmethod
    def ingest(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        """Read *source* and return a list of text chunks ready for embedding.

        Args:
            source: File path or URL identifying the document.
            source_id: Opaque identifier for the parent document/source (used
                       to generate deterministic chunk IDs and for Qdrant filtering).

        Returns:
            Ordered list of :class:`DocumentChunk` objects.
        """
        ...

    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        """Return ``True`` if this connector can handle *source*.

        Typically checks the file extension (e.g. ``.pdf``) or protocol scheme.
        """
        ...
