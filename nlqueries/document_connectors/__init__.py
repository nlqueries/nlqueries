"""
nlqueries.document_connectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Document connector registry for nlqueries-core.

Usage:
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY
    connector_cls = DOCUMENT_CONNECTOR_REGISTRY["pdf"]
    connector = connector_cls()
    chunks = connector.ingest("/path/to/file.pdf", source_id="my-doc-uuid")
"""

from nlqueries.document_connectors._limits import (
    DocumentExtractionTimeout,
    DocumentTooComplexError,
)
from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.confluence import ConfluenceConnector
from nlqueries.document_connectors.excel import ExcelConnector
from nlqueries.document_connectors.notion import NotionConnector
from nlqueries.document_connectors.pdf import PdfConnector
from nlqueries.document_connectors.text import MarkdownConnector, TextConnector
from nlqueries.document_connectors.word import WordConnector

DOCUMENT_CONNECTOR_REGISTRY: dict[str, type[DocumentConnector]] = {
    "pdf": PdfConnector,
    "word": WordConnector,
    "excel": ExcelConnector,
    "notion": NotionConnector,
    "confluence": ConfluenceConnector,
    "markdown": MarkdownConnector,
    "text": TextConnector,
}

#: `_limits` is private -- the module name says so and it is free to be
#: renamed. The two exceptions it defines are not private: a caller has to name
#: them to handle a refused document, and enterprise's ingestion task does. They
#: are re-exported here so that dependency sits on a declared surface, which is
#: what the `CORE_REF`-bump-with-the-change convention assumes it can rely on.
__all__ = [
    "DocumentChunk",
    "DocumentExtractionTimeout",
    "DocumentTooComplexError",
    "DocumentConnector",
    "PdfConnector",
    "WordConnector",
    "ExcelConnector",
    "NotionConnector",
    "ConfluenceConnector",
    "MarkdownConnector",
    "TextConnector",
    "DOCUMENT_CONNECTOR_REGISTRY",
]
