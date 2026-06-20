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

from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.confluence import ConfluenceConnector
from nlqueries.document_connectors.excel import ExcelConnector
from nlqueries.document_connectors.notion import NotionConnector
from nlqueries.document_connectors.pdf import PdfConnector
from nlqueries.document_connectors.word import WordConnector

DOCUMENT_CONNECTOR_REGISTRY: dict[str, type[DocumentConnector]] = {
    "pdf": PdfConnector,
    "word": WordConnector,
    "excel": ExcelConnector,
    "notion": NotionConnector,
    "confluence": ConfluenceConnector,
}

__all__ = [
    "DocumentChunk",
    "DocumentConnector",
    "PdfConnector",
    "WordConnector",
    "ExcelConnector",
    "NotionConnector",
    "ConfluenceConnector",
    "DOCUMENT_CONNECTOR_REGISTRY",
]
