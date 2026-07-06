"""
nlqueries.document_connectors.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PDF document connector using ``pdfplumber`` for text extraction and a
built-in recursive character chunker.

Requires the ``docs`` optional dependency group:
    pip install "nlqueries-core[docs]"
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.chunker import RecursiveCharacterTextSplitter


def _make_chunk_id(source_id: str, page_number: int, chunk_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:page:index)[:16]."""
    raw = f"{source_id}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class PdfConnector(DocumentConnector):
    """Extract and chunk text from PDF files using pdfplumber.

    Each page is independently chunked with
    ``RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)``.
    Empty pages are skipped silently.
    """

    _CHUNK_SIZE = 800
    _CHUNK_OVERLAP = 100

    def ingest(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PdfConnector. "
                "Install it with: pip install 'nlqueries-core[docs]'"
            ) from exc

        source_path = Path(source)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
        )

        chunks: list[DocumentChunk] = []
        with pdfplumber.open(source_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                page_number = page.page_number  # 1-based in pdfplumber
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                page_chunks = splitter.split_text(text)
                for chunk_index, chunk_text in enumerate(page_chunks):
                    chunk_id = _make_chunk_id(source_id, page_number, chunk_index)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_id=source_id,
                            source_name=source_path.name,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            text=chunk_text,
                            metadata={
                                "connector": "pdf",
                                "file_path": str(source_path),
                                "total_pages": total_pages,
                            },
                        )
                    )

        return chunks

    def supports(self, source: str | Path) -> bool:
        return Path(source).suffix.lower() == ".pdf"
