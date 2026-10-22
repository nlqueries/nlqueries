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
from typing import Any

from nlqueries.document_connectors import _limits
from nlqueries.document_connectors._isolation import IsolatedIngest
from nlqueries.document_connectors._limits import (
    DocumentTooComplexError,
    ExtractionBudget,
    _size_over,
)
from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.chunker import RecursiveCharacterTextSplitter


def _make_chunk_id(source_id: str, page_number: int, chunk_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:page:index)[:16]."""
    raw = f"{source_id}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _declared_page_count(pdf: Any) -> int | None:
    """The page count the document's page tree declares, or ``None``.

    Read from ``/Pages /Count`` in the catalog, which pdfminer has parsed by the
    time ``pdfplumber.open`` returns; no page is built or extracted to learn it.
    Self-reported, like a zip directory, so it is a cheap first gate and the
    pages are counted again before extraction. Anything unreadable returns
    ``None`` and is left to that second count.
    """
    try:
        from pdfminer.pdftypes import resolve1

        count = resolve1(resolve1(pdf.doc.catalog["Pages"])["Count"])
    except Exception:  # noqa: BLE001 -- a malformed tree is the parser's to report
        return None
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def _refuse_page_count(name: str, pages: int) -> None:
    if pages > _limits.PDF_PAGES:
        raise DocumentTooComplexError(
            f"{name} has {pages} pages, over the {_limits.PDF_PAGES}-page limit "
            "for a single document; split it and ingest the parts."
        )


def _refuse_oversized(source_path: Path) -> None:
    """Refuse a PDF on its size on disk, before any parser opens it.

    A file that cannot be stat'ed is left for the parser to report, which names
    the problem better than anything this could.
    """
    try:
        size = source_path.stat().st_size
    except OSError:
        return
    if size > _limits.PDF_BYTES:
        shown, cap = _size_over(size, _limits.PDF_BYTES)
        raise DocumentTooComplexError(
            f"{source_path.name} is {shown}, over the {cap} limit for a PDF."
        )


class PdfConnector(IsolatedIngest, DocumentConnector):
    """Extract and chunk text from PDF files using pdfplumber.

    Each page is independently chunked with
    ``RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)``.
    Empty pages are skipped silently.
    """

    _CHUNK_SIZE = 800
    _CHUNK_OVERLAP = 100

    def _refuse_before_parsing(self, source: Path) -> None:
        # The page count is not checked here: reading it means parsing the file.
        _refuse_oversized(source)

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PdfConnector. "
                "Install it with: pip install 'nlqueries-core[docs]'"
            ) from exc

        source_path = Path(source)
        _refuse_oversized(source_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
        )

        chunks: list[DocumentChunk] = []
        # A PDF is not a zip, so the expansion gate never saw it. Page count is a
        # poor proxy for cost -- a 3,000-page PDF of scanned images and a 30-page
        # one of dense tables differ by orders of magnitude -- so cost is bounded
        # by the clock, checked once per page. The page cap is a separate,
        # deterministic refusal of a document known to be too long.
        budget = ExtractionBudget(name=source_path.name)
        with pdfplumber.open(source_path) as pdf:
            declared = _declared_page_count(pdf)
            if declared is not None:
                _refuse_page_count(source_path.name, declared)
            total_pages = len(pdf.pages)
            _refuse_page_count(source_path.name, total_pages)
            for page in pdf.pages:
                budget.check(f"{page.page_number - 1} of {total_pages} pages")
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
