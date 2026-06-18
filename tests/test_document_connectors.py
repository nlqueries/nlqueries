"""
Tests for nlqueries.document_connectors — PdfConnector + WordConnector + registry.

All tests mock heavy dependencies (pdfplumber, langchain_text_splitters, python-docx)
so no live files or installed extras are required.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_expected_chunk_id(source_id: str, page_number: int, chunk_index: int) -> str:
    raw = f"{source_id}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_mock_page(page_number: int, text: str) -> MagicMock:
    page = MagicMock()
    page.page_number = page_number
    page.extract_text.return_value = text
    return page


def _make_mock_pdf(pages: list[MagicMock]) -> MagicMock:
    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    return pdf


# ---------------------------------------------------------------------------
# Fixture: stub out pdfplumber and langchain_text_splitters before import
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_heavy_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject lightweight stubs for pdfplumber and langchain_text_splitters
    so the test suite runs without the [docs] extras installed."""

    # --- pdfplumber stub ---
    pdfplumber_mod = MagicMock()

    def _open_pdf(path: Any) -> MagicMock:
        # The real open() call will be patched per-test; this stub makes
        # the module importable.
        return MagicMock()

    pdfplumber_mod.open = _open_pdf
    monkeypatch.setitem(sys.modules, "pdfplumber", pdfplumber_mod)

    # --- langchain_text_splitters stub ---
    splitter_mod = MagicMock()

    class _FakeSplitter:
        def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
            self._size = chunk_size

        def split_text(self, text: str) -> list[str]:
            # Naive split: one chunk per _size chars (matches production intent)
            if not text:
                return []
            return [text[i : i + self._size] for i in range(0, len(text), self._size)]

    splitter_mod.RecursiveCharacterTextSplitter = _FakeSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitter_mod)


# ---------------------------------------------------------------------------
# test_pdf_connector_chunks_page_correctly
# ---------------------------------------------------------------------------


def test_pdf_connector_chunks_page_correctly() -> None:
    """A 3-page PDF produces at least 3 chunks, one per page, with correct page_number."""
    from nlqueries.document_connectors.pdf import PdfConnector

    pages = [
        _make_mock_page(1, "Content of page one. " * 20),
        _make_mock_page(2, "Content of page two. " * 20),
        _make_mock_page(3, "Content of page three. " * 20),
    ]
    mock_pdf = _make_mock_pdf(pages)

    with patch("pdfplumber.open", return_value=mock_pdf):
        connector = PdfConnector()
        chunks = connector.ingest(Path("fake.pdf"), source_id="test-src-001")

    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    page_numbers_seen = {c.page_number for c in chunks}
    assert page_numbers_seen == {1, 2, 3}

    for chunk in chunks:
        assert chunk.source_id == "test-src-001"
        assert chunk.source_name == "fake.pdf"
        assert chunk.metadata["connector"] == "pdf"
        assert "total_pages" in chunk.metadata


# ---------------------------------------------------------------------------
# test_chunk_id_is_deterministic
# ---------------------------------------------------------------------------


def test_chunk_id_is_deterministic() -> None:
    """The same input always produces the same chunk_id."""
    from nlqueries.document_connectors.pdf import PdfConnector

    page_text = "Deterministic content. " * 10
    pages = [_make_mock_page(1, page_text)]
    mock_pdf = _make_mock_pdf(pages)

    with patch("pdfplumber.open", return_value=mock_pdf):
        connector = PdfConnector()
        chunks_first = connector.ingest(Path("report.pdf"), source_id="src-abc")

    mock_pdf2 = _make_mock_pdf([_make_mock_page(1, page_text)])
    with patch("pdfplumber.open", return_value=mock_pdf2):
        chunks_second = connector.ingest(Path("report.pdf"), source_id="src-abc")

    assert len(chunks_first) == len(chunks_second)
    for a, b in zip(chunks_first, chunks_second, strict=True):
        assert a.chunk_id == b.chunk_id

    # Also verify the ID matches the documented formula
    first_chunk = chunks_first[0]
    expected_id = _make_expected_chunk_id("src-abc", 1, 0)
    assert first_chunk.chunk_id == expected_id


# ---------------------------------------------------------------------------
# test_supports_returns_true_for_pdf
# ---------------------------------------------------------------------------


def test_supports_returns_true_for_pdf() -> None:
    """supports() returns True for .pdf paths and False for everything else."""
    from nlqueries.document_connectors.pdf import PdfConnector

    connector = PdfConnector()

    assert connector.supports(Path("document.pdf")) is True
    assert connector.supports("UPPER.PDF") is True  # case-insensitive
    assert connector.supports(Path("report.docx")) is False
    assert connector.supports(Path("data.xlsx")) is False
    assert connector.supports(Path("notes.txt")) is False
    assert connector.supports(Path("archive.pdf.gz")) is False


# ---------------------------------------------------------------------------
# test_document_connector_registry_contains_pdf
# ---------------------------------------------------------------------------


def test_document_connector_registry_contains_pdf() -> None:
    """DOCUMENT_CONNECTOR_REGISTRY must have a 'pdf' key pointing to PdfConnector."""
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY
    from nlqueries.document_connectors.pdf import PdfConnector

    assert "pdf" in DOCUMENT_CONNECTOR_REGISTRY
    assert DOCUMENT_CONNECTOR_REGISTRY["pdf"] is PdfConnector


# ---------------------------------------------------------------------------
# test_empty_pages_are_skipped
# ---------------------------------------------------------------------------


def test_empty_pages_are_skipped() -> None:
    """Pages with no extractable text produce no chunks."""
    from nlqueries.document_connectors.pdf import PdfConnector

    pages = [
        _make_mock_page(1, "Real content on page one. " * 10),
        _make_mock_page(2, ""),  # blank / scanned page
        _make_mock_page(3, "   "),  # whitespace-only
    ]
    mock_pdf = _make_mock_pdf(pages)

    with patch("pdfplumber.open", return_value=mock_pdf):
        connector = PdfConnector()
        chunks = connector.ingest(Path("mixed.pdf"), source_id="src-mixed")

    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 not in page_numbers
    assert 3 not in page_numbers


# ---------------------------------------------------------------------------
# test_document_chunk_metadata_fields
# ---------------------------------------------------------------------------


def test_document_chunk_metadata_fields() -> None:
    """Each chunk carries connector, file_path, and total_pages in metadata."""
    from nlqueries.document_connectors.pdf import PdfConnector

    pages = [_make_mock_page(1, "Some text content. " * 5)]
    mock_pdf = _make_mock_pdf(pages)
    source_path = Path("/tmp/annual-report.pdf")

    with patch("pdfplumber.open", return_value=mock_pdf):
        connector = PdfConnector()
        chunks = connector.ingest(source_path, source_id="rpt-2024")

    assert chunks, "Expected at least one chunk"
    meta = chunks[0].metadata
    assert meta["connector"] == "pdf"
    assert meta["file_path"] == str(source_path)
    assert meta["total_pages"] == 1


# ===========================================================================
# WordConnector tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers — build in-memory python-docx paragraphs without a real file
# ---------------------------------------------------------------------------


def _make_mock_paragraph(text: str, style_name: str = "Normal") -> MagicMock:
    para = MagicMock()
    para.text = text
    style = MagicMock()
    style.name = style_name
    para.style = style
    return para


def _make_mock_doc(paragraphs: list[MagicMock]) -> MagicMock:
    doc = MagicMock()
    doc.paragraphs = paragraphs
    return doc


# ---------------------------------------------------------------------------
# Fixture: stub out python-docx and langchain_text_splitters for Word tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _stub_word_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject lightweight stubs for python-docx and langchain_text_splitters."""

    # --- python-docx stub ---
    docx_mod = MagicMock()

    def _document(path: str) -> MagicMock:
        # Per-test patch replaces this; stub makes the module importable.
        return MagicMock()

    docx_mod.Document = _document
    monkeypatch.setitem(sys.modules, "docx", docx_mod)

    # --- langchain_text_splitters (reuse the same _FakeSplitter) ---
    splitter_mod = sys.modules.get("langchain_text_splitters", MagicMock())

    class _FakeSplitter:
        def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
            self._size = chunk_size

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []
            return [text[i : i + self._size] for i in range(0, len(text), self._size)]

    splitter_mod.RecursiveCharacterTextSplitter = _FakeSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitter_mod)


# ---------------------------------------------------------------------------
# test_word_connector_chunks_by_heading
# ---------------------------------------------------------------------------


def test_word_connector_chunks_by_heading(_stub_word_deps: None) -> None:
    """A document with 3 headings produces 3 sections with correct section_heading metadata."""
    from nlqueries.document_connectors.word import WordConnector

    paragraphs = [
        _make_mock_paragraph("Introduction", "Heading 1"),
        _make_mock_paragraph("Intro body text. " * 5),
        _make_mock_paragraph("Background", "Heading 2"),
        _make_mock_paragraph("Background body text. " * 5),
        _make_mock_paragraph("Conclusion", "Heading 1"),
        _make_mock_paragraph("Conclusion body text. " * 5),
    ]
    mock_doc = _make_mock_doc(paragraphs)

    with patch("docx.Document", return_value=mock_doc):
        connector = WordConnector()
        chunks = connector.ingest(Path("report.docx"), source_id="word-src-001")

    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    headings_seen = [c.metadata["section_heading"] for c in chunks]
    assert "Introduction" in headings_seen
    assert "Background" in headings_seen
    assert "Conclusion" in headings_seen

    for chunk in chunks:
        assert chunk.source_id == "word-src-001"
        assert chunk.source_name == "report.docx"
        assert chunk.page_number is None
        assert chunk.metadata["connector"] == "word"


# ---------------------------------------------------------------------------
# test_word_connector_no_headings_produces_chunks
# ---------------------------------------------------------------------------


def test_word_connector_no_headings_produces_chunks(_stub_word_deps: None) -> None:
    """A plain-text document with no headings produces at least 1 chunk."""
    from nlqueries.document_connectors.word import WordConnector

    paragraphs = [
        _make_mock_paragraph("First paragraph. " * 10),
        _make_mock_paragraph("Second paragraph. " * 10),
        _make_mock_paragraph("Third paragraph. " * 10),
    ]
    mock_doc = _make_mock_doc(paragraphs)

    with patch("docx.Document", return_value=mock_doc):
        connector = WordConnector()
        chunks = connector.ingest(Path("plain.docx"), source_id="word-src-002")

    assert len(chunks) >= 1, "Expected at least one chunk for a plain-text document"
    assert chunks[0].metadata["section_heading"] == "untitled"
    assert chunks[0].page_number is None


# ---------------------------------------------------------------------------
# test_word_supports_docx_only
# ---------------------------------------------------------------------------


def test_word_supports_docx_only() -> None:
    """.docx is accepted; .pdf and .doc (old binary format) are rejected."""
    from nlqueries.document_connectors.word import WordConnector

    connector = WordConnector()

    assert connector.supports(Path("report.docx")) is True
    assert connector.supports("UPPER.DOCX") is True  # case-insensitive
    assert connector.supports(Path("report.pdf")) is False
    assert connector.supports(Path("report.doc")) is False
    assert connector.supports(Path("data.xlsx")) is False
    assert connector.supports(Path("notes.txt")) is False


# ---------------------------------------------------------------------------
# test_word_connector_registry_contains_word
# ---------------------------------------------------------------------------


def test_word_connector_registry_contains_word() -> None:
    """DOCUMENT_CONNECTOR_REGISTRY must have a 'word' key pointing to WordConnector."""
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY
    from nlqueries.document_connectors.word import WordConnector

    assert "word" in DOCUMENT_CONNECTOR_REGISTRY
    assert DOCUMENT_CONNECTOR_REGISTRY["word"] is WordConnector
