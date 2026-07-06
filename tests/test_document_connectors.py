"""
Tests for nlqueries.document_connectors — PdfConnector + WordConnector + registry.

All tests mock heavy dependencies (pdfplumber, python-docx)
so no live files or installed extras are required.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
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
# Fixture: stub out pdfplumber before import
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_heavy_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight stub for pdfplumber so the test suite runs
    without the [docs] extras installed."""

    # --- pdfplumber stub ---
    pdfplumber_mod = MagicMock()

    def _open_pdf(path: Any) -> MagicMock:
        # The real open() call will be patched per-test; this stub makes
        # the module importable.
        return MagicMock()

    pdfplumber_mod.open = _open_pdf
    monkeypatch.setitem(sys.modules, "pdfplumber", pdfplumber_mod)


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
# Fixture: stub out python-docx for Word tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _stub_word_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight stub for python-docx."""

    # --- python-docx stub ---
    docx_mod = MagicMock()

    def _document(path: str) -> MagicMock:
        # Per-test patch replaces this; stub makes the module importable.
        return MagicMock()

    docx_mod.Document = _document
    monkeypatch.setitem(sys.modules, "docx", docx_mod)


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


# ===========================================================================
# ExcelConnector tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers — build in-memory openpyxl sheet mocks without a real file
# ---------------------------------------------------------------------------


def _make_mock_sheet(title: str, rows: list[tuple[Any, ...]]) -> MagicMock:
    sheet = MagicMock()
    sheet.title = title
    sheet.iter_rows = MagicMock(return_value=iter(rows))
    return sheet


def _make_mock_workbook(sheets: list[MagicMock]) -> MagicMock:
    wb = MagicMock()
    wb.worksheets = sheets
    wb.close = MagicMock()
    return wb


# ---------------------------------------------------------------------------
# Fixture: stub out openpyxl for Excel tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _stub_excel_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight stub for openpyxl so tests run without the [docs] extras."""
    openpyxl_mod = MagicMock()
    openpyxl_mod.load_workbook = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "openpyxl", openpyxl_mod)


# ---------------------------------------------------------------------------
# test_excel_connector_chunks_each_sheet
# ---------------------------------------------------------------------------


def test_excel_connector_chunks_each_sheet(_stub_excel_deps: None) -> None:
    """A workbook with 2 sheets produces chunks from both sheets with page_number 1 and 2."""
    from nlqueries.document_connectors.excel import ExcelConnector

    sheet1 = _make_mock_sheet(
        "Sales",
        [
            ("Product", "Revenue", "Units"),  # header row
            ("Widget A", 1000, 50),
            ("Widget B", 2000, 100),
        ],
    )
    sheet2 = _make_mock_sheet(
        "Inventory",
        [
            ("Item", "Stock"),  # header row
            ("Widget A", 200),
            ("Widget B", 150),
        ],
    )
    mock_wb = _make_mock_workbook([sheet1, sheet2])

    with patch("openpyxl.load_workbook", return_value=mock_wb):
        connector = ExcelConnector()
        chunks = connector.ingest(Path("data.xlsx"), source_id="excel-src-001")

    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers

    sheet_names = {c.metadata["sheet_name"] for c in chunks}
    assert "Sales" in sheet_names
    assert "Inventory" in sheet_names

    for chunk in chunks:
        assert chunk.source_id == "excel-src-001"
        assert chunk.source_name == "data.xlsx"
        assert chunk.metadata["connector"] == "excel"
        assert "row_range" in chunk.metadata
        assert "file_path" in chunk.metadata


# ---------------------------------------------------------------------------
# test_excel_row_batch_produces_text
# ---------------------------------------------------------------------------


def test_excel_row_batch_produces_text(_stub_excel_deps: None) -> None:
    """Batch text contains column header names and row values."""
    from nlqueries.document_connectors.excel import ExcelConnector

    sheet = _make_mock_sheet(
        "People",
        [
            ("Name", "Age", "City"),  # header row
            ("Alice", 30, "New York"),
            ("Bob", 25, "Chicago"),
        ],
    )
    mock_wb = _make_mock_workbook([sheet])

    with patch("openpyxl.load_workbook", return_value=mock_wb):
        connector = ExcelConnector()
        chunks = connector.ingest(Path("people.xlsx"), source_id="excel-src-002")

    assert len(chunks) == 1
    text = chunks[0].text

    # Column headers must appear as prefixes in the row text
    assert "Name" in text
    assert "Age" in text
    assert "City" in text

    # Row values must appear
    assert "Alice" in text
    assert "Bob" in text
    assert "New York" in text
    assert "Chicago" in text

    # row_range should reflect the data rows (not the header)
    assert chunks[0].metadata["row_range"] == "2-3"
    assert chunks[0].page_number == 1


# ---------------------------------------------------------------------------
# test_excel_supports_xlsx_only
# ---------------------------------------------------------------------------


def test_excel_supports_xlsx_only() -> None:
    """.xlsx is accepted; .xls and .csv are rejected."""
    from nlqueries.document_connectors.excel import ExcelConnector

    connector = ExcelConnector()

    assert connector.supports(Path("data.xlsx")) is True
    assert connector.supports("UPPER.XLSX") is True  # case-insensitive
    assert connector.supports(Path("data.xls")) is False
    assert connector.supports(Path("data.csv")) is False
    assert connector.supports(Path("data.docx")) is False
    assert connector.supports(Path("data.pdf")) is False


# ---------------------------------------------------------------------------
# test_document_connector_registry_contains_excel
# ---------------------------------------------------------------------------


def test_document_connector_registry_contains_excel() -> None:
    """DOCUMENT_CONNECTOR_REGISTRY must have an 'excel' key pointing to ExcelConnector."""
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY
    from nlqueries.document_connectors.excel import ExcelConnector

    assert "excel" in DOCUMENT_CONNECTOR_REGISTRY
    assert DOCUMENT_CONNECTOR_REGISTRY["excel"] is ExcelConnector


# ===========================================================================
# NotionConnector tests
# ===========================================================================


@pytest.fixture()
def _stub_notion_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight stub for notion_client so tests run without the [wiki] extra."""
    notion_mod = MagicMock()
    monkeypatch.setitem(sys.modules, "notion_client", notion_mod)


def _make_notion_page(
    last_edited_time: str = "2024-06-01T10:00:00.000Z",
    title_text: str = "Test Page",
) -> dict[str, Any]:
    return {
        "last_edited_time": last_edited_time,
        "properties": {
            "Name": {
                "type": "title",
                "title": [{"plain_text": title_text}],
            }
        },
    }


def _make_blocks_response(
    texts: list[str],
    has_more: bool = False,
) -> dict[str, Any]:
    results = [
        {
            "type": "paragraph",
            "paragraph": {"rich_text": [{"plain_text": t}]},
        }
        for t in texts
    ]
    return {"results": results, "has_more": has_more, "next_cursor": None}


# ---------------------------------------------------------------------------
# test_notion_connector_chunks_page_blocks
# ---------------------------------------------------------------------------


def test_notion_connector_chunks_page_blocks(_stub_notion_deps: None) -> None:
    """Blocks from a Notion page are concatenated and split into chunks."""
    from nlqueries.document_connectors.notion import NotionConnector

    mock_client = MagicMock()
    mock_client.pages.retrieve.return_value = _make_notion_page()
    mock_client.blocks.children.list.return_value = _make_blocks_response(
        [
            "Introduction paragraph. " * 10,
            "Background section content. " * 10,
            "Conclusion and summary. " * 10,
        ]
    )

    with patch("notion_client.Client", return_value=mock_client):
        connector = NotionConnector(api_token="test-token")
        chunks = connector.ingest("page-abc-123", source_id="src-notion-001")

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.source_id == "src-notion-001"
        assert chunk.source_name == "page-abc-123"
        assert chunk.page_number is None
        assert chunk.metadata["connector"] == "notion"
        assert chunk.metadata["page_id"] == "page-abc-123"


# ---------------------------------------------------------------------------
# test_incremental_sync_filters_by_last_edited_time
# ---------------------------------------------------------------------------


def test_incremental_sync_filters_by_last_edited_time(_stub_notion_deps: None) -> None:
    """ingest() returns [] when the page's last_edited_time is not after `since`."""
    from nlqueries.document_connectors.notion import NotionConnector

    # Page was last edited on 2024-01-10 — before since=2024-01-15.
    mock_client = MagicMock()
    mock_client.pages.retrieve.return_value = _make_notion_page(
        last_edited_time="2024-01-10T10:00:00.000Z"
    )

    since = datetime(2024, 1, 15, tzinfo=UTC)

    with patch("notion_client.Client", return_value=mock_client):
        connector = NotionConnector(api_token="test-token")
        chunks = connector.ingest("page-abc-123", source_id="src-notion-002", since=since)

    assert chunks == []
    mock_client.blocks.children.list.assert_not_called()


# ---------------------------------------------------------------------------
# test_page_title_in_metadata
# ---------------------------------------------------------------------------


def test_page_title_in_metadata(_stub_notion_deps: None) -> None:
    """Page title extracted from Notion properties appears in every chunk's metadata."""
    from nlqueries.document_connectors.notion import NotionConnector

    mock_client = MagicMock()
    mock_client.pages.retrieve.return_value = _make_notion_page(title_text="My Engineering Runbook")
    mock_client.blocks.children.list.return_value = _make_blocks_response(
        ["Step 1: do something. " * 5]
    )

    with patch("notion_client.Client", return_value=mock_client):
        connector = NotionConnector(api_token="test-token")
        chunks = connector.ingest("page-runbook-001", source_id="src-notion-003")

    assert len(chunks) >= 1
    assert chunks[0].metadata["page_title"] == "My Engineering Runbook"
    assert chunks[0].metadata["last_edited_time"] == "2024-06-01T10:00:00.000Z"


# ---------------------------------------------------------------------------
# test_document_connector_registry_contains_notion
# ---------------------------------------------------------------------------


def test_document_connector_registry_contains_notion() -> None:
    """DOCUMENT_CONNECTOR_REGISTRY must have a 'notion' key pointing to NotionConnector."""
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY
    from nlqueries.document_connectors.notion import NotionConnector

    assert "notion" in DOCUMENT_CONNECTOR_REGISTRY
    assert DOCUMENT_CONNECTOR_REGISTRY["notion"] is NotionConnector


# ===========================================================================
# ConfluenceConnector tests
# ===========================================================================


@pytest.fixture()
def _stub_confluence_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject lightweight stubs for atlassian and bs4 so tests run without [wiki] extras."""

    # --- atlassian stub ---
    atlassian_mod = MagicMock()
    monkeypatch.setitem(sys.modules, "atlassian", atlassian_mod)

    # --- bs4 stub with a minimal BeautifulSoup ---
    bs4_mod = MagicMock()

    class _FakeSoup:
        def __init__(self, markup: str, parser: str = "html.parser") -> None:
            self._markup = markup

        def get_text(self, separator: str = "", strip: bool = False) -> str:
            import re

            text = re.sub(r"<[^>]+>", separator, self._markup)
            if strip:
                text = text.strip()
            return text

    bs4_mod.BeautifulSoup = _FakeSoup
    monkeypatch.setitem(sys.modules, "bs4", bs4_mod)


def _make_cql_result(page_id: str, title: str, web_ui: str) -> dict[str, Any]:
    return {
        "content": {
            "id": page_id,
            "title": title,
            "_links": {"webui": web_ui},
        }
    }


def _make_page_body(html: str) -> dict[str, Any]:
    return {"body": {"storage": {"value": html}}}


# ---------------------------------------------------------------------------
# test_confluence_connector_fetches_all_space_pages
# ---------------------------------------------------------------------------


def test_confluence_connector_fetches_all_space_pages(_stub_confluence_deps: None) -> None:
    """CQL results are iterated and each page produces chunks with correct metadata."""
    from nlqueries.document_connectors.confluence import ConfluenceConnector

    mock_client = MagicMock()
    mock_client.cql.return_value = {
        "results": [
            _make_cql_result("111", "Page One", "/spaces/ENG/pages/111"),
            _make_cql_result("222", "Page Two", "/spaces/ENG/pages/222"),
        ]
    }
    mock_client.get_page_by_id.side_effect = [
        _make_page_body("<p>Content of page one. " + "word " * 20 + "</p>"),
        _make_page_body("<p>Content of page two. " + "word " * 20 + "</p>"),
    ]

    with patch("atlassian.Confluence", return_value=mock_client):
        connector = ConfluenceConnector(
            base_url="https://acme.atlassian.net",
            username="alice@acme.com",
            api_token="test-token",
        )
        chunks = connector.ingest("ENG", source_id="src-conf-001")

    assert len(chunks) >= 2
    page_ids_seen = {c.metadata["page_id"] for c in chunks}
    assert "111" in page_ids_seen
    assert "222" in page_ids_seen

    for chunk in chunks:
        assert chunk.source_id == "src-conf-001"
        assert chunk.source_name == "ENG"
        assert chunk.page_number is None
        assert chunk.metadata["connector"] == "confluence"
        assert chunk.metadata["space_key"] == "ENG"


# ---------------------------------------------------------------------------
# test_html_stripped_from_body
# ---------------------------------------------------------------------------


def test_html_stripped_from_body(_stub_confluence_deps: None) -> None:
    """HTML tags are stripped so only plain text appears in chunk content."""
    from nlqueries.document_connectors.confluence import ConfluenceConnector

    mock_client = MagicMock()
    mock_client.cql.return_value = {
        "results": [_make_cql_result("999", "Tech Spec", "/spaces/ENG/pages/999")]
    }
    mock_client.get_page_by_id.return_value = _make_page_body(
        "<h1>Overview</h1><p>This is the <strong>important</strong> section.</p>"
    )

    with patch("atlassian.Confluence", return_value=mock_client):
        connector = ConfluenceConnector(
            base_url="https://acme.atlassian.net",
            username="alice@acme.com",
            api_token="test-token",
        )
        chunks = connector.ingest("ENG", source_id="src-conf-002")

    assert chunks, "Expected at least one chunk from non-empty page"
    combined = " ".join(c.text for c in chunks)
    # HTML tags must not appear in chunk text
    assert "<h1>" not in combined
    assert "<p>" not in combined
    assert "<strong>" not in combined
    # Prose content must be present
    assert "Overview" in combined or "important" in combined or "section" in combined


# ---------------------------------------------------------------------------
# test_incremental_sync_adds_cql_date_filter
# ---------------------------------------------------------------------------


def test_incremental_sync_adds_cql_date_filter(_stub_confluence_deps: None) -> None:
    """Passing `since` includes lastModified > date filter in the CQL query."""
    from datetime import UTC

    from nlqueries.document_connectors.confluence import ConfluenceConnector

    mock_client = MagicMock()
    mock_client.cql.return_value = {"results": []}

    since = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)

    with patch("atlassian.Confluence", return_value=mock_client):
        connector = ConfluenceConnector(
            base_url="https://acme.atlassian.net",
            username="alice@acme.com",
            api_token="test-token",
        )
        chunks = connector.ingest("ENG", source_id="src-conf-003", since=since)

    assert chunks == []
    call_args = mock_client.cql.call_args
    cql_query: str = call_args[0][0] if call_args[0] else call_args[1].get("cql", "")
    assert "lastModified" in cql_query
    assert "2024-03-15" in cql_query


# ---------------------------------------------------------------------------
# test_document_connector_registry_contains_confluence
# ---------------------------------------------------------------------------


def test_document_connector_registry_contains_confluence() -> None:
    """DOCUMENT_CONNECTOR_REGISTRY must have a 'confluence' key pointing to ConfluenceConnector."""
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY
    from nlqueries.document_connectors.confluence import ConfluenceConnector

    assert "confluence" in DOCUMENT_CONNECTOR_REGISTRY
    assert DOCUMENT_CONNECTOR_REGISTRY["confluence"] is ConfluenceConnector
