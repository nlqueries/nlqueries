"""Bounds on document and wiki ingestion.

The PDF, Word and archive cases use the real parsers, for the reason given in
``test_document_expansion_limits.py``: a limit on what a parser consumes is only
meaningful against the parser. The wiki connectors talk to a remote API, so their
clients are stubbed and the stub records what the connector asked for.

Every limit is lowered for the test rather than a document made large enough to
reach the default: the behaviour under test is the comparison.
"""

from __future__ import annotations

import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from nlqueries.document_connectors import _limits
from nlqueries.document_connectors._limits import (
    DocumentTooComplexError,
    check_archive_expansion,
)


def _pdf(path: Path, pages: int, declared: int | None = None) -> Path:
    """A syntactically valid PDF with *pages* pages, written by hand.

    *declared* is the ``/Count`` the page tree claims, which defaults to the
    truth; setting it apart from *pages* makes a document that misstates itself.
    """
    objects: list[bytes] = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    count = pages if declared is None else declared
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode())
    for _ in range(pages):
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_an_oversized_pdf_is_refused_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdfplumber = pytest.importorskip("pdfplumber")
    from nlqueries.document_connectors.pdf import PdfConnector

    path = _pdf(tmp_path / "big.pdf", pages=2)
    monkeypatch.setattr(_limits, "PDF_BYTES", path.stat().st_size - 1)

    def _must_not_be_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("pdfplumber opened a PDF the size cap should have refused")

    monkeypatch.setattr(pdfplumber, "open", _must_not_be_called)

    with pytest.raises(DocumentTooComplexError, match="limit for a PDF"):
        PdfConnector().ingest(path, source_id="s")


def _refusal_for_pdf_of(size: int, limit: int, tmp_path: Path, monkeypatch: Any) -> str:
    """The PDF connector's size refusal for a file of *size* bytes.

    The file is sparse: it has the size on disk without the bytes, and the
    parser is never reached.
    """
    pdfplumber = pytest.importorskip("pdfplumber")
    from nlqueries.document_connectors.pdf import PdfConnector

    path = tmp_path / "big.pdf"
    with path.open("wb") as handle:
        handle.truncate(size)
    monkeypatch.setattr(_limits, "PDF_BYTES", limit)
    monkeypatch.setattr(pdfplumber, "open", MagicMock(side_effect=AssertionError("parsed")))
    with pytest.raises(DocumentTooComplexError) as refused:
        PdfConnector().ingest(path, source_id="s")
    return str(refused.value)


def test_a_pdf_just_over_the_limit_does_not_read_as_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """100.5 MiB against 100 MiB once read "100 MiB, over the 100 MiB limit"."""
    message = _refusal_for_pdf_of(
        100 * 1024 * 1024 + 512 * 1024, 100 * 1024 * 1024, tmp_path, monkeypatch
    )
    assert "is 100.5 MiB, over the 100.0 MiB limit for a PDF" in message


def test_a_pdf_one_byte_over_the_limit_is_shown_in_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where both would round to the same figure, both are given exactly."""
    message = _refusal_for_pdf_of(100 * 1024 * 1024 + 1, 100 * 1024 * 1024, tmp_path, monkeypatch)
    assert "is 104,857,601 bytes, over the 104,857,600 bytes limit" in message


def test_a_sub_mib_pdf_limit_prints_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator's 500 KiB limit once read "0 MiB, over the 0 MiB limit"."""
    message = _refusal_for_pdf_of(600 * 1024, 500 * 1024, tmp_path, monkeypatch)
    assert "is 600.0 KiB, over the 500.0 KiB limit for a PDF" in message


def test_a_pdf_at_the_size_cap_is_ingested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is a ceiling: a file exactly at it is allowed."""
    pytest.importorskip("pdfplumber")
    from nlqueries.document_connectors.pdf import PdfConnector

    path = _pdf(tmp_path / "fits.pdf", pages=2)
    monkeypatch.setattr(_limits, "PDF_BYTES", path.stat().st_size)

    assert PdfConnector().ingest(path, source_id="s") == []  # blank pages, no text


def test_a_pdf_over_the_page_cap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pdfplumber")
    from nlqueries.document_connectors.pdf import PdfConnector

    path = _pdf(tmp_path / "long.pdf", pages=5)
    monkeypatch.setattr(_limits, "PDF_PAGES", 5)
    PdfConnector().ingest(path, source_id="s")  # at the cap: allowed

    monkeypatch.setattr(_limits, "PDF_PAGES", 4)
    with pytest.raises(DocumentTooComplexError, match="5 pages, over the 4-page limit"):
        PdfConnector().ingest(path, source_id="s")


def test_the_declared_page_count_refuses_before_pages_are_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page tree's ``/Count`` is read before any page is built.

    The file declares 10 pages and holds 3, so only the declared count can
    produce "10 pages": the count taken from the built pages would say 3 and
    pass.
    """
    pytest.importorskip("pdfplumber")
    from nlqueries.document_connectors.pdf import PdfConnector

    path = _pdf(tmp_path / "claims-more.pdf", pages=3, declared=10)
    monkeypatch.setattr(_limits, "PDF_PAGES", 5)

    with pytest.raises(DocumentTooComplexError, match="10 pages"):
        PdfConnector().ingest(path, source_id="s")


def test_a_pdf_that_understates_its_pages_is_still_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared count is self-reported, so the pages are counted again."""
    pytest.importorskip("pdfplumber")
    from nlqueries.document_connectors.pdf import PdfConnector

    path = _pdf(tmp_path / "claims-less.pdf", pages=6, declared=1)
    monkeypatch.setattr(_limits, "PDF_PAGES", 4)

    with pytest.raises(DocumentTooComplexError, match="6 pages"):
        PdfConnector().ingest(path, source_id="s")


# ---------------------------------------------------------------------------
# OOXML archives
# ---------------------------------------------------------------------------


def _diluted_archive(path: Path, member_bytes: int = 8 * 1024 * 1024) -> Path:
    """One member that expands about 1000x, diluted by a large stored member.

    The stored member is random bytes, so it neither compresses nor expands,
    and the whole-archive ratio averages out well under the limit.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            zipfile.ZipInfo("word/document.xml"),
            b"\0" * member_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            zipfile.ZipInfo("word/media/image1.bin"),
            os.urandom(1024 * 1024),
            compress_type=zipfile.ZIP_STORED,
        )
    return path


def test_a_diluted_archive_is_refused_on_its_worst_member(tmp_path: Path) -> None:
    path = _diluted_archive(tmp_path / "diluted.docx")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    whole = sum(i.file_size for i in infos) / sum(i.compress_size for i in infos)
    assert whole < _limits.MAX_EXPANSION_RATIO, f"premise failed: whole ratio {whole:.0f}x"
    assert sum(i.file_size for i in infos) < _limits.MAX_EXPANDED_BYTES

    with pytest.raises(DocumentTooComplexError, match="member 'word/document.xml'"):
        check_archive_expansion(path)


def test_the_word_connector_refuses_a_diluted_archive(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    from nlqueries.document_connectors.word import WordConnector

    path = _diluted_archive(tmp_path / "diluted.docx")
    with pytest.raises(DocumentTooComplexError, match="member"):
        WordConnector().ingest(path, source_id="s")


def test_a_repetitive_member_under_the_floor_is_accepted(tmp_path: Path) -> None:
    """A real Word document whose ``document.xml`` is far over the ratio but small.

    15,000 identical paragraphs rather than 20,000: at 20,000 the member is
    4,181,548 bytes and the whole archive 98x, both just under their limits, so
    a writer update could tip the control over for reasons unrelated to it.
    """
    docx = pytest.importorskip("docx")

    path = tmp_path / "repetitive.docx"
    document = docx.Document()
    for _ in range(15_000):
        document.add_paragraph("some text that compresses well " * 5)
    document.save(str(path))

    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("word/document.xml")
        infos = archive.infolist()
    ratio = member.file_size / member.compress_size
    whole = sum(i.file_size for i in infos) / sum(i.compress_size for i in infos)
    assert ratio > 2 * _limits.MAX_EXPANSION_RATIO, f"premise failed: member {ratio:.0f}x"
    assert member.file_size < _limits.MEMBER_RATIO_FLOOR_BYTES, "premise failed: member size"
    assert whole < _limits.MAX_EXPANSION_RATIO, f"premise failed: whole ratio {whole:.0f}x"

    check_archive_expansion(path)  # must not raise


@pytest.mark.parametrize(
    ("member_bytes", "refused"),
    [
        (_limits.MEMBER_RATIO_FLOOR_BYTES - 1, False),
        (_limits.MEMBER_RATIO_FLOOR_BYTES, True),
        (_limits.MEMBER_RATIO_FLOOR_BYTES + 1, True),
    ],
)
def test_the_member_ratio_applies_from_the_floor(
    member_bytes: int, refused: bool, tmp_path: Path
) -> None:
    """About 1000x either side of the floor, diluted so only the member rule can fire."""
    path = _diluted_archive(tmp_path / "edge.docx", member_bytes=member_bytes)
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("word/document.xml")
        infos = archive.infolist()
    assert member.file_size / member.compress_size > _limits.MAX_EXPANSION_RATIO
    whole = sum(i.file_size for i in infos) / sum(i.compress_size for i in infos)
    assert whole < _limits.MAX_EXPANSION_RATIO, f"premise failed: whole ratio {whole:.0f}x"

    if refused:
        with pytest.raises(DocumentTooComplexError, match="member 'word/document.xml'"):
            check_archive_expansion(path)
    else:
        check_archive_expansion(path)  # must not raise


def test_the_member_message_never_rounds_a_size_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory may claim any compressed size, including under one KiB.

    The entry is faked because deflate cannot reach that ratio, and the
    directory is what the check reads.
    """
    entry = zipfile.ZipInfo("xl/worksheets/sheet1.xml")
    entry.compress_size = 500
    entry.file_size = 5 * 1024 * 1024 + 123_456
    # A stored member large enough to hold the whole-archive ratio under 100x.
    stored = zipfile.ZipInfo("xl/media/image1.png")
    stored.compress_size = stored.file_size = 1024 * 1024

    class _Archive:
        def __init__(self, *args: object) -> None:
            pass

        def __enter__(self) -> _Archive:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def infolist(self) -> list[zipfile.ZipInfo]:
            return [entry, stored]

    monkeypatch.setattr(_limits.zipfile, "is_zipfile", lambda path: True)
    monkeypatch.setattr(_limits.zipfile, "ZipFile", _Archive)
    monkeypatch.setattr(_limits, "MAX_EXPANSION_RATIO", 100)

    with pytest.raises(DocumentTooComplexError) as refused:
        check_archive_expansion(tmp_path / "claims.xlsx")
    assert "(500 bytes to 5.1 MiB)" in str(refused.value)


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (1, "1 bytes"),
        (1023, "1,023 bytes"),
        (1024, "1.0 KiB"),
        (5_000, "4.9 KiB"),
        (1024 * 1024, "1.0 MiB"),
        (3_000 * 1024 * 1024, "3,000.0 MiB"),
    ],
)
def test_sizes_are_shown_in_a_unit_that_keeps_them_non_zero(size: int, text: str) -> None:
    assert _limits._size(size) == text


@pytest.mark.parametrize(
    ("size", "limit", "expected"),
    [
        (2048, 1024, ("2.0 KiB", "1.0 KiB")),
        (1025, 1024, ("1,025 bytes", "1,024 bytes")),
        (500, 100, ("500 bytes", "100 bytes")),
    ],
)
def test_a_size_and_its_limit_never_format_alike(
    size: int, limit: int, expected: tuple[str, str]
) -> None:
    assert _limits._size_over(size, limit) == expected


def test_the_whole_archive_messages_print_small_sizes_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both whole-archive refusals, on an archive far under one MiB."""
    path = tmp_path / "small.xlsx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"\0" * 200_000)

    monkeypatch.setattr(_limits, "MAX_EXPANDED_BYTES", 100_000)
    with pytest.raises(DocumentTooComplexError) as refused:
        check_archive_expansion(path)
    assert "expands to 195.3 KiB, over the 97.7 KiB limit" in str(refused.value)

    monkeypatch.setattr(_limits, "MAX_EXPANDED_BYTES", 10**12)
    with pytest.raises(DocumentTooComplexError) as refused:
        check_archive_expansion(path)
    assert "bytes to 195.3 KiB)" in str(refused.value)
    assert " 0 KiB" not in str(refused.value)


def test_ordinary_documents_pass_the_per_member_check(tmp_path: Path) -> None:
    """The control: real files from the real writers, at the default limits."""
    docx = pytest.importorskip("docx")
    openpyxl = pytest.importorskip("openpyxl")

    document = docx.Document()
    for n in range(2_000):
        document.add_paragraph(f"Paragraph {n} of an ordinary report, with some prose.")
    document.save(str(tmp_path / "report.docx"))
    check_archive_expansion(tmp_path / "report.docx")

    workbook = openpyxl.Workbook(write_only=True)
    sheet = workbook.create_sheet("data")
    for n in range(5_000):
        sheet.append([n, n * 2, f"row {n}", 1.5, "category"])
    workbook.save(str(tmp_path / "data.xlsx"))
    check_archive_expansion(tmp_path / "data.xlsx")


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------


def test_the_word_budget_is_checked_during_the_paragraph_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document with no headings is one section, so the per-section check
    would come only after every paragraph was walked; the walk checks too."""
    docx = pytest.importorskip("docx")
    from nlqueries.document_connectors import word as word_module

    path = tmp_path / "flat.docx"
    document = docx.Document()
    for n in range(2_500):
        document.add_paragraph(f"line {n}")
    document.save(str(path))

    progress: list[str] = []

    class _RecordingBudget:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def check(self, stage: str) -> None:
            progress.append(stage)

    monkeypatch.setattr(word_module, "ExtractionBudget", _RecordingBudget)
    word_module.WordConnector().ingest(path, source_id="s")

    walked = [p for p in progress if "paragraphs" in p]
    assert walked == ["1000 of 2500 paragraphs", "2000 of 2500 paragraphs"], progress


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------


class _Soup:
    """Enough of BeautifulSoup for the connector, without the wiki extra."""

    def __init__(self, markup: str, parser: str = "html.parser") -> None:
        self._markup = markup

    def get_text(self, separator: str = "", strip: bool = False) -> str:
        return self._markup.strip() if strip else self._markup


@pytest.fixture
def confluence_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A stubbed Confluence client, returned by the connector's ``_client``."""
    client = MagicMock()
    atlassian = MagicMock()
    atlassian.Confluence.return_value = client
    bs4 = MagicMock()
    bs4.BeautifulSoup = _Soup
    monkeypatch.setitem(sys.modules, "atlassian", atlassian)
    monkeypatch.setitem(sys.modules, "bs4", bs4)
    return client


def _connector(base_url: str = "https://acme.atlassian.net") -> Any:
    from nlqueries.document_connectors.confluence import ConfluenceConnector

    return ConfluenceConnector(base_url=base_url, username="u", api_token="t")


def _result(page_id: str) -> dict[str, Any]:
    return {"content": {"id": page_id, "title": f"Page {page_id}", "_links": {}}}


@pytest.mark.parametrize(
    "space_key",
    [
        'ENG" OR space = "HR',
        "ENG\n",
        "",
        "EN G",
        "ENG-1",
        "ENG.1",
        "~",
        "../ENG",
        "ENG;",
        "~john smith",
        '~john"smith',
        "~john.smith\n",
        "~john\\smith",
    ],
)
def test_an_invalid_space_key_is_refused_before_any_query(
    space_key: str, confluence_client: MagicMock
) -> None:
    from datetime import UTC, datetime

    connector = _connector()
    with pytest.raises(ValueError, match="Invalid Confluence space key"):
        connector.ingest(space_key, source_id="s")
    with pytest.raises(ValueError, match="Invalid Confluence space key"):
        connector.has_changes_since(space_key, datetime(2024, 1, 1, tzinfo=UTC))
    confluence_client.cql.assert_not_called()


@pytest.mark.parametrize(
    "space_key",
    [
        "ENG",
        "eng_2",
        "~alice",
        "~5f00a1b2c3",
        "~john.smith",
        "~jane-doe",
        "~jane@acme.com",
        "~557058:f2d5e7a1-0b3c-4d5e-8f90-123456789abc",
    ],
)
def test_a_valid_space_key_reaches_the_query(space_key: str, confluence_client: MagicMock) -> None:
    confluence_client.cql.return_value = {"results": []}
    assert _connector().ingest(space_key, source_id="s") == []
    assert f'space = "{space_key}"' in confluence_client.cql.call_args[0][0]


def test_an_unparseable_base_url_does_not_raise_in_the_constructor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unbalanced IPv6 bracket makes ``urlsplit`` raise; the constructor must not.

    The CLI builds the connector outside the ``try`` that reports a failed sync,
    so a raise here was a traceback. Such a URL is "cannot tell" for the
    warning, and fails when it is used.
    """
    from urllib.parse import urlsplit

    base_url = "http://[fe80::1:8090"
    with pytest.raises(ValueError):
        urlsplit(base_url)  # the premise

    with caplog.at_level(logging.WARNING, logger="nlqueries.document_connectors.confluence"):
        _connector(base_url)
    assert not caplog.records, caplog.text


@pytest.mark.parametrize(
    ("base_url", "warns"),
    [
        ("http://confluence.example.com", True),
        ("HTTP://10.0.0.5:8090/", True),
        ("https://confluence.example.com", False),
        ("http://localhost:8090", False),
        ("http://127.0.0.1:8090", False),
        ("http://[::1]:8090", False),
    ],
)
def test_an_http_base_url_to_a_remote_host_warns(
    base_url: str, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="nlqueries.document_connectors.confluence"):
        _connector(base_url)
    warned = [r for r in caplog.records if "http://" in r.getMessage()]
    assert bool(warned) is warns, caplog.text


def test_a_space_over_the_page_cap_is_refused_on_its_reported_size(
    confluence_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``totalSize`` refuses on the first response, before any body is read."""
    monkeypatch.setattr(_limits, "CONFLUENCE_PAGES", 5)
    confluence_client.cql.return_value = {
        "results": [_result(str(n)) for n in range(3)],
        "totalSize": 6,
    }
    with pytest.raises(DocumentTooComplexError, match="more than 5 pages"):
        _connector().ingest("ENG", source_id="s")
    assert confluence_client.cql.call_count == 1
    confluence_client.get_page_by_id.assert_not_called()


def test_a_space_over_the_page_cap_is_refused_by_count(
    confluence_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``totalSize`` the pages read are counted, and the search stops."""
    monkeypatch.setattr(_limits, "CONFLUENCE_PAGES", 5)

    def _search(cql: str, start: int = 0, limit: int = 0, **_: Any) -> dict[str, Any]:
        batch = [_result(str(n)) for n in range(start, start + 3)]
        return {"results": batch, "_links": {"next": "/rest/api/search?next"}}

    confluence_client.cql.side_effect = _search
    with pytest.raises(DocumentTooComplexError, match="more than 5 pages"):
        _connector().ingest("ENG", source_id="s")
    assert confluence_client.cql.call_count == 2
    confluence_client.get_page_by_id.assert_not_called()


def test_a_space_at_the_page_cap_is_read(
    confluence_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_limits, "CONFLUENCE_PAGES", 3)
    confluence_client.cql.return_value = {
        "results": [_result(str(n)) for n in range(3)],
        "totalSize": 3,
    }
    confluence_client.get_page_by_id.return_value = {"body": {"storage": {"value": "text"}}}
    assert len(_connector().ingest("ENG", source_id="s")) == 3


def test_a_space_over_the_byte_cap_is_refused(
    confluence_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bodies are counted as they arrive; the read stops at the one that crosses."""
    confluence_client.cql.return_value = {"results": [_result(str(n)) for n in range(5)]}
    confluence_client.get_page_by_id.return_value = {"body": {"storage": {"value": "x" * 1000}}}

    monkeypatch.setattr(_limits, "CONFLUENCE_BYTES", 5000)
    chunks = _connector().ingest("ENG", source_id="s")  # exactly at the cap
    assert {c.metadata["page_id"] for c in chunks} == {"0", "1", "2", "3", "4"}

    confluence_client.get_page_by_id.reset_mock()
    monkeypatch.setattr(_limits, "CONFLUENCE_BYTES", 2500)
    with pytest.raises(DocumentTooComplexError, match="NLQ_MAX_CONFLUENCE_BYTES") as refused:
        _connector().ingest("ENG", source_id="s")
    assert "reached 2.9 KiB, over the 2.4 KiB limit" in str(refused.value)
    assert confluence_client.get_page_by_id.call_count == 3


def test_the_byte_cap_counts_encoded_bytes_not_characters(
    confluence_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1,000 three-byte characters are 3,000 bytes, not 1,000."""
    confluence_client.cql.return_value = {"results": [_result("1")]}
    confluence_client.get_page_by_id.return_value = {"body": {"storage": {"value": "€" * 1000}}}
    monkeypatch.setattr(_limits, "CONFLUENCE_BYTES", 2000)
    with pytest.raises(DocumentTooComplexError):
        _connector().ingest("ENG", source_id="s")


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------


def _blocks(texts: list[str], next_cursor: str | None) -> dict[str, Any]:
    return {
        "results": [
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": t}]}} for t in texts
        ],
        "has_more": next_cursor is not None,
        "next_cursor": next_cursor,
    }


@pytest.fixture
def notion_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    client.pages.retrieve.return_value = {"last_edited_time": "", "properties": {}}
    client.blocks.children.list.side_effect = [
        _blocks(["one", "two", "three"], next_cursor="c1"),
        _blocks(["four", "five", "six"], next_cursor="c2"),
        _blocks(["seven"], next_cursor=None),
    ]
    module = MagicMock()
    module.Client.return_value = client
    monkeypatch.setitem(sys.modules, "notion_client", module)
    return client


def test_a_notion_page_over_the_block_cap_is_refused(
    notion_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries.document_connectors.notion import NotionConnector

    monkeypatch.setattr(_limits, "NOTION_BLOCKS", 5)
    with pytest.raises(DocumentTooComplexError, match="more than 5 blocks"):
        NotionConnector(api_token="t").ingest("page", source_id="s")
    assert notion_client.blocks.children.list.call_count == 2  # stopped, not read to the end


def test_a_notion_page_at_the_block_cap_is_read(
    notion_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries.document_connectors.notion import NotionConnector

    monkeypatch.setattr(_limits, "NOTION_BLOCKS", 7)
    chunks = NotionConnector(api_token="t").ingest("page", source_id="s")
    assert "seven" in " ".join(c.text for c in chunks)


def test_a_notion_page_under_the_block_cap_but_over_the_byte_cap_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three blocks of 1,000 bytes each: well under the block cap, over 2,500 bytes."""
    from nlqueries.document_connectors.notion import NotionConnector

    client = MagicMock()
    client.pages.retrieve.return_value = {"last_edited_time": "", "properties": {}}
    client.blocks.children.list.return_value = _blocks(["x" * 1000] * 3, next_cursor=None)
    module = MagicMock()
    module.Client.return_value = client
    monkeypatch.setitem(sys.modules, "notion_client", module)
    monkeypatch.setattr(_limits, "NOTION_BLOCKS", 100)

    monkeypatch.setattr(_limits, "NOTION_BYTES", 3000)
    chunks = NotionConnector(api_token="t").ingest("page", source_id="s")  # at the cap
    assert chunks

    monkeypatch.setattr(_limits, "NOTION_BYTES", 2500)
    with pytest.raises(DocumentTooComplexError, match="NLQ_MAX_NOTION_BYTES") as refused:
        NotionConnector(api_token="t").ingest("page", source_id="s")
    assert "reached 2.9 KiB of text, over the 2.4 KiB limit" in str(refused.value)


def test_the_notion_byte_cap_counts_encoded_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """1,000 three-byte characters are 3,000 bytes, not 1,000."""
    from nlqueries.document_connectors.notion import NotionConnector

    client = MagicMock()
    client.pages.retrieve.return_value = {"last_edited_time": "", "properties": {}}
    client.blocks.children.list.return_value = _blocks(["€" * 1000], next_cursor=None)
    module = MagicMock()
    module.Client.return_value = client
    monkeypatch.setitem(sys.modules, "notion_client", module)
    monkeypatch.setattr(_limits, "NOTION_BYTES", 2000)
    with pytest.raises(DocumentTooComplexError):
        NotionConnector(api_token="t").ingest("page", source_id="s")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config_name", "limits_name"),
    [
        ("MAX_PDF_BYTES", "PDF_BYTES"),
        ("MAX_PDF_PAGES", "PDF_PAGES"),
        ("MAX_CONFLUENCE_PAGES", "CONFLUENCE_PAGES"),
        ("MAX_CONFLUENCE_BYTES", "CONFLUENCE_BYTES"),
        ("MAX_NOTION_BLOCKS", "NOTION_BLOCKS"),
        ("MAX_NOTION_BYTES", "NOTION_BYTES"),
    ],
)
def test_the_new_limits_come_from_config(config_name: str, limits_name: str) -> None:
    import nlqueries.config as config_module

    assert getattr(_limits, limits_name) == getattr(config_module, config_name)


def test_the_new_settings_read_their_environment_variables(tmp_path: Path) -> None:
    """Each setting is read from the variable the docs name.

    In a fresh interpreter, since ``config`` reads the environment once, at
    import. Run from an empty directory so no ``.env`` file is picked up.
    """
    import json
    import subprocess

    values = {
        "NLQ_MAX_PDF_BYTES": "11",
        "NLQ_MAX_PDF_PAGES": "12",
        "NLQ_MAX_CONFLUENCE_PAGES": "13",
        "NLQ_MAX_CONFLUENCE_BYTES": "14",
        "NLQ_MAX_NOTION_BLOCKS": "15",
        "NLQ_MAX_NOTION_BYTES": "16",
    }
    script = (
        "import json, nlqueries.config as c; print(json.dumps([c.MAX_PDF_BYTES, "
        "c.MAX_PDF_PAGES, c.MAX_CONFLUENCE_PAGES, c.MAX_CONFLUENCE_BYTES, "
        "c.MAX_NOTION_BLOCKS, c.MAX_NOTION_BYTES]))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **values},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [11, 12, 13, 14, 15, 16]
