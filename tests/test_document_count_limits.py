"""Bounds on how much work extracting one document may do (finding 2.8).

The companion to ``test_document_expansion_limits.py``. That gate reads what an
archive *declares* about itself before parsing; these bound what extraction
actually does while reading, so they hold for a `.pdf` (not an archive at all)
and for an archive whose directory misstated its contents.

Real files and real parsers, for the same reason as the expansion tests: a
mocked ``openpyxl`` would confirm the arithmetic and say nothing about whether a
real workbook is refused.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from nlqueries.document_connectors import _limits
from nlqueries.document_connectors._limits import DocumentTooComplexError, ExtractionBudget

openpyxl = pytest.importorskip("openpyxl")
docx = pytest.importorskip("docx")

_BATCH = 50


def _sheet(path: Path, rows: int, *, header: bool = True, width: int = 3) -> Path:
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("data")
    if header:
        ws.append([f"col{i}" for i in range(width)])
    for n in range(rows):
        ws.append([f"r{n}"] * width)
    wb.save(str(path))
    return path


# ---------------------------------------------------------------------------
# The row cap
# ---------------------------------------------------------------------------


def test_a_workbook_under_the_row_cap_is_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control. Every limit here is a ceiling over documents people really have."""
    from nlqueries.document_connectors.excel import ExcelConnector

    monkeypatch.setattr(_limits, "MAX_ROWS", 200)
    chunks = ExcelConnector().ingest(_sheet(tmp_path / "ok.xlsx", rows=200), source_id="s")
    assert len(chunks) == 4, [c.chunk_index for c in chunks]


def test_a_workbook_over_the_row_cap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counted while reading, so it bounds an archive that lied about its size."""
    from nlqueries.document_connectors.excel import ExcelConnector

    monkeypatch.setattr(_limits, "MAX_ROWS", 200)
    with pytest.raises(DocumentTooComplexError, match="more than 200 rows"):
        ExcelConnector().ingest(_sheet(tmp_path / "big.xlsx", rows=201), source_id="s")


def test_the_row_cap_counts_across_sheets_not_per_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workbook is refused on its total, not on its largest sheet.

    Per-sheet would be trivially evaded by splitting the same rows across
    several tabs, which costs a worker exactly as much.
    """
    from nlqueries.document_connectors.excel import ExcelConnector

    path = tmp_path / "many.xlsx"
    wb = openpyxl.Workbook(write_only=True)
    for s in range(4):
        ws = wb.create_sheet(f"s{s}")
        ws.append(["a", "b"])
        for n in range(60):
            ws.append([f"v{n}", n])
    wb.save(str(path))

    monkeypatch.setattr(_limits, "MAX_ROWS", 200)  # 4 x 60 = 240 total, 60 per sheet
    with pytest.raises(DocumentTooComplexError, match="more than 200 rows"):
        ExcelConnector().ingest(path, source_id="s")


# ---------------------------------------------------------------------------
# The refactor that made the row cap possible: streaming, not accumulating
# ---------------------------------------------------------------------------


def test_a_blank_batch_still_advances_the_chunk_index(tmp_path: Path) -> None:
    """The subtlety the streaming rewrite had to preserve exactly.

    The old connector built every row of a sheet, enumerated over fixed offsets
    and skipped a batch whose text was blank -- so a blank stretch leaves a *gap*
    in the emitted indexes rather than renumbering what follows. Chunk ids are
    derived from that index, so getting it wrong silently re-ids every chunk
    after a blank run, and a re-ingest would orphan the old vectors instead of
    replacing them.

    Measured against the pre-refactor connector: indexes ``[0, 2]`` and row
    ranges ``2-51`` / ``102-121``.
    """
    from nlqueries.document_connectors.excel import ExcelConnector

    path = tmp_path / "gap.xlsx"
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("data")
    ws.append(["alpha", "beta", "gamma"])
    for n in range(1, 51):  # batch 0: 50 rows
        ws.append([f"r{n}", n, n * 1.5])
    for _ in range(50):  # batch 1: blank, skipped but still counted
        ws.append([None, None, None])
    for n in range(1, 21):  # batch 2: 20 rows
        ws.append([f"r{n}", n, n * 1.5])
    wb.save(str(path))

    chunks = ExcelConnector().ingest(path, source_id="fixed-source-id")

    assert [c.chunk_index for c in chunks] == [0, 2]
    assert [(c.metadata or {})["row_range"] for c in chunks] == ["2-51", "102-121"]


def test_the_last_partial_batch_is_emitted(tmp_path: Path) -> None:
    """Streaming emits on a full batch, so the remainder needs its own flush.

    Forgetting it is the obvious way to break this refactor, and it would drop
    the tail of every sheet whose row count is not a multiple of 50 -- quietly,
    since the chunks that remain are all correct.
    """
    from nlqueries.document_connectors.excel import ExcelConnector

    chunks = ExcelConnector().ingest(_sheet(tmp_path / "tail.xlsx", rows=125), source_id="s")

    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert (chunks[-1].metadata or {})["row_range"] == "102-126"


def test_a_header_only_sheet_produces_nothing(tmp_path: Path) -> None:
    """No data rows means no partial batch to flush, and no empty chunk either."""
    from nlqueries.document_connectors.excel import ExcelConnector

    assert ExcelConnector().ingest(_sheet(tmp_path / "h.xlsx", rows=0), source_id="s") == []


# ---------------------------------------------------------------------------
# The wall-clock budget
# ---------------------------------------------------------------------------


def test_the_budget_allows_an_ordinary_document() -> None:
    """The control for the clock: a budget that has not elapsed must not fire."""
    ExtractionBudget(seconds=60, name="x").check("0 pages")  # must not raise


def test_the_budget_refuses_once_it_is_spent() -> None:
    """And names how far extraction got, which is what makes the error useful."""
    budget = ExtractionBudget(seconds=0.01, name="slow.pdf")
    time.sleep(0.05)
    with pytest.raises(DocumentTooComplexError, match=r"slow\.pdf.*budget.*412 of 5000 pages"):
        budget.check("412 of 5000 pages")


def test_the_budget_defaults_to_the_configured_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read at call time, so a deployment's setting and a test both take effect."""
    monkeypatch.setattr(_limits, "MAX_SECONDS", 0.0)
    budget = ExtractionBudget(name="x")
    time.sleep(0.01)
    with pytest.raises(DocumentTooComplexError):
        budget.check("somewhere")


@pytest.mark.parametrize("connector_name", ["excel", "word", "pdf"])
def test_every_connector_is_bounded_by_the_clock(
    connector_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One budget across all three, rather than a count cap invented per format.

    Parameterised because the point of a shared budget is that no connector is
    exempt -- and `pdf` is the one that matters most, being the only format the
    expansion gate never sees (a PDF is not a zip).
    """
    monkeypatch.setattr(_limits, "MAX_SECONDS", 0.0)

    if connector_name == "excel":
        from nlqueries.document_connectors.excel import ExcelConnector

        connector: Any = ExcelConnector()
        path = _sheet(tmp_path / "x.xlsx", rows=120)
    elif connector_name == "word":
        from nlqueries.document_connectors.word import WordConnector

        connector = WordConnector()
        path = tmp_path / "x.docx"
        document = docx.Document()
        for n in range(40):
            document.add_paragraph(f"paragraph {n} " * 10)
        document.save(str(path))
    else:
        pdfplumber = pytest.importorskip("pdfplumber")
        assert pdfplumber  # the connector imports it itself
        from nlqueries.document_connectors.pdf import PdfConnector

        connector = PdfConnector()
        path = _minimal_pdf(tmp_path / "x.pdf", pages=3)

    with pytest.raises(DocumentTooComplexError, match="budget"):
        connector.ingest(path, source_id="s")


def _minimal_pdf(path: Path, pages: int) -> Path:
    """A syntactically valid multi-page PDF, written by hand.

    No PDF *writer* is among the dev dependencies -- pdfplumber only reads -- and
    adding one to test a clock would be a dependency for a stopwatch.
    """
    objects: list[bytes] = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
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
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    out += trailer.encode() + f"startxref\n{xref_at}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def test_the_word_budget_covers_parsing_not_only_chunking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the clock starts, which the parameterised test above cannot see.

    With `MAX_SECONDS` patched to `0.0` the first check fires wherever the
    budget was constructed, so that test passes for a budget started after the
    expensive phases — which is exactly the defect review found. This one uses a
    small non-zero budget and makes *parsing* dominate, so it fails if the clock
    starts after `docx.Document()` and passes only when it starts before.
    """
    from nlqueries.document_connectors import word as word_module

    path = tmp_path / "slow.docx"
    document = docx.Document()
    for n in range(5):
        document.add_paragraph(f"paragraph {n}")
    document.save(str(path))

    real_document = docx.Document

    def _slow_document(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.2)  # stand in for a document that is expensive to parse
        return real_document(*args, **kwargs)

    # Patched on the `docx` module itself, not on `word_module`: the connector
    # imports docx *inside* `ingest`, so there is no module attribute to replace
    # and it resolves the name from `sys.modules` at call time.
    monkeypatch.setattr(docx, "Document", _slow_document)
    monkeypatch.setattr(_limits, "MAX_SECONDS", 0.05)

    with pytest.raises(DocumentTooComplexError, match="parsing the document"):
        word_module.WordConnector().ingest(path, source_id="s")


def test_padding_rows_do_not_count_towards_the_row_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap must count rows a person can count in their own file.

    `iter_rows` yields every *materialised* row, and whole-column formatting
    leaves thousands of empty ones behind. Measured on this version: a 200-row
    sheet with such padding yields 5,201 tuples. A cap counting those refuses a
    workbook its owner sees 200 rows in.

    (The reviewer proposed the mechanism as `<dimension>` over-declaration.
    Measured, that does *not* pad — a sheet declaring `A1:C1048576` still yields
    only its real rows. Materialised blank rows are the route that does.)
    """
    from nlqueries.document_connectors.excel import ExcelConnector

    path = tmp_path / "padded.xlsx"
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("data")
    ws.append(["a", "b", "c"])
    for n in range(100):
        ws.append([f"r{n}", n, n * 2])
    for _ in range(1_000):
        ws.append([None, None, None])
    wb.save(str(path))

    monkeypatch.setattr(_limits, "MAX_ROWS", 200)  # 100 real rows, 1,100 yielded
    chunks = ExcelConnector().ingest(path, source_id="s")  # must not raise
    assert chunks, "a padded workbook produced no chunks"


def test_the_seconds_setting_falls_back_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed value must not raise while `nlqueries.config` is imported.

    That would take down the CLI and the MCP server at startup, not merely fail
    one ingest -- and `docs/configuration.md` promises fallback behaviour for the
    neighbouring document limits, so an operator expects it here.

    `nan` has its own case: every comparison against it is false, so a budget of
    `nan` would not fail loudly, it would silently never expire.
    """
    from nlqueries.config import _positive_float

    monkeypatch.setenv("NLQ_MAX_EXTRACTION_SECONDS", "45.5")
    assert _positive_float("NLQ_MAX_EXTRACTION_SECONDS", 120.0) == 45.5

    for bad in ("120s", "", "abc", "0", "-1", "inf", "-inf", "nan"):
        monkeypatch.setenv("NLQ_MAX_EXTRACTION_SECONDS", bad)
        assert _positive_float("NLQ_MAX_EXTRACTION_SECONDS", 120.0) == 120.0, bad


def test_the_excel_budget_covers_opening_the_workbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the Excel clock starts, for the same reason as the Word case.

    `test_every_connector_is_bounded_by_the_clock[excel]` patches `MAX_SECONDS`
    to `0.0`, so its first check fires wherever the budget was constructed and it
    passes equally for a budget built before `load_workbook` and one built after
    — which is why it did not catch the ordering. `read_only=True` makes row
    iteration lazy, not opening: measured at ~0.33s for a 60,000-row workbook
    before a single row is read.

    Small non-zero budget, with opening made slow, so it fails if the clock
    starts after `load_workbook`.
    """
    import openpyxl as real_openpyxl
    from nlqueries.document_connectors.excel import ExcelConnector

    path = _sheet(tmp_path / "slow.xlsx", rows=10)
    real_load = real_openpyxl.load_workbook

    def _slow_load(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.2)
        return real_load(*args, **kwargs)

    # Patched on `openpyxl` itself: the connector imports it inside `ingest`, so
    # there is no attribute on the connector module to replace.
    monkeypatch.setattr(real_openpyxl, "load_workbook", _slow_load)
    monkeypatch.setattr(_limits, "MAX_SECONDS", 0.05)

    with pytest.raises(DocumentTooComplexError, match="opening the workbook"):
        ExcelConnector().ingest(path, source_id="s")


def test_a_refused_workbook_does_not_stay_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget firing must not leak the workbook's file handle.

    The check used to sit between `load_workbook` and the `try` whose `finally`
    calls `wb.close()`, so the one case the budget exists for was the one case
    that skipped the close — leaving the zip handle to a finaliser, which on
    Windows is long enough to keep the uploaded file locked against the caller
    deleting it.

    Asserted on `close()` actually being called, rather than on the file being
    deletable, because the latter passes on this platform either way and would
    prove nothing.
    """
    import openpyxl as real_openpyxl
    from nlqueries.document_connectors.excel import ExcelConnector

    path = _sheet(tmp_path / "leak.xlsx", rows=10)
    real_load = real_openpyxl.load_workbook
    closed: list[bool] = []

    def _tracking_load(*args: Any, **kwargs: Any) -> Any:
        wb = real_load(*args, **kwargs)
        real_close = wb.close

        def _close() -> None:
            closed.append(True)
            real_close()

        wb.close = _close  # type: ignore[method-assign]
        time.sleep(0.2)
        return wb

    monkeypatch.setattr(real_openpyxl, "load_workbook", _tracking_load)
    monkeypatch.setattr(_limits, "MAX_SECONDS", 0.05)

    with pytest.raises(DocumentTooComplexError):
        ExcelConnector().ingest(path, source_id="s")

    assert closed, "the workbook was not closed when the budget refused it"
