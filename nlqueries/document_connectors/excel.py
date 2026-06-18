"""
nlqueries.document_connectors.excel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Microsoft Excel (.xlsx) document connector using ``openpyxl`` for sheet
and row extraction.

Chunking strategy: row-batch-to-text per sheet.  Each sheet is processed
in batches of 50 rows.  The first row of a sheet is tested with a header
heuristic: if all non-empty cells are non-numeric strings (i.e. ``str`` type,
not ``int`` or ``float``), those values are used as column-name prefixes for
every row in the sheet.  Each data row is serialised as
``"col1: val1 | col2: val2 | ..."``.

``page_number`` maps to the sheet index (1-based).

``.xls`` and ``.csv`` are not supported — openpyxl handles ``.xlsx`` only.

Requires the ``docs`` optional dependency group:
    pip install "nlqueries-core[docs]"
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector

_BATCH_SIZE = 50


def _make_chunk_id(source_id: str, sheet_index: int, batch_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:sheet_index:batch_index)[:16]."""
    raw = f"{source_id}:{sheet_index}:{batch_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cell_str(value: Any) -> str:
    """Convert a cell value to a stripped string; return empty string for None."""
    if value is None:
        return ""
    return str(value).strip()


def _is_header_row(values: list[Any]) -> bool:
    """Heuristic: all non-empty cells are non-numeric strings (``str`` type)."""
    non_empty = [v for v in values if v is not None and str(v).strip() != ""]
    if not non_empty:
        return False
    return all(isinstance(v, str) for v in non_empty)


def _rows_to_text(rows: list[list[Any]], headers: list[str] | None) -> str:
    """Serialise a batch of data rows to a single text block.

    When *headers* is provided each row becomes ``"col1: val1 | col2: val2 | ..."``.
    Without headers the raw cell values are joined with `` | ``.
    """
    lines: list[str] = []
    for row in rows:
        if headers:
            pairs = [f"{headers[i]}: {_cell_str(v)}" for i, v in enumerate(row) if i < len(headers)]
        else:
            pairs = [_cell_str(v) for v in row]
        line = " | ".join(p for p in pairs if p)
        if line:
            lines.append(line)
    return "\n".join(lines)


class ExcelConnector(DocumentConnector):
    """Extract and chunk text from Excel (.xlsx) files using openpyxl.

    Each worksheet maps to a ``page_number`` equal to its 1-based sheet index.
    Rows are consumed in batches of ``_BATCH_SIZE`` (50).  The first row of each
    sheet is analysed with ``_is_header_row()``; when all its non-empty cells are
    plain strings, those values are treated as column headers and used to prefix
    every subsequent data row.
    """

    def ingest(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required for ExcelConnector. "
                "Install it with: pip install 'nlqueries-core[docs]'"
            ) from exc

        source_path = Path(source)
        wb = openpyxl.load_workbook(str(source_path), read_only=True, data_only=True)

        chunks: list[DocumentChunk] = []

        try:
            for sheet_index, sheet in enumerate(wb.worksheets, start=1):
                headers: list[str] | None = None
                data_rows: list[list[Any]] = []
                first_data_row_num = 1
                header_detected = False

                for row_num, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                    values = list(row)
                    if not header_detected:
                        header_detected = True
                        if _is_header_row(values):
                            headers = [_cell_str(v) or f"col{i + 1}" for i, v in enumerate(values)]
                            first_data_row_num = row_num + 1
                            continue  # header consumed; not added to data_rows
                    data_rows.append(values)

                for batch_index, batch_offset in enumerate(range(0, len(data_rows), _BATCH_SIZE)):
                    batch = data_rows[batch_offset : batch_offset + _BATCH_SIZE]
                    text = _rows_to_text(batch, headers)
                    if not text.strip():
                        continue

                    row_start = first_data_row_num + batch_offset
                    row_end = row_start + len(batch) - 1
                    chunk_id = _make_chunk_id(source_id, sheet_index, batch_index)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_id=source_id,
                            source_name=source_path.name,
                            page_number=sheet_index,
                            chunk_index=batch_index,
                            text=text,
                            metadata={
                                "connector": "excel",
                                "file_path": str(source_path),
                                "sheet_name": sheet.title,
                                "row_range": f"{row_start}-{row_end}",
                            },
                        )
                    )
        finally:
            wb.close()

        return chunks

    def supports(self, source: str | Path) -> bool:
        return Path(source).suffix.lower() == ".xlsx"
