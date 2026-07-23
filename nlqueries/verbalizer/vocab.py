# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.verbalizer.vocab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vocabulary for the verbalizer: short human labels for tables and columns, plus
glossary terms. The verbalizer falls back to a humanized identifier when a label
is absent, so a bare ``Vocab()`` still produces readable English; enterprise
supplies business names (KB descriptions, Nexus bridge/Beacon names) through
:func:`build_vocab` and by constructing a ``Vocab`` directly. Kept a plain core
value type so nothing enterprise leaks across the tier boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def humanize(identifier: str) -> str:
    """Render a bare identifier as words: drop quoting, underscores → spaces,
    lower-case (so dialect-uppercased identifiers like ``TOTAL`` read naturally)."""
    ident = identifier.strip().strip('"`[]')
    ident = ident.replace("_", " ").strip().lower()
    return ident or identifier


@dataclass(frozen=True)
class Vocab:
    """Labels used when verbalizing SQL.

    ``tables`` maps a table name to its label; ``columns`` accepts both
    ``"table.column"`` (preferred) and bare ``"column"`` keys; ``terms`` maps a
    glossary term to its definition (used by later ACE tasks, carried here so the
    shape is stable). Every lookup falls back to :func:`humanize`.
    """

    tables: dict[str, str] = field(default_factory=dict)
    columns: dict[str, str] = field(default_factory=dict)
    terms: dict[str, str] = field(default_factory=dict)

    def table_label(self, name: str) -> str:
        return self.tables.get(name) or humanize(name)

    def column_label(self, table: str, column: str) -> str:
        if table and f"{table}.{column}" in self.columns:
            return self.columns[f"{table}.{column}"]
        return self.columns.get(column) or humanize(column)


def build_vocab(kb: dict[str, Any]) -> Vocab:
    """Build a :class:`Vocab` from a core KB dict (``kb_version`` 2).

    Uses table/column names (humanized) as labels and loads glossary terms. Core
    KBs carry sentence-length ``description`` fields rather than short labels, so
    those are intentionally not used as inline labels here — enterprise layers
    business names on top. Tolerant of partial/legacy KB shapes.
    """
    tables: dict[str, str] = {}
    columns: dict[str, str] = {}
    schema = kb.get("schema", {}) if isinstance(kb, dict) else {}
    for table in schema.get("tables", []) if isinstance(schema, dict) else []:
        if not isinstance(table, dict):
            continue
        name = str(table.get("name", ""))
        if not name:
            continue
        tables[name] = humanize(name)
        for col in table.get("columns", []):
            if not isinstance(col, dict):
                continue
            col_name = str(col.get("name", ""))
            if not col_name:
                continue
            columns[f"{name}.{col_name}"] = humanize(col_name)
            columns.setdefault(col_name, humanize(col_name))

    terms: dict[str, str] = {}
    biz = kb.get("business_context", {}) if isinstance(kb, dict) else {}
    for entry in biz.get("glossary", []) if isinstance(biz, dict) else []:
        if isinstance(entry, dict) and entry.get("term"):
            terms[str(entry["term"])] = str(entry.get("definition", ""))

    return Vocab(tables=tables, columns=columns, terms=terms)
