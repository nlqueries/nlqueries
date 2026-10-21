"""``export-kb`` for a connector with no registered connector class.

That path introspects the database through SQLAlchemy and writes whole sample
rows. It must withhold the same columns the knowledge-base generator refuses to
sample (:func:`nlqueries.knowledge.kb_generator.is_pii_column`).
"""

from __future__ import annotations

import sqlite3
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from nlqueries.cli import main as cli_main


def test_the_fallback_export_drops_sensitive_columns_from_sample_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "shop.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, order_total REAL,"
            " email TEXT, api_key TEXT, first_name TEXT)"
        )
        conn.execute(
            "INSERT INTO orders VALUES (1, 'open', 12.5, 'jane@example.com', 'sk-123', 'Jane')"
        )

    cfg = {"db_type": "unregistered", "database": "shop"}
    monkeypatch.setattr(cli_main, "_resolve_alias", lambda value: value)
    monkeypatch.setattr(cli_main, "_require_connector", lambda connector_id: cfg)
    monkeypatch.setattr(cli_main, "connector_class_for", lambda db_type, cfg: None)
    monkeypatch.setattr(cli_main, "_get_full_url", lambda connector_id, cfg: f"sqlite:///{db_path}")

    out = tmp_path / "kb.yaml"
    result = CliRunner().invoke(cli_main.cli, ["export-kb", "c1", "--output", str(out)])
    assert result.exit_code == 0, result.output

    table = yaml.safe_load(out.read_text(encoding="utf-8"))["tables"]["orders"]
    assert table["sample_rows"] == [{"id": 1, "status": "open", "order_total": 12.5}]
    # The columns themselves are still described; only their values are withheld.
    assert [c["name"] for c in table["columns"]] == [
        "id",
        "status",
        "order_total",
        "email",
        "api_key",
        "first_name",
    ]


class _Label(str):
    """A driver's own string type: a `str` subclass, which `safe_dump` refuses."""


class _Ratio(float):
    """A driver's own float type, in the manner of `numpy.float64`."""


def test_the_fallback_export_writes_sample_values_safe_dump_cannot_represent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sqlalchemy

    db_path = tmp_path / "shop.db"
    order_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, total DECIMAL, ref UUID,"
            " status LABEL, share RATIO)"
        )
        conn.execute("INSERT INTO orders VALUES (1, '12.50', ?, 'open', 0.25)", (str(order_id),))

    # Make the driver return these types for these declared column types, as a
    # warehouse driver would for its own NUMERIC, UUID and string types.
    monkeypatch.setitem(sqlite3.converters, "DECIMAL", lambda b: Decimal(b.decode()))
    monkeypatch.setitem(sqlite3.converters, "UUID", lambda b: uuid.UUID(b.decode()))
    monkeypatch.setitem(sqlite3.converters, "LABEL", lambda b: _Label(b.decode()))
    monkeypatch.setitem(sqlite3.converters, "RATIO", lambda b: _Ratio(b.decode()))
    real_create_engine = sqlalchemy.create_engine
    monkeypatch.setattr(
        sqlalchemy,
        "create_engine",
        lambda url: real_create_engine(url, connect_args={"detect_types": sqlite3.PARSE_DECLTYPES}),
    )

    cfg = {"db_type": "unregistered", "database": "shop"}
    monkeypatch.setattr(cli_main, "_resolve_alias", lambda value: value)
    monkeypatch.setattr(cli_main, "_require_connector", lambda connector_id: cfg)
    monkeypatch.setattr(cli_main, "connector_class_for", lambda db_type, cfg: None)
    monkeypatch.setattr(cli_main, "_get_full_url", lambda connector_id, cfg: f"sqlite:///{db_path}")

    out = tmp_path / "kb.yaml"
    result = CliRunner().invoke(cli_main.cli, ["export-kb", "c1", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert result.exception is None

    table = yaml.safe_load(out.read_text(encoding="utf-8"))["tables"]["orders"]
    assert table["sample_rows"] == [
        {"id": 1, "total": "12.5", "ref": str(order_id), "status": "open", "share": "0.25"}
    ]
