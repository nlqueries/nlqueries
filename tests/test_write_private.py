"""
tests.test_write_private
~~~~~~~~~~~~~~~~~~~~~~~~
Knowledge bases, query capsules, the connector registry and ``import-dbt``
output are written through :func:`nlqueries.state_files.write_private`: readable
by the owner only, and replaced atomically, so a symlink at the target is
replaced rather than followed and an interrupted write leaves the previous file
in place.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import sqlalchemy
import yaml
from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.connectors import sqlalchemy_connector as sqlalchemy_connector_module
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector
from nlqueries.knowledge.kb_generator import generate_knowledge_base, save_knowledge_base
from nlqueries.processing.parameterizer import QueryCapsule
from nlqueries.processing.pipeline import save_capsules
from nlqueries.state_files import write_private
from sqlalchemy.sql.elements import quoted_name

from tests.conftest import granted

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# write_private itself
# ---------------------------------------------------------------------------


@posix_only
def test_the_file_is_owner_only(tmp_path: Path) -> None:
    target = write_private(tmp_path / "kb.yaml", "a: 1\n")
    assert target.read_text(encoding="utf-8") == "a: 1\n"
    assert _mode(target) == 0o600


@posix_only
def test_an_existing_readable_file_ends_up_owner_only(tmp_path: Path) -> None:
    target = tmp_path / "kb.yaml"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o644)

    write_private(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert _mode(target) == 0o600


@posix_only
def test_a_missing_parent_is_created_owner_only(tmp_path: Path) -> None:
    target = write_private(tmp_path / "state" / "kb" / "c1.yaml", "a: 1\n")
    assert _mode(target.parent) == 0o700


@posix_only
def test_an_existing_parent_keeps_its_mode(tmp_path: Path) -> None:
    # An --output path the operator chose lives in their directory, which is
    # not this module's to lock down.
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    out_dir.chmod(0o755)

    write_private(out_dir / "kb.yaml", "a: 1\n")

    assert _mode(out_dir) == 0o755


def test_a_symlink_at_the_target_is_replaced_not_followed(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("untouched\n", encoding="utf-8")
    target = tmp_path / "kb.yaml"
    try:
        target.symlink_to(elsewhere)
    except OSError:
        pytest.skip("symlinks are not available here")

    write_private(target, "a: 1\n")

    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "a: 1\n"
    assert elsewhere.read_text(encoding="utf-8") == "untouched\n"


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    target = tmp_path / "kb.yaml"
    target.write_text("old\n", encoding="utf-8")

    def boom(*_args: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, failing, boom)
    with pytest.raises(OSError, match="disk full"):
        write_private(target, "new\n")
    monkeypatch.undo()

    assert target.read_text(encoding="utf-8") == "old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kb.yaml"]


# ---------------------------------------------------------------------------
# The callers
# ---------------------------------------------------------------------------


def _q(name: Any) -> Any:
    return None if name is None else quoted_name(name, None)


class _QuotingInspector:
    """The real inspector, returning identifiers as ``quoted_name``.

    SQLite's dialect hands back plain ``str``; the PostgreSQL, Oracle and
    Snowflake dialects (among others) hand back ``quoted_name``. This makes the
    SQLite fixture behave like them.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def default_schema_name(self) -> Any:
        return _q(self._inner.default_schema_name)

    def get_table_names(self) -> list[Any]:
        return [_q(n) for n in self._inner.get_table_names()]

    def get_columns(self, table_name: str) -> list[dict[str, Any]]:
        return [{**c, "name": _q(c["name"])} for c in self._inner.get_columns(table_name)]

    def get_pk_constraint(self, table_name: str) -> dict[str, Any]:
        pk = self._inner.get_pk_constraint(table_name)
        return {**pk, "constrained_columns": [_q(c) for c in pk["constrained_columns"]]}

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        return [
            {
                **fk,
                "constrained_columns": [_q(c) for c in fk["constrained_columns"]],
                "referred_table": _q(fk["referred_table"]),
                "referred_columns": [_q(c) for c in fk["referred_columns"]],
            }
            for fk in self._inner.get_foreign_keys(table_name)
        ]

    def get_indexes(self, table_name: str) -> list[dict[str, Any]]:
        return [
            {**i, "name": _q(i["name"]), "column_names": [_q(c) for c in i["column_names"]]}
            for i in self._inner.get_indexes(table_name)
        ]


_real_inspect = sqlalchemy.inspect


def _quoting_inspect(subject: Any) -> _QuotingInspector:
    return _QuotingInspector(_real_inspect(subject))


def _shop_db(tmp_path: Path) -> Path:
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY,"
            " customer_id INTEGER REFERENCES customers(id))"
        )
        conn.execute("CREATE INDEX ix_orders_customer ON orders (customer_id)")
    return db


def _reflect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(sqlalchemy_connector_module, "inspect", _quoting_inspect)
    connector = granted(SQLAlchemyConnector())
    connector.connect({"url": f"sqlite:///{_shop_db(tmp_path)}"})
    try:
        return connector.extract_schema()
    finally:
        connector.close()


def test_reflected_identifiers_are_plain_str(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = _reflect(tmp_path, monkeypatch)
    for table in schema.tables:
        assert type(table.name) is str
        assert type(table.schema) is str
        assert all(type(c.name) is str for c in table.columns)
        assert all(c.references is None or type(c.references) is str for c in table.columns)
    orders = next(t for t in schema.tables if t.name == "orders")
    assert [c.name for c in orders.columns if c.is_foreign_key] == ["customer_id"]


def test_a_reflected_table_name_round_trips_through_save_and_safe_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = generate_knowledge_base(_reflect(tmp_path, monkeypatch), [], agent_name="c1")
    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))

    kb = yaml.safe_load(out.read_text(encoding="utf-8"))
    names = [t["name"] for t in kb["schema"]["tables"]]
    assert names == ["customers", "orders"]
    orders = kb["schema"]["tables"][1]
    assert orders["columns"][1]["references"] == "customers.id"


def test_saving_a_knowledge_base_refuses_a_type_safe_load_cannot_read(tmp_path: Path) -> None:
    # The file is written for `safe_load`; an object that would come back as a
    # Python tag is refused at write time, not discovered at read time.
    with pytest.raises(yaml.representer.RepresenterError):
        save_knowledge_base({"db_name": quoted_name("shop", None)}, str(tmp_path / "kb.yaml"))
    assert not (tmp_path / "kb.yaml").exists()


@posix_only
def test_a_saved_knowledge_base_is_owner_only(tmp_path: Path) -> None:
    out = tmp_path / "kb.yaml"
    save_knowledge_base({"kb_version": 2}, str(out))
    assert _mode(out) == 0o600


@posix_only
def test_saved_capsules_are_owner_only(tmp_path: Path) -> None:
    capsule = QueryCapsule(
        template_sql="SELECT * FROM orders WHERE id = :p1",
        placeholders=[],
        tables=["orders"],
        columns=["id"],
        frequency=3,
        auto_description="orders by id",
    )
    with patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path / "capsules"):
        out = save_capsules([capsule], "c1")
    assert _mode(out) == 0o600
    assert json.loads(out.read_text(encoding="utf-8"))[0]["frequency"] == 3


@posix_only
def test_the_connector_registry_is_owner_only_and_replaces_a_symlink(tmp_path: Path) -> None:
    registry = tmp_path / "state" / "connectors.yaml"
    registry.parent.mkdir()
    elsewhere = tmp_path / "elsewhere.yaml"
    elsewhere.write_text("{}\n", encoding="utf-8")
    registry.symlink_to(elsewhere)

    with (
        patch("nlqueries.config.CONNECTORS_FILE", registry),
        patch("nlqueries.cli.main.CONNECTORS_FILE", registry),
    ):
        cli_main._save_connector("c1", {"db_type": "sqlite", "database": "shop.db"})

    assert not registry.is_symlink()
    assert _mode(registry) == 0o600
    assert yaml.safe_load(registry.read_text(encoding="utf-8"))["c1"]["db_type"] == "sqlite"
    assert elsewhere.read_text(encoding="utf-8") == "{}\n"


@posix_only
def test_the_fallback_knowledge_base_is_owner_only_and_safe_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _shop_db(tmp_path)
    # The fallback imports `inspect` from SQLAlchemy when it runs.
    monkeypatch.setattr(sqlalchemy, "inspect", _quoting_inspect)

    cfg = {"db_type": "unregistered", "database": "shop"}
    monkeypatch.setattr(cli_main, "_resolve_alias", lambda value: value)
    monkeypatch.setattr(cli_main, "_require_connector", lambda connector_id: cfg)
    monkeypatch.setattr(cli_main, "connector_class_for", lambda db_type, cfg: None)
    monkeypatch.setattr(cli_main, "_get_full_url", lambda connector_id, cfg: f"sqlite:///{db_path}")

    out = tmp_path / "kb.yaml"
    result = CliRunner().invoke(cli_main.cli, ["export-kb", "c1", "--output", str(out)])
    assert result.exit_code == 0, result.output

    assert _mode(out) == 0o600
    orders = yaml.safe_load(out.read_text(encoding="utf-8"))["tables"]["orders"]
    assert orders["foreign_keys"] == [
        {
            "columns": ["customer_id"],
            "references_table": "customers",
            "references_columns": ["id"],
        }
    ]
    assert orders["indexes"][0]["name"] == "ix_orders_customer"


@posix_only
def test_import_dbt_output_is_owner_only(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.yaml"
    kb_path.write_text(
        yaml.safe_dump({"schema": {"tables": [{"name": "orders", "columns": []}]}}),
        encoding="utf-8",
    )
    out = tmp_path / "merged.yaml"
    manifest = Path(__file__).parent / "fixtures" / "dbt" / "manifest.json"

    result = CliRunner().invoke(
        cli_main.cli, ["import-dbt", str(manifest), "--kb", str(kb_path), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert _mode(out) == 0o600
