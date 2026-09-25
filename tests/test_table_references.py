"""
tests.test_table_references
~~~~~~~~~~~~~~~~~~~~~~~~~~~
How a table is named in SQL the product builds itself.

A bare name resolves only in the connection's default schema. On Snowflake it
did not: `SELECT * FROM CALL_CENTER` failed where `TPCDS_SF10TCL.CALL_CENTER`
worked, and sampling for column descriptions skipped every table silently.
"""

from __future__ import annotations

from typing import Any

import pytest
from nlqueries.connectors.base import qualified_table_sql, table_sample_sql


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("snowflake", '"TPCDS_SF10TCL"."CALL_CENTER"'),
        ("postgresql", '"TPCDS_SF10TCL"."CALL_CENTER"'),  # a db_type sqlglot spells differently
        ("mysql", "`TPCDS_SF10TCL`.`CALL_CENTER`"),
        ("bigquery", "`TPCDS_SF10TCL`.`CALL_CENTER`"),
        ("mssql", "[TPCDS_SF10TCL].[CALL_CENTER]"),
        (None, '"TPCDS_SF10TCL"."CALL_CENTER"'),
    ],
)
def test_a_table_is_named_with_its_schema_quoted_for_the_dialect(
    dialect: str | None, expected: str
) -> None:
    assert qualified_table_sql("CALL_CENTER", "TPCDS_SF10TCL", dialect) == expected


def test_the_catalogue_s_case_is_kept() -> None:
    """Quoted, so a mixed-case name is not folded to lower case and missed."""
    assert qualified_table_sql("CallCenter", "Sales", "postgresql") == '"Sales"."CallCenter"'


def test_a_quote_inside_a_name_is_escaped() -> None:
    assert qualified_table_sql('odd"name', "s") == '"s"."odd""name"'


@pytest.mark.parametrize("schema", ["", None])
def test_without_a_schema_the_name_stands_alone(schema: str | None) -> None:
    """A knowledge base written before `schema` was recorded."""
    assert qualified_table_sql("orders", schema, "snowflake") == '"orders"'


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("snowflake", 'SELECT * FROM "TPCDS_SF10TCL"."CALL_CENTER" LIMIT 100'),
        ("mysql", "SELECT * FROM `TPCDS_SF10TCL`.`CALL_CENTER` LIMIT 100"),
        # SQL Server has no LIMIT; the string this replaces was invalid T-SQL.
        ("mssql", "SELECT TOP 100 * FROM [TPCDS_SF10TCL].[CALL_CENTER]"),
    ],
)
def test_the_sample_query_is_qualified_and_bounded_for_the_dialect(
    dialect: str, expected: str
) -> None:
    assert table_sample_sql("CALL_CENTER", "TPCDS_SF10TCL", 100, dialect) == expected


@pytest.mark.parametrize("dialect", ["sqlalchemy", "not-a-grammar"])
def test_a_dialect_sqlglot_does_not_know_renders_as_ansi(dialect: str) -> None:
    """The generic connector's db_type is `sqlalchemy`, which names no grammar.
    sqlglot raises on it; these helpers are public, so they fall back instead."""
    assert qualified_table_sql("orders", "sales", dialect) == '"sales"."orders"'
    assert (
        table_sample_sql("orders", "sales", 5, dialect) == 'SELECT * FROM "sales"."orders" LIMIT 5'
    )


# ---------------------------------------------------------------------------
# `export-kb --describe-columns`, driven through the command itself
# ---------------------------------------------------------------------------


def _export_kb_with_descriptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, cfg: dict[str, Any]
) -> tuple[Any, list[str]]:
    from click.testing import CliRunner
    from nlqueries.cli import main as cli_main
    from nlqueries.connectors.base import ColumnSpec, QueryResult, SchemaSpec, TableSpec

    queries: list[str] = []

    class _Connector:
        def connect(self, credentials: Any) -> None:
            pass

        def extract_schema(self) -> SchemaSpec:
            column = ColumnSpec("region", "text", True, False, False, None, None)
            table = TableSpec("orders", "sales", None, [column], None)
            return SchemaSpec(database="db", tables=[table], extracted_at="2026-09-25T00:00:00")

        def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> QueryResult:
            queries.append(sql)
            return QueryResult(["region"], [["north"]], 1, 0.0, None)

    class _LLM:
        def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
            return '{"region": "Sales region of the order"}'

    def _no_capsules(connector_id: str) -> list[Any]:
        raise FileNotFoundError(connector_id)

    monkeypatch.setattr(cli_main, "_resolve_alias", lambda value: value)
    monkeypatch.setattr(cli_main, "_require_connector", lambda connector_id: cfg)
    monkeypatch.setattr(cli_main, "connector_class_for", lambda db_type, cfg: _Connector)
    monkeypatch.setattr(cli_main, "credentials_for", lambda connector_id, cfg: {})
    monkeypatch.setattr("nlqueries.config.llm_credentials_available", lambda: True)
    monkeypatch.setattr("nlqueries.llm.get_llm_client", lambda *a, **k: _LLM())
    monkeypatch.setattr("nlqueries.processing.pipeline.load_capsules", _no_capsules)
    monkeypatch.setattr("nlqueries.feedback.store.load_feedback", lambda connector_id: [])

    result = CliRunner().invoke(
        cli_main.cli,
        ["export-kb", "c1", "--output", str(tmp_path / "kb.yaml"), "--describe-columns"],
    )
    return result, queries


def test_export_kb_samples_a_generic_connector_in_its_url_s_grammar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Before, `sqlalchemy` reached sqlglot, which raised, and the command
    reported "Knowledge base generation failed" and wrote no knowledge base."""
    cfg = {"db_type": "sqlalchemy", "url": "mysql+pymysql://u@h:3306/db"}

    result, queries = _export_kb_with_descriptions(monkeypatch, tmp_path, cfg)

    assert result.exit_code == 0, result.output
    assert queries == ["SELECT * FROM `sales`.`orders` LIMIT 3"]
    assert "LLM described 1 column(s)" in result.output
    assert (tmp_path / "kb.yaml").exists()


def test_export_kb_samples_a_named_connector_in_its_own_grammar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    cfg = {"db_type": "snowflake", "url": "snowflake://u@acct/db"}

    result, queries = _export_kb_with_descriptions(monkeypatch, tmp_path, cfg)

    assert result.exit_code == 0, result.output
    assert queries == ['SELECT * FROM "sales"."orders" LIMIT 3']
