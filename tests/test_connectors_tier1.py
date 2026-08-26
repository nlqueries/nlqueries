"""Unit tests for Tier-1 database connectors: Redshift, MSSQL, DuckDB (#28)."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import granted

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cursor(
    rows: list[tuple[Any, ...]], description: list[tuple[str, ...]] | None = None
) -> MagicMock:
    """Return a mock cursor that yields *rows* from fetchall() and *description*."""
    cur = MagicMock()
    cur.fetchall.return_value = rows
    cur.fetchone.return_value = rows[0] if rows else None
    cur.description = description
    return cur


def _make_conn(cursor: MagicMock | None = None) -> MagicMock:
    conn = MagicMock()
    if cursor is not None:
        conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# RedshiftConnector
# ---------------------------------------------------------------------------


class TestRedshiftConnector:
    def test_connect_raises_without_driver(self) -> None:
        """connect() raises ImportError with install hint when redshift_connector is missing."""
        from nlqueries.connectors.redshift import RedshiftConnector

        with patch.dict(sys.modules, {"redshift_connector": None}):
            connector = granted(RedshiftConnector())
            with pytest.raises(ImportError, match="redshift-connector"):
                connector.connect(
                    {"host": "x", "port": 5439, "database": "dev", "user": "u", "password": "p"}
                )

    def test_test_connection_true(self) -> None:
        """test_connection returns True when cursor.execute succeeds."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = _make_cursor([(1,)])
        connector = granted(RedshiftConnector())
        connector._conn = _make_conn(cur)
        assert connector.test_connection() is True

    def test_test_connection_false_on_exception(self) -> None:
        """test_connection returns False when the cursor raises."""
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = MagicMock()
        conn.cursor.side_effect = Exception("connection lost")
        connector = granted(RedshiftConnector())
        connector._conn = conn
        assert connector.test_connection() is False

    def test_extract_schema_builds_spec(self) -> None:
        """extract_schema returns a SchemaSpec with the expected table and column."""
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = MagicMock()
        # SVV_TABLE_INFO row: (schema, table, row_count)
        cur_tables = _make_cursor([("public", "users", 100)])
        # columns row: (schema, table, column_name, data_type, is_nullable_bool)
        cur_cols = _make_cursor([("public", "users", "id", "integer", False)])
        # primary-key row: (schema, table, column_name)
        cur_pks = _make_cursor([("public", "users", "id")])
        # foreign-key row: empty
        cur_fks = _make_cursor([])

        conn.cursor.side_effect = [cur_tables, cur_cols, cur_pks, cur_fks]

        connector = granted(RedshiftConnector())
        connector._conn = conn
        connector._database = "dev"
        spec = connector.extract_schema()

        assert spec.database == "dev"
        assert len(spec.tables) == 1
        tbl = spec.tables[0]
        assert tbl.name == "users"
        assert tbl.row_count == 100
        assert len(tbl.columns) == 1
        assert tbl.columns[0].name == "id"
        assert tbl.columns[0].is_primary_key is True
        assert tbl.columns[0].is_foreign_key is False

    def test_execute_query_returns_result(self) -> None:
        """execute_query returns a QueryResult with rows and column names."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = MagicMock()
        cur.description = [("id",), ("name",)]
        cur.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        cur.__iter__.return_value = iter([(1, "Alice"), (2, "Bob")])
        conn = _make_conn(cur)

        connector = granted(RedshiftConnector())
        connector._conn = conn
        result = connector.execute_query("SELECT id, name FROM users")

        assert result.error is None
        assert result.columns == ["id", "name"]
        assert result.row_count == 2
        assert result.rows == [[1, "Alice"], [2, "Bob"]]

    def test_execute_query_surfaces_error(self) -> None:
        """execute_query surfaces driver errors via QueryResult.error."""
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = MagicMock()
        conn.cursor.side_effect = Exception("syntax error")

        connector = granted(RedshiftConnector())
        connector._conn = conn
        result = connector.execute_query("BAD SQL")

        assert result.error is not None
        assert "syntax error" in result.error
        assert result.row_count == 0

    def test_extract_query_history_returns_records(self) -> None:
        """extract_query_history returns QueryRecord objects from STL_QUERY rows."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = _make_cursor([("SELECT * FROM users", 10, 42.5, "2026-06-01 10:00:00")])
        connector = granted(RedshiftConnector())
        connector._conn = _make_conn(cur)
        records = connector.extract_query_history(days=30, limit=100)

        assert len(records) == 1
        assert records[0].sql == "SELECT * FROM users"
        assert records[0].execution_count == 10
        assert records[0].avg_duration_ms == pytest.approx(42.5)

    def test_extract_query_history_empty_on_permission_error(self) -> None:
        """extract_query_history returns [] when STL_QUERY is inaccessible."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = MagicMock()
        cur.execute.side_effect = Exception("permission denied")
        connector = granted(RedshiftConnector())
        connector._conn = _make_conn(cur)
        records = connector.extract_query_history()
        assert records == []


# ---------------------------------------------------------------------------
# MSSQLConnector
# ---------------------------------------------------------------------------


class TestMSSQLConnector:
    def test_connect_raises_without_driver(self) -> None:
        """connect() raises ImportError with install hint when pymssql is missing."""
        from nlqueries.connectors.mssql import MSSQLConnector

        with patch.dict(sys.modules, {"pymssql": None}):
            connector = granted(MSSQLConnector())
            with pytest.raises(ImportError, match="pymssql"):
                connector.connect(
                    {"host": "srv", "port": 1433, "database": "mydb", "user": "u", "password": "p"}
                )

    def test_test_connection_true(self) -> None:
        """test_connection returns True when SELECT 1 succeeds."""
        from nlqueries.connectors.mssql import MSSQLConnector

        mock_engine = MagicMock()
        mock_conn_ctx = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn_ctx
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        connector = granted(MSSQLConnector())
        connector._engine = mock_engine
        assert connector.test_connection() is True

    def test_test_connection_false_on_exception(self) -> None:
        """test_connection returns False when the engine raises."""
        from nlqueries.connectors.mssql import MSSQLConnector

        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("server unreachable")

        connector = granted(MSSQLConnector())
        connector._engine = mock_engine
        assert connector.test_connection() is False

    def test_extract_schema_builds_spec(self) -> None:
        """extract_schema returns a SchemaSpec populated from mocked query results."""

        from nlqueries.connectors.mssql import MSSQLConnector

        # _fetch_tables → (schema, name, row_count)
        tables_rows = [MagicMock()]
        tables_rows[0].__getitem__ = lambda s, k: {
            "table_schema": "dbo",
            "table_name": "orders",
            "row_count": 500,
        }[k]

        # _fetch_columns → (TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, is_nullable)
        col_rows = [MagicMock()]
        col_rows[0].__getitem__ = lambda s, k: {
            "TABLE_SCHEMA": "dbo",
            "TABLE_NAME": "orders",
            "COLUMN_NAME": "order_id",
            "DATA_TYPE": "int",
            "is_nullable": 0,
        }[k]

        # Use MappingResult-like behaviour
        def _mappings_tables() -> Any:
            m = MagicMock()
            m.__iter__ = lambda s: iter(tables_rows)
            return m

        def _mappings_cols() -> Any:
            m = MagicMock()
            m.__iter__ = lambda s: iter(col_rows)
            return m

        def _mappings_empty() -> Any:
            m = MagicMock()
            m.__iter__ = lambda s: iter([])
            return m

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # DB_NAME() scalar
        mock_conn.execute.return_value.scalar_one.return_value = "mydb"

        # Override _fetch_* staticmethods
        connector = granted(MSSQLConnector())
        connector._engine = mock_engine

        with (
            patch.object(
                MSSQLConnector,
                "_fetch_tables",
                return_value={("dbo", "orders"): 500},
            ),
            patch.object(
                MSSQLConnector,
                "_fetch_columns",
                return_value={
                    ("dbo", "orders"): [
                        {"column_name": "order_id", "data_type": "int", "is_nullable": False}
                    ]
                },
            ),
            patch.object(
                MSSQLConnector,
                "_fetch_primary_keys",
                return_value={("dbo", "orders"): {"order_id"}},
            ),
            patch.object(MSSQLConnector, "_fetch_foreign_keys", return_value={}),
        ):
            spec = connector.extract_schema()

        assert spec.database == "mydb"
        assert len(spec.tables) == 1
        assert spec.tables[0].name == "orders"
        assert spec.tables[0].columns[0].is_primary_key is True

    def test_execute_query_returns_result(self) -> None:
        """execute_query returns rows when cursor_result has data."""
        from nlqueries.connectors.mssql import MSSQLConnector

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = lambda s: mock_conn
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        cursor_result = MagicMock()
        cursor_result.returns_rows = True
        cursor_result.keys.return_value = ["id", "name"]
        cursor_result.fetchall.return_value = [(1, "Alice")]
        cursor_result.__iter__.return_value = iter([(1, "Alice")])
        mock_conn.execute.return_value = cursor_result

        connector = granted(MSSQLConnector())
        connector._engine = mock_engine
        result = connector.execute_query("SELECT id, name FROM users")

        assert result.error is None
        assert result.columns == ["id", "name"]
        assert result.rows == [[1, "Alice"]]

    def test_extract_query_history_empty_on_permission_error(self) -> None:
        """extract_query_history returns [] when the DMV is inaccessible."""
        from nlqueries.connectors.mssql import MSSQLConnector

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = Exception("VIEW SERVER STATE required")

        connector = granted(MSSQLConnector())
        connector._engine = mock_engine
        records = connector.extract_query_history()
        assert records == []


# ---------------------------------------------------------------------------
# DuckDBConnector
# ---------------------------------------------------------------------------


class TestDuckDBConnector:
    def test_connect_raises_without_driver(self) -> None:
        """connect() raises ImportError with install hint when duckdb is missing."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        with patch.dict(sys.modules, {"duckdb": None}):
            connector = granted(DuckDBConnector())
            with pytest.raises(ImportError, match="duckdb"):
                connector.connect({"database": ":memory:"})

    def test_connect_defaults_to_in_memory(self) -> None:
        """connect() uses ':memory:' when 'database' key is absent."""
        mock_duckdb = MagicMock()
        with patch.dict(sys.modules, {"duckdb": mock_duckdb}):
            from nlqueries.connectors.duckdb import DuckDBConnector

            connector = granted(DuckDBConnector())
            connector.connect({})
            mock_duckdb.connect.assert_called_once_with(database=":memory:")
            assert connector._database == ":memory:"

    def test_test_connection_true(self) -> None:
        """test_connection returns True when SELECT 1 executes."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result

        connector = granted(DuckDBConnector())
        connector._conn = mock_conn
        assert connector.test_connection() is True

    def test_test_connection_false_on_exception(self) -> None:
        """test_connection returns False when execute raises."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("file not found")

        connector = granted(DuckDBConnector())
        connector._conn = mock_conn
        assert connector.test_connection() is False

    def test_extract_query_history_always_empty(self) -> None:
        """extract_query_history always returns an empty list (no DuckDB history)."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        connector = granted(DuckDBConnector())
        connector._conn = MagicMock()
        assert connector.extract_query_history() == []

    def test_execute_query_returns_result(self) -> None:
        """execute_query returns rows and columns from duckdb result."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        mock_result = MagicMock()
        mock_result.description = [("n",), ("label",)]
        mock_result.fetchall.return_value = [(42, "hello")]
        mock_result.__iter__.return_value = iter([(42, "hello")])

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result

        connector = granted(DuckDBConnector())
        connector._conn = mock_conn
        result = connector.execute_query("SELECT 42 AS n, 'hello' AS label")

        assert result.error is None
        assert result.columns == ["n", "label"]
        assert result.rows == [[42, "hello"]]
        assert result.row_count == 1

    def test_execute_query_surfaces_error(self) -> None:
        """execute_query surfaces exceptions via QueryResult.error."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("Parser error")

        connector = granted(DuckDBConnector())
        connector._conn = mock_conn
        result = connector.execute_query("BAD SQL")

        assert result.error is not None
        assert "Parser error" in result.error
        assert result.row_count == 0

    def test_extract_schema_builds_spec(self) -> None:
        """extract_schema returns a SchemaSpec with tables and columns."""
        from nlqueries.connectors.duckdb import DuckDBConnector

        connector = granted(DuckDBConnector())
        connector._database = ":memory:"

        # duckdb_tables() rows: (schema_name, table_name, estimated_size)
        tables_result = MagicMock()
        tables_result.fetchall.return_value = [("main", "sales", 1000)]
        tables_result.__iter__.return_value = iter([("main", "sales", 1000)])

        # information_schema.columns rows: (schema, table, col, dtype, nullable)
        cols_result = MagicMock()
        cols_result.fetchall.return_value = [("main", "sales", "amount", "DOUBLE", "NO")]
        cols_result.__iter__.return_value = iter([("main", "sales", "amount", "DOUBLE", "NO")])

        # duckdb_constraints() rows: (schema, table, col_names_list)
        pk_result = MagicMock()
        pk_result.fetchall.return_value = [("main", "sales", ["amount"])]
        pk_result.__iter__.return_value = iter([("main", "sales", ["amount"])])

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [tables_result, cols_result, pk_result]

        connector._conn = mock_conn
        spec = connector.extract_schema()

        assert spec.database == ":memory:"
        assert len(spec.tables) == 1
        tbl = spec.tables[0]
        assert tbl.name == "sales"
        assert tbl.row_count == 1000
        assert tbl.columns[0].name == "amount"
        assert tbl.columns[0].is_primary_key is True


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestConnectorRegistry:
    def test_registry_always_importable(self) -> None:
        """CONNECTOR_REGISTRY is always importable even without optional extras."""
        from nlqueries.connectors import CONNECTOR_REGISTRY

        assert "postgres" in CONNECTOR_REGISTRY
        assert "snowflake" in CONNECTOR_REGISTRY
        assert "bigquery" in CONNECTOR_REGISTRY

    def test_optional_connectors_registered_when_available(self) -> None:
        """When connector files are importable, their types appear in the registry."""
        from nlqueries.connectors import CONNECTOR_REGISTRY
        from nlqueries.connectors.duckdb import DuckDBConnector
        from nlqueries.connectors.mssql import MSSQLConnector
        from nlqueries.connectors.redshift import RedshiftConnector

        # The module files are present so they should be registered regardless of
        # whether their optional drivers are installed (drivers are imported lazily
        # inside connect(), not at module level).
        assert CONNECTOR_REGISTRY.get("redshift") is RedshiftConnector
        assert CONNECTOR_REGISTRY.get("mssql") is MSSQLConnector
        assert CONNECTOR_REGISTRY.get("duckdb") is DuckDBConnector
