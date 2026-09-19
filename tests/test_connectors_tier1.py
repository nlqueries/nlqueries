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

        # `connect()`, not `begin()`: execution must never run inside a block
        # that commits on the way out. See tests/test_connector_read_only.py.
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

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
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_called_once()

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
        """connect() uses ':memory:' when 'database' key is absent -- and the
        sandbox goes on regardless of which database it is."""
        mock_duckdb = MagicMock()
        with patch.dict(sys.modules, {"duckdb": mock_duckdb}):
            from nlqueries.connectors.duckdb import _SANDBOX, DuckDBConnector

            connector = granted(DuckDBConnector())
            connector.connect({})
            mock_duckdb.connect.assert_called_once_with(
                database=":memory:",
                # DuckDB refuses to open an in-memory database read-only, and
                # it is private to this process and gone when it exits, so
                # there is nothing in it to protect.
                read_only=False,
                config=_SANDBOX,
            )
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


class TestSchemaDegradesWithoutConstraintViews:
    """Keys are an enrichment; tables and columns are the schema.

    Observed on Snowflake against a share-imported database: the constraint
    views are absent, and losing them threw away a complete table and column
    listing that had already been read. Redshift restricts the same views on a
    datashare consumer, and a SQL Server login without VIEW DEFINITION can fail
    the query outright, so the same degrade applies to all three.
    """

    def test_redshift_omits_keys_rather_than_failing(self) -> None:
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = MagicMock()
        cur_tables = _make_cursor([("public", "users", 100)])
        cur_cols = _make_cursor([("public", "users", "id", "integer", False)])
        denied = MagicMock()
        denied.execute.side_effect = Exception("permission denied for relation key_column_usage")

        conn.cursor.side_effect = [cur_tables, cur_cols, denied, denied]

        connector = granted(RedshiftConnector())
        connector._conn = conn
        connector._database = "dev"
        spec = connector.extract_schema()

        assert [t.name for t in spec.tables] == ["users"]
        assert [c.name for c in spec.tables[0].columns] == ["id"]
        # Degraded, not invented.
        assert spec.tables[0].columns[0].is_primary_key is False

    @pytest.mark.parametrize("failing_call", ["execute", "fetchall"])
    def test_redshift_closes_its_cursor_however_the_key_query_fails(
        self, failing_call: str
    ) -> None:
        """Both call sites, because the one that matters is `execute`.

        The driver reports a missing or restricted view from `execute`, not from
        the fetch, so a guard starting after it covered only the path that does
        not fail -- and with the caller swallowing the error, each degraded
        extraction leaked a cursor instead of raising.
        """
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = MagicMock()
        getattr(cur, failing_call).side_effect = Exception("permission denied")
        conn = MagicMock()
        conn.cursor.return_value = cur

        with pytest.raises(Exception, match="permission denied"):
            RedshiftConnector._fetch_primary_keys(conn)

        cur.close.assert_called_once()

    def test_redshift_rolls_back_after_a_degraded_enrichment(self) -> None:
        """`Cursor.execute` opens a transaction when autocommit is off, so a
        failed statement leaves the block aborted.

        Swallowed without ending it, every later statement on the connection
        fails with 25P02: the foreign-key fetch is then logged as though the
        datashare caused it, and the `SET TRANSACTION READ ONLY` opening the
        next user query fails for no reason the user can see.
        """
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = MagicMock()
        cur_tables = _make_cursor([("public", "users", 100)])
        cur_cols = _make_cursor([("public", "users", "id", "integer", False)])
        denied = MagicMock()
        denied.execute.side_effect = Exception("permission denied for relation")
        conn.cursor.side_effect = [cur_tables, cur_cols, denied, denied]

        connector = granted(RedshiftConnector())
        connector._conn = conn
        connector._database = "dev"
        connector.extract_schema()

        # Once per failed enrichment, so the next statement starts clean, plus
        # once as `extract_schema` returns -- see
        # TestRedshiftExtractSchemaTransaction for that third one.
        assert conn.rollback.call_count == 3

    def test_mssql_omits_keys_rather_than_failing(self) -> None:
        from nlqueries.connectors.mssql import MSSQLConnector
        from sqlalchemy.exc import DatabaseError

        connector = granted(MSSQLConnector())
        refused = DatabaseError("SELECT ...", {}, Exception("VIEW DEFINITION denied"))

        with (
            patch.object(MSSQLConnector, "_require_engine"),
            patch.object(MSSQLConnector, "_fetch_tables", return_value={("dbo", "users"): 5}),
            patch.object(
                MSSQLConnector,
                "_fetch_columns",
                return_value={
                    ("dbo", "users"): [
                        {"column_name": "id", "data_type": "int", "is_nullable": False}
                    ]
                },
            ),
            patch.object(MSSQLConnector, "_fetch_primary_keys", side_effect=refused),
            patch.object(MSSQLConnector, "_fetch_foreign_keys", side_effect=refused),
        ):
            spec = connector.extract_schema()

        assert [t.name for t in spec.tables] == ["users"]
        assert spec.tables[0].columns[0].is_primary_key is False

    def test_mssql_still_raises_a_fault_that_is_not_the_database_refusing(self) -> None:
        """Only DatabaseError degrades. A bug in this module must still surface."""
        from nlqueries.connectors.mssql import MSSQLConnector

        connector = granted(MSSQLConnector())

        with (
            patch.object(MSSQLConnector, "_require_engine"),
            patch.object(MSSQLConnector, "_fetch_tables", return_value={("dbo", "users"): 5}),
            patch.object(MSSQLConnector, "_fetch_columns", return_value={}),
            patch.object(MSSQLConnector, "_fetch_primary_keys", side_effect=TypeError("bug")),
            pytest.raises(TypeError),
        ):
            connector.extract_schema()


class TestRedshiftRowCountFallback:
    """The documented fallback has to survive the statement that triggers it.

    Reading `SVV_TABLE_INFO` needs a grant most analyst roles are not given, so
    this is the common Redshift path, not an edge case.
    """

    @staticmethod
    def _conn(second_cursor_rows: list[tuple[str, str]]) -> tuple[MagicMock, MagicMock]:
        denied = MagicMock()
        denied.execute.side_effect = Exception("permission denied for relation svv_table_info")
        fallback = _make_cursor(second_cursor_rows)
        conn = MagicMock()
        conn.cursor.side_effect = [denied, fallback]
        return conn, fallback

    def test_falls_back_on_a_fresh_cursor(self) -> None:
        """Reusing the cursor ran the fallback inside the aborted block, so it
        failed with 25P02 and propagated -- the fallback never ran at all."""
        from nlqueries.connectors.redshift import RedshiftConnector

        conn, _ = self._conn([("public", "users"), ("public", "orders")])

        counts = RedshiftConnector._fetch_row_counts(conn)

        assert counts == {("public", "users"): None, ("public", "orders"): None}
        # A second cursor was taken rather than the failed one reused.
        assert conn.cursor.call_count == 2

    def test_rolls_back_before_the_fallback(self) -> None:
        """Before, not merely at some point.

        `assert_called_once` would hold with the rollback moved below the second
        `execute`, which is the 25P02 failure this exists to prevent -- the test
        would stay green while the bug came back. Ordering is the property.
        """
        from nlqueries.connectors.redshift import RedshiftConnector

        conn, _ = self._conn([("public", "users")])

        RedshiftConnector._fetch_row_counts(conn)

        names = [c[0] for c in conn.mock_calls]
        assert names.count("rollback") == 1
        # cursor, cursor().execute, ... rollback ... then the SECOND cursor.
        assert names.index("rollback") < len(names) - 1 - names[::-1].index("cursor")

    def test_rolls_back_when_the_fallback_also_fails(self) -> None:
        """Otherwise the block stays aborted on the way out, and the next
        `execute_query` fails at `SET TRANSACTION READ ONLY` for a reason
        unrelated to the query the user ran."""
        from nlqueries.connectors.redshift import RedshiftConnector

        denied = MagicMock()
        denied.execute.side_effect = Exception("permission denied for relation svv_table_info")
        also_denied = MagicMock()
        also_denied.execute.side_effect = Exception("permission denied for information_schema")
        conn = MagicMock()
        conn.cursor.side_effect = [denied, also_denied]

        with pytest.raises(Exception, match="information_schema"):
            RedshiftConnector._fetch_row_counts(conn)

        # Once before the fallback, once on the way out.
        assert conn.rollback.call_count == 2
        also_denied.close.assert_called_once()

    def test_closes_the_cursor_that_failed(self) -> None:
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = MagicMock()
        denied = MagicMock()
        denied.execute.side_effect = Exception("permission denied")
        fallback = _make_cursor([])
        conn.cursor.side_effect = [denied, fallback]

        RedshiftConnector._fetch_row_counts(conn)

        denied.close.assert_called_once()
        fallback.close.assert_called_once()

    def test_the_happy_path_still_returns_counts(self) -> None:
        """Negative control: the fallback must not have become the only path."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = _make_cursor([("public", "users", 42)])
        conn = MagicMock()
        conn.cursor.return_value = cur

        assert RedshiftConnector._fetch_row_counts(conn) == {("public", "users"): 42}
        conn.rollback.assert_not_called()


class TestRedshiftSiblingTransactions:
    """The other two methods that leave a transaction open.

    Same defect as `extract_schema` had, same cause: every statement opens a
    transaction implicitly and the connector's connection is reused across
    requests, so whatever it leaves open is the next query's problem -- which
    surfaces as `SET TRANSACTION READ ONLY` failing for a reason unrelated to
    the query the user ran.
    """

    def test_test_connection_closes_its_transaction(self) -> None:
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = _make_cursor([(1,)])
        conn = _make_conn(cur)
        connector = granted(RedshiftConnector())
        connector._conn = conn

        assert connector.test_connection() is True

        assert [c[0] for c in conn.mock_calls][-1] == "rollback"

    def test_test_connection_closes_cursor_and_transaction_when_it_fails(self) -> None:
        """The old code closed the cursor inside the `try`, so a failing
        `SELECT 1` leaked the cursor as well as the transaction."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = MagicMock()
        cur.execute.side_effect = Exception("connection reset")
        conn = _make_conn(cur)
        connector = granted(RedshiftConnector())
        connector._conn = conn

        assert connector.test_connection() is False

        cur.close.assert_called_once()
        conn.rollback.assert_called_once()

    def test_extract_query_history_closes_its_transaction(self) -> None:
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = _make_cursor([("SELECT 1", 3, 12.0, "2026-01-01")])
        conn = _make_conn(cur)
        connector = granted(RedshiftConnector())
        connector._conn = conn

        records = connector.extract_query_history()

        # Negative control: ending the transaction did not cost the history.
        assert [r.sql for r in records] == ["SELECT 1"]
        assert [c[0] for c in conn.mock_calls][-1] == "rollback"

    def test_extract_query_history_closes_its_transaction_when_stl_query_is_denied(
        self,
    ) -> None:
        """The swallowed failure leaves the block aborted, so this path needs it
        more than the one that worked."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur = MagicMock()
        cur.execute.side_effect = Exception("permission denied for relation stl_query")
        conn = _make_conn(cur)
        connector = granted(RedshiftConnector())
        connector._conn = conn

        assert connector.extract_query_history() == []

        cur.close.assert_called_once()
        assert [c[0] for c in conn.mock_calls][-1] == "rollback"


class TestRedshiftExtractSchemaTransaction:
    """`extract_schema` must not hand the next query an open transaction.

    `_execute_query`'s own docstring says `SET TRANSACTION READ ONLY` has to be
    the first statement of the transaction it opens. Every `cur.execute` in the
    schema readers opens one implicitly, and the success path returned without
    ending it -- so the first user query on a reused connection was refused for
    a reason that had nothing to do with the query. The failure paths already
    rolled back for their own reasons; the paths that worked did not.
    """

    @staticmethod
    def _conn_for_extract_then_query() -> MagicMock:
        conn = MagicMock()
        cur_tables = _make_cursor([("public", "users", 100)])
        cur_cols = _make_cursor([("public", "users", "id", "integer", False)])
        cur_pks = _make_cursor([("public", "users", "id")])
        cur_fks = _make_cursor([])
        query_cur = MagicMock()
        query_cur.description = [("id",)]
        query_cur.fetchall.return_value = [(1,)]
        query_cur.__iter__.return_value = iter([(1,)])
        conn.cursor.side_effect = [cur_tables, cur_cols, cur_pks, cur_fks, query_cur]
        # `cursor.side_effect` hands back mocks that are not children of `conn`,
        # so their calls never reach `conn.mock_calls`. Attaching the query
        # cursor puts the rollback and the statement that must follow it into
        # one ordered log.
        conn.attach_mock(query_cur, "query_cur")
        return conn

    def test_the_transaction_is_closed_by_the_time_extract_returns(self) -> None:
        """Ordering is the property, so the assertion is on ordering.

        `rollback.assert_called()` would stay green with the rollback moved into
        `execute_query`, which is where it already happens and is too late: the
        query has issued `SET TRANSACTION READ ONLY` by then. Requiring it to be
        the last thing `extract_schema` does pins the fix to the right place.
        """
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = self._conn_for_extract_then_query()
        connector = granted(RedshiftConnector())
        connector._conn = conn
        connector._database = "dev"

        spec = connector.extract_schema()
        during_extract = [c[0] for c in conn.mock_calls]

        # Negative control: ending the transaction must not cost the schema.
        assert [t.name for t in spec.tables] == ["users"]
        assert during_extract[-1] == "rollback"

    def test_the_next_query_can_still_be_made_read_only(self) -> None:
        """The consequence the rollback exists for, asserted end to end."""
        from nlqueries.connectors.redshift import RedshiftConnector

        conn = self._conn_for_extract_then_query()
        connector = granted(RedshiftConnector())
        connector._conn = conn
        connector._database = "dev"

        connector.extract_schema()
        result = connector.execute_query("SELECT id FROM users")

        assert result.error is None
        names = [c[0] for c in conn.mock_calls]
        first_query_stmt = names.index("query_cur.execute")
        # `SET TRANSACTION READ ONLY` is the query's first statement, and the
        # schema extraction closed its transaction before it -- which is the
        # whole point: the server only accepts it as the first statement of a
        # transaction.
        assert conn.mock_calls[first_query_stmt][1][0] == "SET TRANSACTION READ ONLY"
        assert "rollback" in names[:first_query_stmt]

    def test_the_transaction_is_closed_when_the_extract_fails(self) -> None:
        """A raise leaves the connection reusable too."""
        from nlqueries.connectors.redshift import RedshiftConnector

        cur_tables = _make_cursor([("public", "users", 100)])
        broken_cols = MagicMock()
        broken_cols.execute.side_effect = Exception("permission denied for information_schema")
        conn = MagicMock()
        conn.cursor.side_effect = [cur_tables, broken_cols]

        connector = granted(RedshiftConnector())
        connector._conn = conn
        connector._database = "dev"

        with pytest.raises(Exception, match="permission denied"):
            connector.extract_schema()

        assert [c[0] for c in conn.mock_calls][-1] == "rollback"
