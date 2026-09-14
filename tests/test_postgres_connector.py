"""
Tests for PostgresConnector (nlqueries.connectors.postgres).

These are integration tests backed by a real, ephemeral PostgreSQL instance
spun up in Docker via testcontainers. They are skipped automatically when
Docker is not available (e.g. in CI environments without a Docker daemon).

SSL unit tests live in test_postgres_ssl.py — they use unittest.mock and
require no live database.
"""

from __future__ import annotations

import logging

import pytest
from nlqueries import config
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import ColumnSpec, QueryResult, SchemaSpec, TableSpec
from nlqueries.connectors.postgres import PostgresConnector
from sqlalchemy import text as sa_text

from tests.conftest import granted

testcontainers_postgres = pytest.importorskip(
    "testcontainers.postgres", reason="testcontainers[postgres] is not installed"
)
from testcontainers.postgres import PostgresContainer  # noqa: E402


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker is not available in this environment"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_container():
    container = PostgresContainer("postgres:16-alpine")
    try:
        container.start()
    except Exception as exc:
        pytest.skip(f"Could not start Postgres container: {exc}")
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="module")
def credentials(pg_container) -> dict:
    return {
        "host": pg_container.get_container_host_ip(),
        "port": int(pg_container.get_exposed_port(5432)),
        "database": pg_container.dbname,
        "user": pg_container.username,
        "password": pg_container.password,
        # The throwaway container serves no TLS, and the connector now requires
        # it by default rather than silently accepting a plaintext session. So
        # the fixture says so — which is the whole point of the change: the old
        # default did this quietly, and nothing anywhere recorded that the
        # connection was in the clear.
        "ssl_mode": "disable",
    }


@pytest.fixture()
def connector(credentials) -> PostgresConnector:
    c = granted(PostgresConnector())
    c.connect(credentials)
    return c


@pytest.fixture(scope="module")
def seeded_connector(credentials):
    """A connected PostgresConnector with a small schema (PK + FK) created."""
    c = granted(PostgresConnector())
    c.connect(credentials)

    setup_statements = [
        "DROP TABLE IF EXISTS order_items",
        "DROP TABLE IF EXISTS orders",
        "DROP TABLE IF EXISTS customers",
        """
        CREATE TABLE customers (
            id SERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            name TEXT
        )
        """,
        """
        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            total NUMERIC(10, 2) NOT NULL
        )
        """,
        """
        CREATE TABLE order_items (
            id SERIAL PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            sku TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1
        )
        """,
        "INSERT INTO customers (email, name) "
        "VALUES ('a@example.com', 'Alice'), ('b@example.com', 'Bob')",
        "INSERT INTO orders (customer_id, total) VALUES (1, 19.99), (2, 5.00)",
        "INSERT INTO order_items (order_id, sku, quantity) VALUES (1, 'WIDGET-1', 2)",
        "ANALYZE",
    ]
    # Through the engine, not execute_query(). execute_query() is the path that
    # runs LLM-generated SQL and is deliberately read-only (SEC-01); setting a
    # fixture up through it would be asking the untrusted-input seam to do
    # administration, and would only pass because the guard was missing.
    with c._require_engine().begin() as conn:
        for stmt in setup_statements:
            conn.execute(sa_text(stmt))

    return c


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_postgres_is_registered_under_postgres_key():
    assert CONNECTOR_REGISTRY["postgres"] is PostgresConnector


# ---------------------------------------------------------------------------
# connect / test_connection
# ---------------------------------------------------------------------------


def test_connect_builds_engine_and_test_connection_succeeds(credentials):
    connector = granted(PostgresConnector())
    connector.connect(credentials)
    assert connector.test_connection() is True


def test_test_connection_returns_false_for_bad_credentials(credentials):
    bad_credentials = {**credentials, "password": "definitely-not-the-password"}
    connector = granted(PostgresConnector())
    connector.connect(bad_credentials)
    assert connector.test_connection() is False


def test_methods_behave_before_connect_is_called():
    connector = granted(PostgresConnector())

    # _require_engine() raises directly...
    with pytest.raises(RuntimeError):
        connector._require_engine()

    # ...but test_connection() and execute_query() catch it and surface it
    # gracefully (False / QueryResult.error) rather than propagating.
    assert connector.test_connection() is False

    result = connector.execute_query("SELECT 1")
    assert result.error is not None
    assert "connect()" in result.error


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


def test_execute_query_returns_columns_and_rows(connector):
    result = connector.execute_query("SELECT 1 AS one, 'two' AS two")

    assert isinstance(result, QueryResult)
    assert result.error is None
    assert result.columns == ["one", "two"]
    assert result.rows == [[1, "two"]]
    assert result.row_count == 1
    assert result.execution_time_ms >= 0


def test_execute_query_captures_errors_without_raising(connector):
    result = connector.execute_query("SELECT * FROM this_table_does_not_exist")

    assert isinstance(result, QueryResult)
    assert result.error is not None
    assert "this_table_does_not_exist" in result.error
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0


def test_execute_query_honors_timeout_seconds_via_statement_timeout(connector):
    """Task 26.5: a tiny timeout_seconds aborts a slow query server-side via
    ``SET LOCAL statement_timeout`` rather than letting it run to completion."""
    result = connector.execute_query("SELECT pg_sleep(2)", timeout_seconds=0.05)

    assert result.error is not None
    assert "statement timeout" in result.error.lower()


def test_execute_query_without_timeout_completes_within_default(connector):
    """No explicit timeout: a quick query still runs to completion (well under the
    default CONNECTOR_STATEMENT_TIMEOUT_SECONDS budget)."""
    result = connector.execute_query("SELECT pg_sleep(0.1)")

    assert result.error is None


def test_execute_query_applies_default_statement_timeout(connector, monkeypatch):
    """A query with no explicit timeout is still bounded by the config default, so
    a runaway query fails fast instead of hanging indefinitely."""
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0.05)
    result = connector.execute_query("SELECT pg_sleep(2)")

    assert result.error is not None
    assert "statement timeout" in result.error.lower()


def test_execute_query_default_timeout_disabled_runs_unbounded(connector, monkeypatch):
    """A config default of 0 disables the timeout — the query runs unbounded."""
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0.0)
    result = connector.execute_query("SELECT pg_sleep(0.2)")

    assert result.error is None


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------


def test_extract_schema_returns_full_schema_spec(seeded_connector):
    schema = seeded_connector.extract_schema()

    assert isinstance(schema, SchemaSpec)
    assert schema.database == seeded_connector._require_engine().url.database
    assert schema.extracted_at  # non-empty ISO timestamp

    tables_by_name = {t.name: t for t in schema.tables}
    assert {"customers", "orders", "order_items"} <= set(tables_by_name)

    customers = tables_by_name["customers"]
    assert isinstance(customers, TableSpec)
    assert customers.schema == "public"

    customers_columns = {c.name: c for c in customers.columns}
    assert isinstance(customers_columns["id"], ColumnSpec)
    assert customers_columns["id"].is_primary_key is True
    assert customers_columns["id"].is_foreign_key is False
    assert customers_columns["email"].nullable is False

    orders = tables_by_name["orders"]
    orders_columns = {c.name: c for c in orders.columns}
    assert orders_columns["customer_id"].is_foreign_key is True
    assert orders_columns["customer_id"].references == "customers.id"
    assert orders_columns["id"].is_primary_key is True


def test_extract_schema_row_counts_are_present_after_analyze(seeded_connector):
    schema = seeded_connector.extract_schema()
    tables_by_name = {t.name: t for t in schema.tables}

    # row_count comes from pg_class.reltuples (an estimate refreshed by ANALYZE),
    # so it should be a non-negative number once the table has been analysed.
    for name in ("customers", "orders", "order_items"):
        assert tables_by_name[name].row_count is not None
        assert tables_by_name[name].row_count >= 0


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------


def test_extract_query_history_returns_empty_list_when_extension_missing(connector, caplog):
    # The default postgres:16-alpine image does not ship pg_stat_statements,
    # so this exercises the "extension missing" graceful-degradation path.
    with caplog.at_level(logging.WARNING, logger="nlqueries.connectors.postgres"):
        history = connector.extract_query_history(days=30)

    assert history == []
    assert any("pg_stat_statements" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# list_security_policies (RLS introspection — Block G)
# ---------------------------------------------------------------------------


def test_list_security_policies_finds_rls_policies(connector):
    # Create a table with two RLS policies against the real database.
    setup = [
        "DROP TABLE IF EXISTS rls_orders",
        "CREATE TABLE rls_orders (id SERIAL PRIMARY KEY, region TEXT, amount NUMERIC)",
        "ALTER TABLE rls_orders ENABLE ROW LEVEL SECURITY",
        "CREATE POLICY emea_only ON rls_orders USING (region = 'EMEA')",
        # A WITH CHECK-only (write-time) policy has no USING clause → must be skipped.
        "CREATE POLICY insert_guard ON rls_orders FOR INSERT WITH CHECK (amount > 0)",
    ]
    with connector._require_engine().begin() as conn:
        for stmt in setup:
            conn.execute(sa_text(stmt))

    report = connector.list_security_policies()
    assert report.supported is True
    by_name = {p.name: p for p in report.policies if p.table == "rls_orders"}

    assert "emea_only" in by_name
    emea = by_name["emea_only"]
    assert emea.kind == "row"
    assert emea.table_schema == "public"
    assert "region" in emea.expression and "EMEA" in emea.expression
    # The WITH CHECK-only policy has no read predicate → not surfaced.
    assert "insert_guard" not in by_name

    with connector._require_engine().begin() as conn:
        conn.execute(sa_text("DROP TABLE IF EXISTS rls_orders"))


def test_list_security_policies_empty_when_no_policies(seeded_connector):
    report = seeded_connector.list_security_policies()
    assert report.supported is True
    # The seeded schema (customers/orders/order_items) has no RLS policies.
    assert all(p.table not in {"customers", "orders", "order_items"} for p in report.policies)


# ---------------------------------------------------------------------------
# Read-only execution (SEC-01)
# ---------------------------------------------------------------------------
#
# Generated SQL used to run inside `engine.begin()`, which opens a transaction
# and COMMITS it. Nothing constrained the statement to be side-effect-free, so a
# volatile function that writes — reached through a plain SELECT, which every
# validator here accepts — committed its write. Found independently by the
# 2026-07-02 core review (finding 1) and the 2026-08-25 audit (NLQ-006).
#
# These prove the effect is absent, not merely that a statement was issued: the
# marker table is read back afterwards.


@pytest.fixture()
def marker_table(connector):
    """A table a SELECT should never be able to write to, and a function that tries."""
    engine = connector._require_engine()
    with engine.begin() as conn:
        conn.execute(sa_text("DROP TABLE IF EXISTS sec01_marker CASCADE"))
        conn.execute(sa_text("CREATE TABLE sec01_marker (note text)"))
        conn.execute(
            sa_text(
                """
                CREATE OR REPLACE FUNCTION sec01_mark(note text) RETURNS text AS $$
                BEGIN
                    INSERT INTO sec01_marker VALUES (note);
                    RETURN note;
                END;
                $$ LANGUAGE plpgsql VOLATILE
                """
            )
        )
        conn.execute(sa_text("DROP SEQUENCE IF EXISTS sec01_seq"))
        conn.execute(sa_text("CREATE SEQUENCE sec01_seq"))
    yield
    with engine.begin() as conn:
        conn.execute(sa_text("DROP FUNCTION IF EXISTS sec01_mark(text)"))
        conn.execute(sa_text("DROP TABLE IF EXISTS sec01_marker"))
        conn.execute(sa_text("DROP SEQUENCE IF EXISTS sec01_seq"))


def _marker_rows(connector) -> int:
    with connector._require_engine().begin() as conn:
        return int(conn.execute(sa_text("SELECT count(*) FROM sec01_marker")).scalar_one())


def test_a_select_calling_a_writing_function_commits_nothing(connector, marker_table):
    """The audit's own payload shape: a SELECT whose function writes.

    Every validator in this codebase accepts it — the root node is a Select. The
    database is the layer that has to refuse, and this is the test that says so.
    """
    result = connector.execute_query("SELECT sec01_mark('sec-01')")

    assert result.error is not None
    assert "read-only" in result.error.lower()
    assert _marker_rows(connector) == 0


def test_sequence_functions_are_refused(connector, marker_table):
    """`nextval` advances state on disk; PostgreSQL guards it explicitly."""
    result = connector.execute_query("SELECT nextval('sec01_seq')")

    assert result.error is not None
    assert "read-only" in result.error.lower()


def test_direct_dml_is_refused(connector, marker_table):
    """Not reachable through the validators today, but the boundary must not
    depend on that remaining true."""
    result = connector.execute_query("INSERT INTO sec01_marker VALUES ('direct')")

    assert result.error is not None
    assert _marker_rows(connector) == 0


def test_ordinary_reads_are_unaffected(connector, marker_table):
    """The control has to be invisible to every legitimate query, or it will be
    turned off by whoever meets the first false refusal."""
    result = connector.execute_query("SELECT count(*) AS n, max(note) AS m FROM sec01_marker")

    assert result.error is None
    assert result.columns == ["n", "m"]


def test_explain_analyze_of_a_select_still_works(connector, marker_table):
    """`query_analyzer` runs EXPLAIN (ANALYZE) through this path. PostgreSQL
    permits it for a read-only statement, and this pins that it stays permitted."""
    result = connector.execute_query(
        "EXPLAIN (ANALYZE, FORMAT JSON) SELECT count(*) FROM sec01_marker"
    )

    assert result.error is None


# ---------------------------------------------------------------------------
# Query-history extraction: what the bounded budget is spent on
# ---------------------------------------------------------------------------

#: (statement, is it a read the extraction should keep)
_HISTORY_STATEMENTS: list[tuple[str, bool]] = [
    ("SELECT a FROM t", True),
    ("select a from t", True),
    ("WITH x AS (SELECT 1) SELECT * FROM x", True),
    # The SQL Console tags every run with its cancellation marker, so this is
    # the shape of our own traffic. Anchoring on the first character discarded
    # all of it.
    ("/* nlq-console:2a2e3031-f2d4-4d30-841b-1c355affb0dd */ select * from web_sales", True),
    ("  \n\t SELECT a FROM t", True),
    ("-- a note\nSELECT a FROM t", True),
    ("/* one */ /* two */ SELECT a FROM t", True),
    ("/* spans\nlines */ SELECT a FROM t", True),
    # Comments *after* the keyword. The first version of this list had none,
    # which is why it passed while the pattern was wrong: with `/*.*?*/` the
    # strip ran to the LAST `*/` in the statement and took the query with it.
    # Every one of these was being dropped -- ordinary reads, discarded by the
    # filter whose whole purpose is to stop discarding them.
    ("/* a */ SELECT x FROM t /* b */", True),
    ("SELECT x FROM t /* b */", True),
    ("/* a */ SELECT '*/' FROM t", True),
    ("/* nlq-console:abc */ SELECT x FROM t /* end */", True),
    ("/* a */ SELECT x\n-- note\nFROM t", True),
    ("/** a **/ SELECT x FROM t", True),
    ("INSERT INTO t VALUES (1)", False),
    ("UPDATE t SET a = 1", False),
    ("DELETE FROM t", False),
    ("CREATE TABLE t (a int)", False),
    # Each streamed read leaves these three behind under a name SQLAlchemy
    # generates fresh every time, so they never normalise and they accumulate
    # without bound. On the deployment that prompted this, 3,750 of a
    # database's 3,782 entries were exactly these.
    ('FETCH FORWARD 1000 FROM "c_7f1449704a10_76"', False),
    ('CLOSE "c_7f1449704a10_76"', False),
    ("DECLARE c CURSOR FOR SELECT 1", False),
    ("/* a comment */ INSERT INTO t VALUES (1)", False),
    # A word that merely begins with the keyword is not the keyword.
    ("selectivity_check()", False),
    ("withdraw_funds()", False),
]


def test_the_history_filter_keeps_reads_and_drops_the_rest(connector) -> None:
    """The predicate that ships, run by Postgres over known statements.

    Asserted against the real regex engine rather than the text of the query
    around it. The defect being fixed was a Postgres-side anchoring detail:
    `query ILIKE 'select%'` reads as "is this a SELECT" and actually means "does
    the character at position one begin one", which is false for every statement
    carrying a leading comment. No amount of reading the Python would surface
    that; only Postgres can answer it.
    """
    from nlqueries.connectors.postgres import STATEMENT_IS_A_READ

    engine = connector._require_engine()
    with engine.connect() as conn:
        for statement, expected in _HISTORY_STATEMENTS:
            got = conn.execute(
                sa_text(
                    f"SELECT {STATEMENT_IS_A_READ} AS matched "  # noqa: S608 — module constant
                    "FROM (SELECT CAST(:q AS text) AS query) s"
                ),
                {"q": statement},
            ).scalar()
            assert got is expected, f"{statement!r}: expected {expected}, got {got}"


def test_history_extraction_is_scoped_to_the_connected_database(seeded_connector) -> None:
    """pg_stat_statements is cluster-wide; one connector must not read another's SQL.

    Without the dbid filter this returned every database's statements on the
    same server — 6,076 of 9,880 entries on the deployment that prompted it,
    which is both noise competing for a bounded budget and somebody else's
    queries. The extension is absent from the throwaway container, so what is
    checked here is that the statement the connector builds really is scoped;
    the empty-result path is covered by the warning test above.
    """
    engine = seeded_connector._require_engine()
    with engine.connect() as conn:
        # The same subselect the extraction uses, against this container.
        current = conn.execute(
            sa_text("SELECT (SELECT oid FROM pg_database WHERE datname = current_database())")
        ).scalar()
        others = conn.execute(
            sa_text("SELECT count(*) FROM pg_database WHERE oid <> :oid"), {"oid": current}
        ).scalar()

    assert current is not None
    # A cluster always has other databases (template0, template1, postgres), so
    # the filter is doing real work rather than being a no-op here.
    assert others and others > 0


def test_both_history_queries_carry_both_filters() -> None:
    """The Postgres<13 fallback is a second copy of the statement.

    It was added for a renamed column and is easy to forget when the WHERE
    clause changes — so a filter added to one and not the other would leave the
    old behaviour reachable on exactly the deployments least likely to notice.
    """
    import inspect

    from nlqueries.connectors.postgres import PostgresConnector as _PC

    source = inspect.getsource(_PC.extract_query_history)
    body = source.split('"""', 2)[-1]  # assertions belong to the code, not the docstring
    assert body.count("mean_exec_time\n") >= 1
    assert body.count("dbid = (") == 2
    assert body.count("{STATEMENT_IS_A_READ}") == 2
