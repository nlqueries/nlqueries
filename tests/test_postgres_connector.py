"""
Tests for PostgresConnector (nlqueries.connectors.postgres).

These are integration tests backed by a real, ephemeral PostgreSQL instance
spun up in Docker via testcontainers. They are skipped automatically when
Docker is not available (e.g. in CI environments without a Docker daemon).

SSL unit tests live in test_postgres_ssl.py — they use unittest.mock and
require no live database.
"""

from __future__ import annotations

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


def test_a_select_only_login_sees_the_keys(seeded_connector, credentials):
    """The login database-hardening.md recommends holds SELECT and nothing else.
    information_schema hides constraints from such a login, so it used to read
    no key at all. Keys come from pg_catalog now: primary keys, foreign keys,
    a composite key paired column to column, and nothing of a table it may not
    read."""
    statements = [
        "DROP TABLE IF EXISTS event_notes, events, secret_links, shipment_lines, shipments, "
        "secrets CASCADE",
        "DROP ROLE IF EXISTS keys_reader",
        "CREATE ROLE keys_reader LOGIN PASSWORD 'reader-pw'",
        "CREATE TABLE shipments (order_id INTEGER, seq INTEGER, PRIMARY KEY (order_id, seq))",
        """
        CREATE TABLE shipment_lines (
            id SERIAL PRIMARY KEY,
            ship_order INTEGER,
            ship_seq INTEGER,
            FOREIGN KEY (ship_order, ship_seq) REFERENCES shipments (order_id, seq)
        )
        """,
        "CREATE TABLE secrets (id SERIAL PRIMARY KEY, value TEXT)",
        # A readable table with a key into the unreadable one.
        "CREATE TABLE secret_links "
        "(id SERIAL PRIMARY KEY, secret_id INTEGER REFERENCES secrets(id))",
        # A key into a partitioned table, which PostgreSQL copies per partition.
        """
        CREATE TABLE events (id INTEGER, day DATE, PRIMARY KEY (id, day))
        PARTITION BY RANGE (day)
        """,
        "CREATE TABLE events_2025 PARTITION OF events "
        "FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')",
        "CREATE TABLE events_2026 PARTITION OF events "
        "FOR VALUES FROM ('2026-01-01') TO ('2027-01-01')",
        """
        CREATE TABLE event_notes (
            id SERIAL PRIMARY KEY,
            event_id INTEGER,
            event_day DATE,
            FOREIGN KEY (event_id, event_day) REFERENCES events (id, day)
        )
        """,
        "GRANT USAGE ON SCHEMA public TO keys_reader",
        "GRANT SELECT ON customers, orders, order_items, shipments, shipment_lines, "
        "secret_links, events, events_2025, events_2026, event_notes TO keys_reader",
    ]
    with seeded_connector._require_engine().begin() as conn:
        for stmt in statements:
            conn.execute(sa_text(stmt))

    reader = granted(PostgresConnector())
    reader.connect({**credentials, "user": "keys_reader", "password": "reader-pw"})
    tables = {t.name: {c.name: c for c in t.columns} for t in reader.extract_schema().tables}

    assert tables["customers"]["id"].is_primary_key is True
    assert tables["orders"]["customer_id"].references == "customers.id"
    assert {n for n, c in tables["shipments"].items() if c.is_primary_key} == {"order_id", "seq"}
    assert tables["shipment_lines"]["ship_order"].references == "shipments.order_id"
    assert tables["shipment_lines"]["ship_seq"].references == "shipments.seq"
    assert "secrets" not in tables
    # From #357's review: no join is offered to a table the login may not read,
    # and a key into a partitioned table names the table, not a partition.
    assert tables["secret_links"]["secret_id"].is_foreign_key is False
    assert tables["event_notes"]["event_id"].references == "events.id"
    assert tables["event_notes"]["event_day"].references == "events.day"
    # pg_constraint is readable by every login; the key queries keep to tables
    # the login may use, so they name nothing of the one it may not.
    with reader._require_engine().connect() as conn:
        assert ("public", "secrets") not in PostgresConnector._fetch_primary_keys(conn)


def test_a_column_level_grant_names_no_key_on_a_column_it_hides(seeded_connector, credentials):
    """Granted only some columns of a table, the login's schema leaves the rest
    out, column by column. The keys did not: a per-table test reported
    `tickets.person_id` as referencing `people.id`, though `id` is not in its
    schema, and a join on it fails with a permission error (from #357's review).
    Granted the column, the same keys appear, as the control."""
    statements = [
        "DROP TABLE IF EXISTS tickets, people CASCADE",
        "DROP ROLE IF EXISTS cols_reader",
        "CREATE ROLE cols_reader LOGIN PASSWORD 'reader-pw'",
        "CREATE TABLE people (id SERIAL PRIMARY KEY, name TEXT, email TEXT)",
        "CREATE TABLE tickets (id SERIAL PRIMARY KEY, person_id INTEGER REFERENCES people(id))",
        "GRANT USAGE ON SCHEMA public TO cols_reader",
        "GRANT SELECT (name, email) ON people TO cols_reader",
        "GRANT SELECT ON tickets TO cols_reader",
    ]
    with seeded_connector._require_engine().begin() as conn:
        for stmt in statements:
            conn.execute(sa_text(stmt))

    reader = granted(PostgresConnector())
    reader.connect({**credentials, "user": "cols_reader", "password": "reader-pw"})
    tables = {t.name: {c.name: c for c in t.columns} for t in reader.extract_schema().tables}

    assert set(tables["people"]) == {"name", "email"}  # the column list, for reference
    with reader._require_engine().connect() as conn:
        assert ("public", "people") not in PostgresConnector._fetch_primary_keys(conn)
    assert tables["tickets"]["person_id"].is_foreign_key is False
    assert tables["tickets"]["id"].is_primary_key is True

    with seeded_connector._require_engine().begin() as conn:
        conn.execute(sa_text("GRANT SELECT (id) ON people TO cols_reader"))
    reader.close()
    reader.connect({**credentials, "user": "cols_reader", "password": "reader-pw"})
    tables = {t.name: {c.name: c for c in t.columns} for t in reader.extract_schema().tables}
    assert tables["people"]["id"].is_primary_key is True
    assert tables["tickets"]["person_id"].references == "people.id"


def test_a_composite_key_is_offered_whole_or_not_at_all(seeded_connector, credentials):
    """Filtered column by column, a composite key lost only the columns the login
    cannot see: granted `shipments.order_id` but not `seq`, the login was offered
    `ship_order -> shipments.order_id` alone, a join that runs and multiplies rows,
    and `order_id` as the whole primary key of `shipments` (from #360's review).
    Granted `seq` too, both keys appear whole, as the control."""
    statements = [
        "DROP TABLE IF EXISTS shipment_lines, shipments CASCADE",
        "DROP ROLE IF EXISTS part_reader",
        "CREATE ROLE part_reader LOGIN PASSWORD 'reader-pw'",
        "CREATE TABLE shipments (order_id INTEGER, seq INTEGER, carrier TEXT,"
        " PRIMARY KEY (order_id, seq))",
        "CREATE TABLE shipment_lines (id SERIAL PRIMARY KEY, ship_order INTEGER,"
        " ship_seq INTEGER, FOREIGN KEY (ship_order, ship_seq)"
        " REFERENCES shipments (order_id, seq))",
        "GRANT USAGE ON SCHEMA public TO part_reader",
        "GRANT SELECT (order_id, carrier) ON shipments TO part_reader",
        "GRANT SELECT ON shipment_lines TO part_reader",
    ]
    with seeded_connector._require_engine().begin() as conn:
        for stmt in statements:
            conn.execute(sa_text(stmt))

    reader = granted(PostgresConnector())
    reader.connect({**credentials, "user": "part_reader", "password": "reader-pw"})
    tables = {t.name: {c.name: c for c in t.columns} for t in reader.extract_schema().tables}

    assert set(tables["shipments"]) == {"order_id", "carrier"}  # the column list
    assert tables["shipments"]["order_id"].is_primary_key is False
    assert tables["shipment_lines"]["ship_order"].is_foreign_key is False
    assert tables["shipment_lines"]["ship_seq"].is_foreign_key is False

    with seeded_connector._require_engine().begin() as conn:
        conn.execute(sa_text("GRANT SELECT (seq) ON shipments TO part_reader"))
    reader.close()
    reader.connect({**credentials, "user": "part_reader", "password": "reader-pw"})
    tables = {t.name: {c.name: c for c in t.columns} for t in reader.extract_schema().tables}
    assert tables["shipments"]["order_id"].is_primary_key is True
    assert tables["shipments"]["seq"].is_primary_key is True
    assert tables["shipment_lines"]["ship_order"].references == "shipments.order_id"
    assert tables["shipment_lines"]["ship_seq"].references == "shipments.seq"


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


def test_extract_query_history_says_the_extension_is_missing(connector):
    # The default postgres:16-alpine image does not ship pg_stat_statements,
    # so this exercises the "extension missing" path: a reason, not an empty
    # list that reads as "no history".
    from nlqueries.connectors.base import QueryHistoryUnavailable

    with pytest.raises(QueryHistoryUnavailable) as err:
        connector.extract_query_history(days=30)

    assert "pg_stat_statements" in str(err.value)
    assert "CREATE EXTENSION" in str(err.value)


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
    # A streamed read is recorded only as the cursor it was declared as. Our own
    # tagged runs are kept, and read as their statement (see STATEMENT_TEXT);
    # every other cursor, the agent's own streamed reads included, stays out.
    (
        'DECLARE "c_7f1449704a10_76" CURSOR WITHOUT HOLD FOR '
        "/* nlq-console:2a2e3031-f2d4-4d30-841b-1c355affb0dd */ SELECT a FROM t",
        True,
    ),
    (
        'DECLARE "c_7f1449704a10_77" CURSOR WITHOUT HOLD FOR '
        "/* nlq-pinboard:12 */ WITH x AS (SELECT 1) SELECT * FROM x",
        True,
    ),
    ('declare "c_1" cursor without hold for /* nlq-scheduled-chart:9 */ select 1', True),
    ('DECLARE "c_7f1449704a10_78" CURSOR WITHOUT HOLD FOR SELECT a FROM t', False),
    ('DECLARE "c_1" CURSOR WITHOUT HOLD FOR /* another tool */ SELECT a FROM t', False),
    ('DECLARE "c_1" CURSOR WITHOUT HOLD FOR /* nlq-console:abc */ DELETE FROM t', False),
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


def test_the_history_reads_a_tagged_cursor_as_its_statement(connector) -> None:
    """What extraction returns for a row: a tagged cursor's statement, and
    every other row as it was recorded."""
    from nlqueries.connectors.postgres import STATEMENT_TEXT

    tagged = "/* nlq-console:abc */ SELECT a FROM t"
    cases = {
        f'DECLARE "c_7f1449704a10_76" CURSOR WITHOUT HOLD FOR {tagged}': tagged,
        f'DECLARE "c_1" NO SCROLL CURSOR WITH HOLD FOR {tagged}': tagged,
        tagged: tagged,
        'DECLARE "c_1" CURSOR WITHOUT HOLD FOR SELECT a FROM t': (
            'DECLARE "c_1" CURSOR WITHOUT HOLD FOR SELECT a FROM t'
        ),
    }
    engine = connector._require_engine()
    with engine.connect() as conn:
        for recorded, expected in cases.items():
            got = conn.execute(
                sa_text(
                    f"SELECT {STATEMENT_TEXT} AS text "  # noqa: S608 — module constant
                    "FROM (SELECT CAST(:q AS text) AS query) s"
                ),
                {"q": recorded},
            ).scalar()
            assert got == expected, recorded


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


def _history_statement_for(server_version: tuple[int, ...]) -> str:
    """The history statement the connector sends to a server of *server_version*."""
    from unittest.mock import MagicMock

    sent: list[str] = []

    def execute(statement, *_args):
        sent.append(str(statement))
        result = MagicMock()
        result.scalar_one_or_none.return_value = 1  # the extension is installed
        result.mappings.return_value = []
        return result

    conn = MagicMock()
    conn.dialect.server_version_info = server_version
    conn.execute.side_effect = execute
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    connector = PostgresConnector()
    connector._engine = engine

    assert connector.extract_query_history() == []
    [statement] = [s for s in sent if "FROM pg_stat_statements" in s]
    return statement


def test_the_history_statement_names_the_column_the_server_has() -> None:
    """Postgres < 13 called ``mean_exec_time`` ``mean_time``. The column is chosen
    from the server's version: the old way tried one statement and, on any
    error, ran the other in the transaction the first had aborted, so every
    failure read as "current transaction is aborted"."""
    old = _history_statement_for((12, 18))
    new = _history_statement_for((16, 4))

    assert "calls, mean_time AS mean_exec_time" in old
    assert "calls, mean_exec_time AS mean_exec_time" in new
    for statement in (old, new):  # one statement, so one set of filters
        assert "dbid = (" in statement
        assert "~* '^(select|with)[^a-z]'" in statement


# ---------------------------------------------------------------------------
# Streaming a tagged read, and naming the session that runs it
# ---------------------------------------------------------------------------


def test_a_read_behind_a_leading_comment_runs_on_a_server_side_cursor(connector) -> None:
    """Streamed means a named cursor: psycopg2 leaves a client-side one unnamed
    and loads its whole result before the budget sees a row. The SQL console
    tags every statement with a leading comment, and a check on the first word
    of the raw text streamed none of them."""
    from sqlalchemy import event

    engine = connector._require_engine()
    cursors: list[str | None] = []

    def record(_conn, cursor, statement, *_args) -> None:
        if "nlq-console:abc" in statement:
            cursors.append(cursor.name)

    event.listen(engine, "before_cursor_execute", record)
    try:
        result = connector.execute_query(
            "/* nlq-console:abc */ SELECT g FROM generate_series(1, 3) AS g"
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert result.error is None
    assert result.rows == [[1], [2], [3]]
    [name] = cursors
    assert name is not None


def test_a_streamed_read_is_bounded_by_its_timeout_as_a_whole(connector) -> None:
    """statement_timeout bounds each statement, and a streamed read is a
    DECLARE and then a FETCH per batch, so a timeout set once let every batch
    run the whole budget again: on Postgres 16 a 2 s timeout let a 6 s read
    finish. Here every 1,000th row pauses 0.6 s, so each batch stays under the
    1.5 s timeout and the ten together take four times it.

    The cancellation is what tells the two apart; the bound only has to sit
    well under the 6 s an unbounded read takes. It was 2.5 s, one second over
    the timeout, which a loaded machine's connection and teardown overran."""
    import time

    start = time.perf_counter()
    result = connector.execute_query(
        "SELECT pg_sleep(CASE WHEN g % 1000 = 0 THEN 0.6 ELSE 0 END), g "
        "FROM generate_series(1, 10000) AS g",
        timeout_seconds=1.5,
    )
    took = time.perf_counter() - start

    assert "canceling statement due to statement timeout" in (result.error or ""), result
    assert took < 4.5


def test_a_streamed_reads_time_includes_reading_its_rows(connector) -> None:
    """It was taken when execute returned, which for a streamed read is after
    the DECLARE and the first row: a read that took a second reported a fifth
    of one."""
    result = connector.execute_query("SELECT pg_sleep(0.2), g FROM generate_series(1, 5) AS g")

    assert result.error is None
    assert result.execution_time_ms >= 900


def test_the_session_carries_the_application_name_it_was_given(credentials) -> None:
    labelled = granted(PostgresConnector())
    labelled.connect({**credentials, "application_name": "nlq-console:abc"})
    plain = granted(PostgresConnector())
    plain.connect(credentials)

    def name(c: PostgresConnector) -> object:
        return c.execute_query("SELECT current_setting('application_name')").rows

    assert name(labelled) == [["nlq-console:abc"]]
    assert name(plain) != [["nlq-console:abc"]]


def test_a_streamed_read_is_found_by_its_application_name_while_it_runs(credentials) -> None:
    """What the label is for. While a streamed read runs, its session's text
    in pg_stat_activity is the cursor's FETCH, which carries no tag, so a
    cancel that matched the text found nothing and the read ran on. Matched by
    application_name, it is found and stopped."""
    import threading
    import time

    tag = "nlq-console:find-me"
    reader = granted(PostgresConnector())
    reader.connect({**credentials, "application_name": tag})
    outcome: dict[str, QueryResult] = {}
    worker = threading.Thread(
        target=lambda: outcome.update(
            result=reader.execute_query(
                f"/* {tag} */ SELECT pg_sleep(0.2), g FROM generate_series(1, 100) AS g"
            )
        )
    )
    worker.start()

    watcher = granted(PostgresConnector())
    watcher.connect(credentials)
    texts: list[str] = []
    cancelled: list[bool] = []
    with watcher._require_engine().connect() as conn:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not any(q.startswith("FETCH") for q in texts):
            texts = list(
                conn.execute(
                    sa_text(
                        "SELECT query FROM pg_stat_activity "
                        "WHERE application_name = :tag AND state = 'active'"
                    ),
                    {"tag": tag},
                ).scalars()
            )
            # pg_stat_activity is a snapshot per transaction: end it, or every
            # poll sees the first one.
            conn.rollback()
            time.sleep(0.05)
        by_text = conn.execute(
            sa_text(
                "SELECT pid FROM pg_stat_activity "
                "WHERE query LIKE :pattern AND pid <> pg_backend_pid()"
            ),
            {"pattern": f"%{tag}%"},
        ).all()
        cancelled = list(
            conn.execute(
                sa_text(
                    "SELECT pg_cancel_backend(pid) FROM pg_stat_activity "
                    "WHERE application_name = :tag AND pid <> pg_backend_pid()"
                ),
                {"tag": tag},
            ).scalars()
        )
    worker.join(timeout=30)

    assert any(q.startswith("FETCH") for q in texts), texts
    assert by_text == []  # the old way finds nothing
    assert cancelled == [True]
    assert "canceling statement due to user request" in (outcome["result"].error or "")


@pytest.fixture(scope="module")
def history_credentials() -> dict:
    """A second container, with pg_stat_statements loaded, which the first lacks."""
    container = PostgresContainer("postgres:16-alpine").with_command(
        "postgres -c shared_preload_libraries=pg_stat_statements"
    )
    try:
        container.start()
    except Exception as exc:
        pytest.skip(f"Could not start Postgres container: {exc}")
    try:
        yield {
            "host": container.get_container_host_ip(),
            "port": int(container.get_exposed_port(5432)),
            "database": container.dbname,
            "user": container.username,
            "password": container.password,
            "ssl_mode": "disable",
        }
    finally:
        container.stop()


def test_a_streamed_read_of_ours_is_extracted_as_its_statement(history_credentials) -> None:
    """End to end. Once a tagged read streams, pg_stat_statements records only
    its DECLARE, which the history drops with every other cursor, so the
    statements run deliberately in the SQL console stopped reaching the
    history. Extraction returns the statement, as before it streamed. An
    untagged streamed read stays out."""
    connector = granted(PostgresConnector())
    connector.connect(history_credentials)
    with connector._require_engine().begin() as conn:
        conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
        conn.execute(sa_text("CREATE TABLE hist_a (id int, b_id int)"))
        conn.execute(sa_text("CREATE TABLE hist_b (id int)"))
        conn.execute(sa_text("SELECT pg_stat_statements_reset()"))
    tagged = (
        "/* nlq-console:e2e */ SELECT hist_a.id FROM hist_a JOIN hist_b ON hist_a.b_id = hist_b.id"
    )
    untagged = (
        "SELECT hist_b.id FROM hist_b JOIN hist_a ON hist_a.b_id = hist_b.id WHERE hist_b.id > 7"
    )
    assert connector.execute_query(tagged).error is None
    assert connector.execute_query(untagged).error is None

    extracted = [record.sql for record in connector.extract_query_history()]

    assert tagged in extracted
    assert not any(s.lstrip().upper().startswith(("DECLARE", "FETCH", "CLOSE")) for s in extracted)
    assert not any("hist_b.id >" in s for s in extracted)


def test_a_login_that_may_not_read_the_history_is_told_why(history_credentials) -> None:
    """A REVOKE EXECUTE ... FROM PUBLIC on the extension's schema takes
    ``pg_stat_statements()`` away from every login. The connector says so, as
    QueryHistoryUnavailable, so a caller can carry on without history; it used
    to raise "current transaction is aborted", which named neither."""
    from nlqueries.connectors.base import QueryHistoryUnavailable

    admin = granted(PostgresConnector())
    admin.connect(history_credentials)
    with admin._require_engine().begin() as conn:
        conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
        conn.execute(sa_text("DROP ROLE IF EXISTS history_reader"))
        conn.execute(sa_text("CREATE ROLE history_reader LOGIN PASSWORD 'reader-pw'"))
        conn.execute(sa_text("REVOKE EXECUTE ON FUNCTION pg_stat_statements(boolean) FROM PUBLIC"))
    reader = PostgresConnector()
    reader.connect({**history_credentials, "user": "history_reader", "password": "reader-pw"})
    try:
        with pytest.raises(QueryHistoryUnavailable) as err:
            reader.extract_query_history()
        message = str(err.value)
        assert "permission denied for function pg_stat_statements" in message
        assert "GRANT EXECUTE ON FUNCTION pg_stat_statements(boolean)" in message
        assert "transaction is aborted" not in message

        # The control: granted the function, the same login reads the history.
        with admin._require_engine().begin() as conn:
            conn.execute(
                sa_text("GRANT EXECUTE ON FUNCTION pg_stat_statements(boolean) TO history_reader")
            )
        assert isinstance(reader.extract_query_history(), list)
    finally:
        reader.close()
        with admin._require_engine().begin() as conn:
            conn.execute(sa_text("GRANT EXECUTE ON FUNCTION pg_stat_statements(boolean) TO PUBLIC"))
            conn.execute(sa_text("DROP OWNED BY history_reader"))
            conn.execute(sa_text("DROP ROLE history_reader"))
        admin.close()


def test_resetting_the_timeout_per_batch_leaves_one_history_row(history_credentials) -> None:
    """A streamed read resets its timeout before every batch. As a SET, each
    value would be a pg_stat_statements row of its own (on PostgreSQL 15 and
    16 alike), crowding the database's own statements out of a fixed-size
    view; set_config is normalised to one."""
    connector = granted(PostgresConnector())
    connector.connect(history_credentials)
    engine = connector._require_engine()
    with engine.begin() as conn:
        conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
        conn.execute(sa_text("SELECT pg_stat_statements_reset()"))

    for _ in range(2):
        result = connector.execute_query(
            "SELECT g FROM generate_series(1, 3500) AS g", timeout_seconds=30
        )
        assert result.error is None
        assert result.row_count == 3500

    with engine.connect() as conn:
        recorded = conn.execute(
            sa_text(
                "SELECT query, calls FROM pg_stat_statements "
                "WHERE query ILIKE '%statement_timeout%' OR query ILIKE '%set_config%'"
            )
        ).all()
    sets = [query for query, _calls in recorded if query.lstrip().upper().startswith("SET")]
    resets = [calls for query, calls in recorded if "set_config" in query]
    assert sets == ["SET LOCAL statement_timeout = 30000"]  # the read's own, once
    assert resets and resets[0] >= 6  # at least three batches in each read
