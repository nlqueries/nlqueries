"""
The payload corpus, as data.

Rebuilt from the manifest the Phase 3 report left behind — `pg_runtime.py`,
`pg_extra.py`, `pg_priv_file.py`, `duckdb_runtime.py` and the rest — after the
scripts themselves were deleted. Keeping them as data rather than as scripts
means every route (fresh generation, cache replay, retry, promotion, the direct
connector API) can be driven from one list, and adding a payload covers all of
them at once.

Every payload is a syntactically valid `SELECT`. Each therefore satisfies the
only condition the current gates test, that the root node is a Select, while
performing an operation a read query should not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Payload:
    """One statement, and what it proves if it runs."""

    id: str
    sql: str
    dialect: str
    #: What an operator would see if this executed.
    effect: str
    #: The register row this belongs to.
    finding: str
    #: What is expected to stop it once the work is done. More than one layer is
    #: the point — a payload with a single line of defence is a payload waiting
    #: for that line to have a bug in it.
    stopped_by: tuple[str, ...]


#: PostgreSQL, from the audit's own reproduction.
POSTGRES: tuple[Payload, ...] = (
    Payload(
        id="volatile_function_write",
        sql="SELECT lab.mark('corpus')",
        dialect="postgres",
        effect="a row committed to lab.marker through a SELECT",
        finding="SEC-01/SEC-02",
        stopped_by=("read-only transaction", "function EXECUTE revoked", "SQL policy"),
    ),
    Payload(
        id="sequence_advance",
        sql="SELECT nextval('lab.counter')",
        dialect="postgres",
        effect="sequence state advanced on disk",
        finding="SEC-01/SEC-02",
        stopped_by=("read-only transaction", "sequence privileges", "SQL policy"),
    ),
    Payload(
        id="advisory_lock",
        sql="SELECT pg_advisory_lock(4242)",
        dialect="postgres",
        effect="a session lock other queries wait on",
        finding="SEC-02",
        # Deliberately not "read-only transaction": it does not stop this, which
        # is exactly why the SQL policy and the role both have to.
        stopped_by=("SQL policy",),
    ),
    Payload(
        id="sleep",
        sql="SELECT pg_sleep(30)",
        dialect="postgres",
        effect="a connection held for 30 seconds",
        finding="SEC-02",
        stopped_by=("SQL policy", "statement_timeout"),
    ),
    Payload(
        id="server_file_read",
        sql="SELECT pg_read_file('/etc/hostname')",
        dialect="postgres",
        effect="a file off the database host in a query result",
        finding="SEC-02",
        stopped_by=("SQL policy", "pg_read_server_files not granted"),
    ),
    Payload(
        id="dml_in_cte",
        sql=(
            "WITH w AS (INSERT INTO lab.marker (note) VALUES ('cte') RETURNING note) "
            "SELECT * FROM w"
        ),
        dialect="postgres",
        effect="a write hidden inside a statement whose root node is a Select",
        finding="SEC-02",
        stopped_by=("read-only transaction", "SQL policy"),
    ),
    Payload(
        id="select_into",
        sql="SELECT * INTO lab.copied FROM lab.orders",
        dialect="postgres",
        effect="a new table created by a SELECT",
        finding="SEC-02",
        stopped_by=("read-only transaction", "SQL policy"),
    ),
    Payload(
        id="row_lock",
        sql="SELECT * FROM lab.orders FOR UPDATE",
        dialect="postgres",
        effect="rows locked against other writers",
        finding="SEC-02",
        # Measured, not assumed: a read-only transaction refuses this outright,
        # because taking a row lock needs a transaction id it will not assign.
        stopped_by=("read-only transaction", "SQL policy"),
    ),
    Payload(
        id="second_statement_after_comment",
        sql="SELECT 1; -- innocent\nDROP TABLE lab.orders",
        dialect="postgres",
        effect="a second statement smuggled past a single-statement check",
        finding="SEC-02",
        stopped_by=("read-only transaction", "SQL policy"),
    ),
)


#: DuckDB reads the local filesystem through ordinary-looking table functions,
#: so the database file is not the boundary.
DUCKDB: tuple[Payload, ...] = (
    Payload(
        id="duckdb_read_csv",
        sql="SELECT * FROM read_csv_auto('/etc/hostname')",
        dialect="duckdb",
        effect="a file outside the database read into a result set",
        finding="SEC-16",
        stopped_by=("SQL policy", "external access disabled", "process sandbox"),
    ),
    Payload(
        id="duckdb_glob",
        sql="SELECT * FROM glob('/**')",
        dialect="duckdb",
        effect="the host filesystem enumerated",
        finding="SEC-16",
        stopped_by=("SQL policy", "external access disabled", "process sandbox"),
    ),
)


ALL: tuple[Payload, ...] = POSTGRES + DUCKDB


#: Queries that must continue to work. Without them the corpus measures only
#: how much a policy refuses, not whether it remains usable. The failure mode of
#: every control here is the refusal of legitimate analytics.
SAFE_POSTGRES: tuple[str, ...] = (
    "SELECT count(*) FROM lab.orders",
    "SELECT customer_id, sum(total) AS revenue FROM lab.orders GROUP BY customer_id",
    "SELECT id, total, row_number() OVER (ORDER BY total DESC) AS rank FROM lab.orders",
    "SELECT date_trunc('month', now()) AS month",
    "SELECT coalesce(max(total), 0)::numeric AS top FROM lab.orders",
    "WITH totals AS (SELECT customer_id, sum(total) t FROM lab.orders GROUP BY 1) "
    "SELECT * FROM totals WHERE t > 10",
)
