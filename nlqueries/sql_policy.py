# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.sql_policy
~~~~~~~~~~~~~~~~~~~~
Whether a generated statement is one this product is willing to run.

Rules are applied to the whole parsed tree. The checks this replaced examined
only the root node, and every payload in ``tests/security/payloads`` has a
``Select`` at its root.

Permitted functions are determined by sqlglot rather than listed here. sqlglot
models standard SQL functions as typed classes and falls back to
:class:`exp.Anonymous` for the vendor, extension and user-defined set.
Measured: that signal refuses all five function-based payloads in the audit
corpus, and misclassifies 4 of 62 common analytics functions (``age``,
``jsonb_agg``, ``regexp_matches``, ``every``), each requiring one allowlist
entry.

Root statement types are allow-listed. ``VACUUM``, ``CALL`` and ``DO`` parse to
:class:`exp.Command`; ``COPY``, ``SET``, ``GRANT`` and ``TRUNCATE`` each parse
to a distinct class. A deny-list would require every such class to be named.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

#: Incremented when a decision made by this module could change. Recorded on
#: every decision so a stored verdict can be told apart from a current one.
POLICY_VERSION = "1"

#: The only statement types a generated query may be.
_ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.Union)

#: Node types that must not appear anywhere in the tree, including inside a CTE
#: or subquery, where a ``Select`` root would otherwise conceal them.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Set,
    exp.Grant,
    exp.Command,
    #: ``SELECT ... INTO`` creates a table from a query result.
    exp.Into,
    #: ``FOR UPDATE`` and its relatives take row locks. PostgreSQL refuses
    #: these inside a read-only transaction; other engines do not.
    exp.Lock,
)

#: Functions sqlglot does not model that are safe to run, per dialect. Each
#: entry records that the named function reads data and performs no other
#: action.
ALLOWED_ANONYMOUS: dict[str, frozenset[str]] = {
    "postgres": frozenset({"age", "jsonb_agg", "regexp_matches", "every", "date_part"}),
    "redshift": frozenset({"age", "regexp_matches", "every", "date_part"}),
    "snowflake": frozenset({"every"}),
    "bigquery": frozenset(),
    "mssql": frozenset(),
    "sqlite": frozenset(),
    "duckdb": frozenset({"every"}),
}

#: Names sqlglot knows under a different spelling. The keys are both connector
#: `db_type` values and SQLAlchemy backend names, since the two vocabularies
#: overlap and neither matches sqlglot's exactly.
#:
#: Measured against sqlglot 30.9 and SQLAlchemy 2.0: `postgresql`, `mssql` and
#: `mariadb` are rejected by sqlglot; `mysql`, `sqlite`, `oracle`, `snowflake`,
#: `bigquery`, `redshift`, `duckdb` and `clickhouse` are accepted unchanged.
_DIALECT_ALIASES = {
    "mssql": "tsql",
    "postgresql": "postgres",
    "mariadb": "mysql",
}


def _sqlglot_dialect(dialect: str) -> str:
    return _DIALECT_ALIASES.get(dialect.lower(), dialect.lower())


def dialect_from_url(url: str) -> str | None:
    """The sqlglot dialect for a SQLAlchemy *url*, or None if unreadable.

    The generic SQLAlchemy connector reaches any engine SQLAlchemy supports, so
    its ``db_type`` does not identify a grammar. The URL identifies the engine:
    ``mysql+pymysql://...`` is MySQL regardless of the connector's name.

    Returns None when the URL cannot be read, leaving the decision to the
    caller. A statement checked against a different grammar is not checked.
    """
    try:
        from sqlalchemy.engine import make_url  # noqa: PLC0415

        backend = make_url(url).get_backend_name()
    except Exception:  # noqa: BLE001 - an unreadable URL names no dialect
        return None
    return _sqlglot_dialect(backend) if backend else None


#: Statements longer than this are refused before parsing.
MAX_BYTES = 100_000

#: Bracket nesting deeper than this is refused, and refused *before* parsing.
#: Measured: sqlglot's parser is recursive and exhausts Python's stack on a
#: deeply nested expression, raising RecursionError -- which is not a
#: SqlglotError -- so a check performed on the parsed tree never runs.
MAX_DEPTH = 100


@dataclass(frozen=True)
class PolicyDecision:
    """Whether *sql* may run, and why not if it may not."""

    allowed: bool
    dialect: str
    policy_version: str
    #: One entry per rule the statement failed. Empty when allowed.
    reasons: tuple[str, ...] = ()
    #: Every unrecognised function seen, whether or not it was allowed.
    #: Recorded so the inventory can report what an allowlist would need.
    anonymous_functions: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.allowed

    def summary(self) -> str:
        if self.allowed:
            return "allowed"
        return "; ".join(self.reasons)


def _bracket_depth(sql: str) -> int:
    """The deepest bracket nesting in *sql*, counted from the text.

    Deliberately crude and performed before parsing. It over-counts brackets
    inside string literals, which costs nothing at a threshold of
    :data:`MAX_DEPTH`, and it runs where the parser cannot.
    """
    depth = best = 0
    for char in sql:
        if char in "([":
            depth += 1
            if depth > best:
                best = depth
        elif char in ")]":
            depth = max(0, depth - 1)
    return best


def evaluate(sql: str, dialect: str) -> PolicyDecision:
    """Decide whether *sql* may run against *dialect*.

    Never raises. A statement that cannot be parsed is refused, since a
    statement this module cannot read is one it cannot vouch for.
    """
    reasons: list[str] = []

    try:
        size = len(sql.encode("utf-8"))
    except UnicodeEncodeError:
        # A Python string can hold unpaired surrogates -- `json.loads('"\ud83d"')`
        # produces one -- and encoding raises rather than returning bytes. Such a
        # string is not valid text and is refused rather than allowed to raise.
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=("contains characters that are not valid text",),
        )

    if size > MAX_BYTES:
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=(f"statement exceeds {MAX_BYTES} bytes",),
        )

    if _bracket_depth(sql) > MAX_DEPTH:
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=(f"nested deeper than {MAX_DEPTH}",),
        )

    try:
        # `parse`, not `parse_one`: `parse_one` returns the first statement and
        # discards the rest, so a second statement hidden after a comment would
        # be invisible here and executed by the driver.
        statements = sqlglot.parse(sql, read=_sqlglot_dialect(dialect))
    except RecursionError:
        # The depth check above should have caught this. Kept because the
        # relationship between bracket depth and parser recursion is not exact,
        # and a crash here would be an outage rather than a refusal.
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=("exhausted the parser's stack",),
        )
    except ValueError as exc:
        # sqlglot raises a plain ValueError for a dialect it does not know,
        # which is not a SqlglotError. Refused rather than parsed with a
        # different grammar: a statement checked against the wrong dialect is
        # not checked.
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=(f"no grammar available for dialect '{dialect}': {exc}",),
        )
    except SqlglotError as exc:
        # SqlglotError, not ParseError. TokenError is a sibling of ParseError
        # rather than a subclass, and prose returned by a model in place of SQL
        # raises TokenError.
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=(f"could not be parsed as {dialect}: {type(exc).__name__}",),
        )

    parsed: list[exp.Expression] = [s for s in statements if isinstance(s, exp.Expression)]
    if len(parsed) != 1:
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            reasons=(f"expected exactly one statement, found {len(parsed)}",),
        )

    tree = parsed[0]

    root_name = type(tree).__name__
    if not isinstance(tree, _ALLOWED_ROOTS):
        reasons.append(f"statement is a {root_name}, not a query")

    # The root is excluded: reporting `DROP TABLE t` as both "is a Drop" and
    # "contains Drop" states one fact twice.
    forbidden = sorted({type(n).__name__ for n in tree.find_all(*_FORBIDDEN_NODES)} - {root_name})
    if forbidden:
        reasons.append(f"contains {', '.join(forbidden)}")

    anonymous = tuple(sorted({n.name.lower() for n in tree.find_all(exp.Anonymous) if n.name}))
    permitted = ALLOWED_ANONYMOUS.get(dialect.lower(), frozenset())
    refused = [name for name in anonymous if name not in permitted]
    if refused:
        reasons.append(f"calls unrecognised function(s): {', '.join(refused)}")

    return PolicyDecision(
        allowed=not reasons,
        dialect=dialect,
        policy_version=POLICY_VERSION,
        reasons=tuple(reasons),
        anonymous_functions=anonymous,
    )
