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

import hashlib
from dataclasses import dataclass

import sqlglot
from sqlglot import TokenType, exp
from sqlglot.errors import SqlglotError

#: Incremented when a decision made by this module could change. Recorded on
#: every decision so a stored verdict can be told apart from a current one.
#:
#: 2: the anonymous-function allowlist is keyed by sqlglot's dialect names and
#: looked up through the aliases below, so ``postgresql`` now gets Postgres's
#: entry, where it used to miss the table and be allowed nothing.
#:
#: 3: the file-reading functions sqlglot models, and a string literal used as a
#: table name, are refused.
POLICY_VERSION = "3"

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

#: Node types that read files, refused wherever they appear. The functions
#: sqlglot does not model (``read_json``, ``read_text``, ``glob`` and the rest)
#: are already refused as unrecognised; these are the ones it does model, which
#: that rule therefore never sees. Looked up by name because the declared floor
#: for sqlglot predates some of them.
_FILE_READING_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        # DuckDB's read_csv and read_parquet.
        "ReadCSV",
        "ReadParquet",
        # Snowflake's DIRECTORY(@stage) and TO_FILE(@stage, path).
        "DirectoryStage",
        "ToFile",
        # LOAD DATA [LOCAL] INPATH, which is never a query root in any case.
        "LoadData",
    )
    if hasattr(exp, name)
)

#: Functions sqlglot does not model that are safe to run, per dialect. Each
#: entry records that the named function reads data and performs no other
#: action.
#:
#: Keyed by sqlglot's dialect names, and read through :func:`allowed_anonymous`,
#: which resolves the other spellings first. Keyed by the callers' spellings it
#: drifted: the CLI hands SQL Server down as ``tsql``, the name sqlglot parses,
#: and an entry under ``mssql`` was out of its reach -- as it already was for
#: an enterprise SQLAlchemy connector, whose dialect comes from its URL.
ALLOWED_ANONYMOUS: dict[str, frozenset[str]] = {
    "postgres": frozenset({"age", "jsonb_agg", "regexp_matches", "every", "date_part"}),
    "redshift": frozenset({"age", "regexp_matches", "every", "date_part"}),
    "snowflake": frozenset({"every"}),
    "bigquery": frozenset(),
    "tsql": frozenset(),
    "mysql": frozenset(),
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


def allowed_anonymous(dialect: str) -> frozenset[str]:
    """The unmodelled functions *dialect* may call, whichever spelling names it.

    ``mssql`` and ``tsql`` share an entry, as do ``postgresql`` and
    ``postgres``. A dialect with no entry may call none.
    """
    return ALLOWED_ANONYMOUS.get(_sqlglot_dialect(dialect), frozenset())


def digest_of(sql: str) -> str:
    """SHA-256 of the UTF-8 bytes of *sql*.

    Uses ``surrogatepass`` so a string containing an unpaired surrogate can be
    digested rather than raising. Such a statement is refused by
    :func:`evaluate` before reaching a driver.
    """
    return hashlib.sha256(sql.encode("utf-8", errors="surrogatepass")).hexdigest()


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
    """Whether a statement may run, and why not if it may not.

    Bound to the statement it was made about. ``sql_digest`` is the SHA-256 of
    the exact bytes evaluated, so :meth:`authorises` can establish that a
    decision applies to the statement a caller is about to run rather than to
    some earlier one.
    """

    allowed: bool
    dialect: str
    policy_version: str
    #: SHA-256 of the UTF-8 bytes evaluated.
    sql_digest: str
    #: One entry per rule the statement failed. Empty when allowed.
    reasons: tuple[str, ...] = ()
    #: Every unrecognised function seen, whether or not it was allowed.
    #: Recorded so the inventory can report what an allowlist would need.
    anonymous_functions: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.allowed

    def authorises(self, sql: str, dialect: str) -> bool:
        """Whether this decision permits *sql* to run against *dialect*.

        False when the decision refused, when it was made about different bytes,
        or when it was made for a different dialect. Whitespace and comments are
        significant: the digest covers what will be sent to the driver.
        """
        return (
            self.allowed
            and self.sql_digest == digest_of(sql)
            and self.dialect.lower() == dialect.lower()
        )

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


def _string_table_names(tree: exp.Expression, sql: str, dialect: str) -> list[str]:
    """Table names in *tree* that were written as string literals.

    DuckDB reads ``FROM 'orders.csv'`` as a scan of that file, and sqlglot
    parses the string into a quoted identifier indistinguishable from
    ``FROM "orders.csv"`` -- a real table name. What tells them apart is the
    token the identifier came from, so the statement is tokenized and any table
    name starting where a string token starts is reported.
    """
    try:
        tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    except (SqlglotError, ValueError):
        return []
    string_starts = {token.start for token in tokens if token.token_type == TokenType.STRING}
    if not string_starts:
        return []
    found: list[str] = []
    for table in tree.find_all(exp.Table):
        for part in table.parts:
            if isinstance(part, exp.Identifier) and part.meta.get("start") in string_starts:
                found.append(part.name)
    return found


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
            sql_digest=digest_of(sql),
            reasons=("contains characters that are not valid text",),
        )

    if size > MAX_BYTES:
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            sql_digest=digest_of(sql),
            reasons=(f"statement exceeds {MAX_BYTES} bytes",),
        )

    if _bracket_depth(sql) > MAX_DEPTH:
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            sql_digest=digest_of(sql),
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
            sql_digest=digest_of(sql),
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
            sql_digest=digest_of(sql),
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
            sql_digest=digest_of(sql),
            reasons=(f"could not be parsed as {dialect}: {type(exc).__name__}",),
        )

    parsed: list[exp.Expression] = [s for s in statements if isinstance(s, exp.Expression)]
    if len(parsed) != 1:
        return PolicyDecision(
            allowed=False,
            dialect=dialect,
            policy_version=POLICY_VERSION,
            sql_digest=digest_of(sql),
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

    file_readers = sorted({type(n).__name__ for n in tree.find_all(*_FILE_READING_NODES)})
    if file_readers:
        reasons.append(f"reads files ({', '.join(file_readers)})")

    string_tables = _string_table_names(tree, sql, _sqlglot_dialect(dialect))
    if string_tables:
        reasons.append(
            "uses a string as a table name, which reads it as a file: "
            + ", ".join(repr(name) for name in string_tables)
        )

    anonymous = tuple(sorted({n.name.lower() for n in tree.find_all(exp.Anonymous) if n.name}))
    permitted = allowed_anonymous(dialect)
    refused = [name for name in anonymous if name not in permitted]
    if refused:
        reasons.append(f"calls unrecognised function(s): {', '.join(refused)}")

    return PolicyDecision(
        allowed=not reasons,
        dialect=dialect,
        policy_version=POLICY_VERSION,
        sql_digest=digest_of(sql),
        reasons=tuple(reasons),
        anonymous_functions=anonymous,
    )
