# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.sql_policy
~~~~~~~~~~~~~~~~~~~~
Whether a generated statement is one this product is willing to run.

Rules are applied to the whole parsed tree. The checks this replaced examined
only the root node, and every payload in ``tests/security/payloads`` has a
``Select`` at its root.

What a query may be built from is listed, per dialect: the expression classes
in :mod:`nlqueries.sql_constructs`, and the functions sqlglot does not model
(:class:`exp.Anonymous`), by name, in :data:`ALLOWED_ANONYMOUS`. Anything not
listed is refused, so a function sqlglot happens to model is no longer allowed
for that alone. ``tests/sql_corpus.py`` holds the analytics queries each list
must keep allowing.

A statement is also parsed again from the form sqlglot renders it in, and
refused when the policy reads that differently: another statement count or
root, other tables, or a refusal. The rules judge sqlglot's reading of a
statement, and one whose reading does not survive a round trip is one sqlglot
is not reading reliably.

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

from nlqueries.sql_constructs import allowed_constructs

#: Incremented when a decision made by this module could change. Recorded on
#: every decision so a stored verdict can be told apart from a current one.
#:
#: 2: the anonymous-function allowlist is keyed by sqlglot's dialect names and
#: looked up through the aliases below, so ``postgresql`` now gets Postgres's
#: entry, where it used to miss the table and be allowed nothing.
#:
#: 3: the file-reading functions sqlglot models, and a string literal used as a
#: table name, are refused.
#:
#: 4: only listed constructs are allowed, per dialect (:mod:`.sql_constructs`);
#: unmodelled aggregates are checked by name like other unmodelled functions;
#: and a statement whose rendered form the policy reads differently is refused.
#:
#: 5: an unmodelled function written with a qualifier is judged by the whole
#: name, so ``some_schema.age`` no longer passes on the entry for ``age``; and
#: one whose name cannot be read is refused.
POLICY_VERSION = "5"

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
    "postgres": frozenset(
        {
            "age", "jsonb_agg", "regexp_matches", "every", "date_part", "cardinality",
            "make_date", "to_json", "json_agg", "json_build_object", "jsonb_build_object",
        }
    ),
    "redshift": frozenset({"age", "regexp_matches", "every", "date_part"}),
    "snowflake": frozenset({"every", "to_timestamp"}),
    "bigquery": frozenset(),
    "tsql": frozenset(),
    "mysql": frozenset({"now", "sysdate", "unix_timestamp", "json_array"}),
    "sqlite": frozenset({"julianday", "datetime", "time", "unixepoch"}),
    "duckdb": frozenset(
        {"every", "age", "cardinality", "date_part", "datepart", "date_sub", "now", "to_json"}
    ),
}  # fmt: skip

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
    #: Expression classes in the statement that the dialect's list does not
    #: have, by name. Recorded for the inventory, as the functions are.
    unlisted_constructs: tuple[str, ...] = ()

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


def _string_table_names(tree: exp.Expression, sql: str, dialect: str) -> list[str] | None:
    """Table names in *tree* that were written as string literals, or None.

    DuckDB reads ``FROM 'orders.csv'`` as a scan of that file, and sqlglot
    parses the string into a quoted identifier indistinguishable from
    ``FROM "orders.csv"`` -- a real table name. What tells them apart is the
    token the identifier came from, so the statement is tokenized and any table
    name starting where a string token starts is reported.

    The start position is parser metadata sqlglot does not document; it is
    present on the versions this was measured against (30.9 and later), and
    the declared floor is older. None means the question cannot be answered:
    the statement has string tokens and table names, but no table name carries
    a start position, or the statement could not be tokenized. The caller
    refuses it rather than assume none of the strings is a table name.
    """
    try:
        tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    except (SqlglotError, ValueError):
        return None
    string_starts = {token.start for token in tokens if token.token_type == TokenType.STRING}
    if not string_starts:
        return []
    parts = [
        part
        for table in tree.find_all(exp.Table)
        for part in table.parts
        if isinstance(part, exp.Identifier)
    ]
    if parts and not any("start" in part.meta for part in parts):
        return None
    return [part.name for part in parts if part.meta.get("start") in string_starts]


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
    reading = _read(tree, sql, dialect)
    reasons.extend(reading.reasons)
    anonymous, unlisted = reading.anonymous, reading.unlisted
    if not reasons:
        differs, second = _rereading_differs(tree, reading, dialect)
        reasons.extend(differs)
        # What only the rendering has -- a construct sqlglot's canonical form
        # introduced -- is what the inventory must count for the refusal to be
        # actionable, so the second reading's lists join the first's.
        if differs and second is not None:
            anonymous = tuple(sorted({*anonymous, *second.anonymous}))
            unlisted = tuple(sorted({*unlisted, *second.unlisted}))

    return PolicyDecision(
        allowed=not reasons,
        dialect=dialect,
        policy_version=POLICY_VERSION,
        sql_digest=digest_of(sql),
        reasons=tuple(reasons),
        anonymous_functions=anonymous,
        unlisted_constructs=unlisted,
    )


@dataclass(frozen=True)
class _Reading:
    """What the rules make of one parsed statement."""

    reasons: tuple[str, ...]
    anonymous: tuple[str, ...]
    unlisted: tuple[str, ...]
    tables: frozenset[tuple[str, ...]]


#: Unmodelled functions, judged by name rather than by class. Aggregates
#: included: sqlglot has a class of its own for an unmodelled aggregate.
_UNMODELLED: tuple[type[exp.Expression], ...] = (exp.Anonymous, exp.AnonymousAggFunc)


def _function_name(node: exp.Func, dialect: str) -> str | None:
    """The name a function call is judged by: "" if it cannot be read, None if
    the call is judged by its class instead.

    An unmodelled call is judged by name. So is any call written with a
    qualifier, ``some_schema.age(a, b)``, modelled or not: the qualifier is
    part of the name, because an entry vouches for one function, and the one
    ``some_schema`` holds is not the built-in it shadows -- nor, for a call
    sqlglot models, the one its class stands for. A qualified call is
    allowed only if listed as written.

    The qualifier is not always the call's parent. In a select list it is a
    `Dot` whose right-hand side is the call; a suffix -- ``FILTER (WHERE
    ...)``, ``WITHIN GROUP (...)``, ``IGNORE NULLS`` -- wraps the call first,
    and then it lands in a `Dot` around the wrapper or, as table, schema and
    catalog, in a `Column` whose `this` the wrapper is; in a ``FROM`` it is
    the schema and catalog of the `Table` whose `this` the call is. So the
    search goes up through every node the call heads (is the `this` of) until
    it finds one of those. Taking a qualifier that was not meant can only
    refuse a call; missing one allows it. The shapes are sqlglot's, measured
    against 30.17: a release that attached the qualifier some other way
    would leave it unseen, and the tests for each shape would fail there.
    """
    qualifier = _qualifier(node)
    if isinstance(node, _UNMODELLED):
        written = node.name
    elif not qualifier:
        return None
    else:
        written = _written_name(node, dialect)
    names = [*qualifier, written]
    if not all(names):
        return ""
    return ".".join(names).lower()


def _written_name(node: exp.Func, dialect: str) -> str:
    """How *dialect* spells the function a modelled call stands for: its class
    is named for sqlglot (`ExplodingGenerateSeries`), not for the database."""
    try:
        rendered = node.sql(dialect=_sqlglot_dialect(dialect))
    except Exception:  # noqa: BLE001 - "" is unreadable, which refuses
        return ""
    name, paren, _rest = rendered.partition("(")
    return name.strip() if paren else ""


def _qualifier(node: exp.Func) -> list[str]:
    """The names qualifying the call *node*, outermost first, "" for one that
    cannot be read; empty when there is no qualifier."""
    # `object`: sqlglot types a node's parent as its base class, not Expression.
    child: object = node
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Dot) and parent.expression is child:
            names = []
            qualifier = parent.this
            while isinstance(qualifier, exp.Dot):
                names.append(qualifier.expression.name)
                qualifier = qualifier.this
            # A Column at the bottom keeps its own table, schema and catalog.
            parts = qualifier.parts if isinstance(qualifier, exp.Column) else [qualifier]
            names.extend(part.name for part in reversed(parts))
            names.reverse()
        elif isinstance(parent, exp.Column) and parent.this is child:
            names = _named_parts(parent, ("catalog", "db", "table"))
        elif isinstance(parent, exp.Table) and parent.this is child:
            names = _named_parts(parent, ("catalog", "db"))
        elif parent.args.get("this") is child:
            child, parent = parent, parent.parent
            continue
        else:
            return []
        return names
    return []


def _named_parts(node: exp.Expression, keys: tuple[str, ...]) -> list[str]:
    return [part.name for part in (node.args.get(key) for key in keys) if part is not None]


def _read(tree: exp.Expression, sql: str, dialect: str) -> _Reading:
    """Apply every rule to *tree*, parsed from *sql*."""
    reasons: list[str] = []
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
    if string_tables is None:
        reasons.append(
            "could not tell whether a table name is a string, which would read it as "
            "a file: the parser recorded no token positions"
        )
    elif string_tables:
        reasons.append(
            "uses a string as a table name, which reads it as a file: "
            + ", ".join(repr(name) for name in string_tables)
        )

    names = [
        name
        for name in (_function_name(n, dialect) for n in tree.find_all(exp.Func))
        if name is not None
    ]
    anonymous = tuple(sorted({name for name in names if name}))
    permitted = allowed_anonymous(dialect)
    refused = [name for name in anonymous if name not in permitted]
    if refused:
        reasons.append(f"calls unrecognised function(s): {', '.join(refused)}")
    # Exempt from the class list below and judged by name, so one whose name
    # cannot be read would otherwise be judged by neither.
    if not all(names):
        reasons.append("calls a function whose name could not be read")

    # Exact classes, not isinstance: a subclass of a listed class is another
    # construct, and is not allowed by inheriting. What a rule above already
    # names is not named twice.
    listed = allowed_constructs(_sqlglot_dialect(dialect))
    # Classes, not names looked up in `exp`: sqlglot does not export every
    # class its parser produces under its own name (`_TimeUnit` is not), and a
    # lookup that failed would raise here rather than refuse.
    # The root's class only when the root rule refused it: a subclass of
    # `Select` passes that rule's isinstance, and must still be listed itself.
    already_named = {*forbidden, *file_readers}
    if not isinstance(tree, _ALLOWED_ROOTS):
        already_named.add(root_name)
    unlisted = tuple(
        sorted(
            {
                type(n).__name__
                for n in tree.walk()
                if not isinstance(n, _UNMODELLED) and type(n) not in listed
            }
            - already_named
        )
    )
    if unlisted:
        reasons.append(
            f"uses construct(s) not on the {_sqlglot_dialect(dialect)} list: {', '.join(unlisted)}"
        )

    # SQL Server still reads `FROM orders o (TABLOCKX)`, a table hint written
    # without WITH, which sqlglot parses as a column list on the alias -- every
    # class of it listed. It allows a column list only on a derived table,
    # whose alias sits on the subquery, so one on a table is only ever a hint.
    if _sqlglot_dialect(dialect) == "tsql":
        hinted = sorted(
            {
                table.name
                for table in tree.find_all(exp.Table)
                if (alias := table.args.get("alias")) is not None and alias.args.get("columns")
            }
        )
        if hinted:
            reasons.append(
                f"gives {', '.join(hinted)} a parenthesised list after its alias, which "
                "SQL Server reads as a table hint"
            )

    tables = frozenset(
        tuple(part.name for part in table.parts) for table in tree.find_all(exp.Table)
    )
    return _Reading(tuple(reasons), anonymous, unlisted, tables)


def _rendered(tree: exp.Expression, dialect: str) -> str:
    return tree.sql(dialect=_sqlglot_dialect(dialect))


def _rereading_differs(
    tree: exp.Expression, reading: _Reading, dialect: str
) -> tuple[list[str], _Reading | None]:
    """Why sqlglot's rendering of *tree*, parsed again, reads differently, if it does,
    with the rendering's reading when there is one.

    Compared as the policy reads it rather than node for node. sqlglot
    canonicalises freely -- MySQL's ``IF`` comes back as ``CASE``, Snowflake's
    ``profile:tier`` as ``GET_PATH`` -- and a node-for-node comparison refused
    six of the corpus queries for it. What matters is whether the round trip
    changes the number of statements, the root, the tables, or whether a rule
    applies.
    """
    try:
        # Rendering inside the `try` too: sqlglot's generator recurses where its
        # parser does not, so `a::INT::INT::...`, five hundred casts long, parses
        # and reads cleanly and then exhausts the stack here. A statement this
        # cannot render is one it cannot vouch for, and evaluate never raises.
        rendered = _rendered(tree, dialect)
        again = [s for s in sqlglot.parse(rendered, read=_sqlglot_dialect(dialect)) if s]
    except Exception:  # noqa: BLE001 - RecursionError, or anything the generator raises
        return ["its rendered form could not be produced or parsed again"], None
    if len(again) != 1:
        return [f"its rendered form parses as {len(again)} statements"], None
    if type(again[0]) is not type(tree):
        return [f"its rendered form parses as a {type(again[0]).__name__}"], None
    second = _read(again[0], rendered, dialect)
    if second.tables != reading.tables:
        return ["its rendered form reads other tables"], second
    if second.reasons:
        return [f"its rendered form is refused: {'; '.join(second.reasons)}"], second
    return [], second
