"""
nlqueries.feedback.promoter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Promotes positively-rated feedback into the ``agent_{id}_verified`` Qdrant
collection so that future prompt assemblies can blend user-confirmed correct
examples into the dynamic context.

Public API
----------
``promote_feedback(agent_id) -> int``
    Load feedback for *agent_id*, filter for positive ratings, validate each
    SQL against the current KB schema, and upsert qualifying pairs to
    ``agent_{safe_id}_verified``.  Returns the count of newly upserted points.

The function is idempotent — the same (question, sql) pair always maps to the
same Qdrant point ID, so repeated calls produce the same collection state.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from nlqueries import config

if TYPE_CHECKING:
    import sqlglot.expressions as exp

_log = logging.getLogger(__name__)


VERIFIED_COLLECTION_PREFIX = "agent_"
VERIFIED_COLLECTION_SUFFIX = "_verified"
VERIFIED_VECTOR_SIZE = config.EMBED_DIMENSIONS  # the embedding model's width


def _verified_collection(agent_id: str) -> str:
    # The rule every store keyed by agent uses. This one turned `.` into `_`,
    # so `sales.eu` and `sales_eu` shared one collection of verified examples.
    from nlqueries.state_files import agent_storage_id  # noqa: PLC0415

    return f"{VERIFIED_COLLECTION_PREFIX}{agent_storage_id(agent_id)}{VERIFIED_COLLECTION_SUFFIX}"


def _pair_point_id(question: str, sql: str) -> int:
    """Deterministic unsigned-64-bit Qdrant point ID for a (question, sql) pair."""
    digest = hashlib.sha256(f"{question}\x00{sql}".encode()).hexdigest()
    return int(digest[:16], 16)


def _load_kb(agent_id: str) -> dict[str, Any]:
    """Load the YAML knowledge base for *agent_id*, returning {} on failure."""
    from nlqueries import config  # noqa: PLC0415
    from nlqueries.state_files import agent_storage_id  # noqa: PLC0415

    kb_path = config.KB_PATH / f"{agent_storage_id(agent_id)}.yaml"
    try:
        import yaml  # noqa: PLC0415

        if kb_path.exists():
            return yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        # Read as empty, it refuses every promotion as having no schema to
        # check against (NO_SCHEMA), so the reason it was empty is said here.
        _log.warning("Could not read the knowledge base for %r at %s: %s", agent_id, kb_path, exc)
    return {}


#: Why :func:`why_not_promotable` refused when there is nothing to check
#: against; `promote_feedback` reports it on its own, since it refuses all.
NO_SCHEMA = "the knowledge base has no schema to check against"


def why_not_promotable(sql: str, schema_tables: set[str], dialect: str | None = None) -> str | None:
    """Why *sql* is not one read-only query over *schema_tables*, or None if it is.

    Fails closed: there is a reason when there is no schema to check against,
    when the statement does not parse in *dialect*, when it is more than one
    statement, when it is not a query or contains a write, when it names no
    table, and when it names a table the schema does not have. SQL that passes
    is shown to the model as a confirmed example for every later question, so
    anything this cannot vouch for is left out; it used to pass the first two.

    *schema_tables* are lower-cased table names, with ``schema.table`` too for
    the tables the knowledge base records a schema for (:func:`_schema_tables`).
    A table the SQL names with a schema must match it where the knowledge base
    has one: ``public.orders`` in the knowledge base does not vouch for
    ``other.orders``. *dialect* is the agent's, so that backtick or bracket
    quoting parses; None is sqlglot's default.
    """
    import sqlglot  # noqa: PLC0415
    import sqlglot.expressions as exp  # noqa: PLC0415
    from sqlglot.errors import SqlglotError  # noqa: PLC0415

    from nlqueries.sql_policy import (  # noqa: PLC0415
        _ALLOWED_ROOTS,
        _FORBIDDEN_NODES,
        _sqlglot_dialect,
    )

    if not schema_tables:
        return NO_SCHEMA
    try:
        read = _sqlglot_dialect(dialect) if dialect else None
        statements = [s for s in sqlglot.parse(sql, read=read) if s is not None]
    except (SqlglotError, ValueError, RecursionError):
        return "it does not parse"
    if len(statements) != 1:
        return "it is not a single statement"
    stmt = statements[0]
    if not isinstance(stmt, _ALLOWED_ROOTS) or next(stmt.find_all(*_FORBIDDEN_NODES), None):
        return "it is not a read-only query"
    try:
        cte_references = _cte_references(stmt)
    except (SqlglotError, ValueError, RecursionError):
        return "its table references cannot be resolved"
    if _repeats_a_source_name(stmt):
        return "it gives two of its sources the same name"
    ast_tables = {
        (t.catalog.lower(), t.db.lower(), t.name.lower())
        for t in stmt.find_all(exp.Table)
        if t.name and id(t) not in cte_references
    }
    if not ast_tables:
        return "it names no table"
    if not all(_known_table(*table, schema_tables) for table in ast_tables):
        return "it names a table not in the knowledge base"
    # The SQL is shown to the model as rendered, on one line once comments are
    # gone; a line break left inside a quoted name or value would start a line
    # of its own in the prompt. No real name needs one, so it is refused.
    if len(stmt.sql(dialect=read, comments=False).splitlines()) > 1:
        return "it has a line break inside a name or value"
    return None


def _cte_references(stmt: exp.Expression) -> set[int]:
    """The ids of the table nodes in *stmt* that refer to a CTE.

    Resolved scope by scope, as sqlglot resolves names: a bare name is a CTE
    only where that CTE is visible. Matching names across the statement let a
    real table through when a CTE of its name was defined in an inner scope,
    or after it. A qualified name is never a CTE.

    sqlglot keeps each real table node itself among its scope's sources, and a
    CTE reference as the CTE's scope, so a node kept as itself is a table
    whatever it is called. Looking the alias up instead let one through: an
    alias that repeats an earlier source's name is filed under a new name,
    and the alias still found the earlier source."""
    from sqlglot.optimizer.scope import Scope, build_scope  # noqa: PLC0415

    root = build_scope(stmt)
    if root is None:
        return set()
    references: set[int] = set()
    for scope in root.traverse():
        kept_as_themselves = {id(source) for source in scope.sources.values()}
        # And the name has to resolve to a CTE here: a node sqlglot keeps
        # neither way (none with a name does today) is checked, not skipped.
        for table in scope.tables:
            if (
                not table.db
                and id(table) not in kept_as_themselves
                and isinstance(scope.sources.get(table.name), Scope)
            ):
                references.add(id(table))
    return references


def _repeats_a_source_name(stmt: exp.Expression) -> bool:
    """Whether a query in *stmt* gives two of its FROM and JOIN sources one name.

    No query needs to, most engines refuse it, and a repeated name is how a
    table once passed for a CTE: refused on sight, it cannot be misread."""
    import sqlglot.expressions as exp  # noqa: PLC0415

    for select in stmt.find_all(exp.Select):
        # sqlglot 30 keys the clause "from_"; the releases pyproject still allows
        # used "from", where "from_" alone was None and the FROM source went
        # uncompared. As verbalizer.ast_walk reads it.
        from_ = select.args.get("from_") or select.args.get("from")
        sources = ([from_.this] if from_ else []) + [
            join.this for join in select.args.get("joins") or []
        ]
        names = [source.alias_or_name.lower() for source in sources if source.alias_or_name]
        if len(names) != len(set(names)):
            return True
    return False


def _known_table(catalog: str, schema: str, name: str, schema_tables: set[str]) -> bool:
    """Whether the table *catalog*.*schema*.*name* the SQL names is one in
    *schema_tables*.

    Unqualified, by name. Qualified, by schema and name -- unless the knowledge
    base recorded that name without a schema, when there is nothing to compare
    the qualifier with and the name has to do. Naming a database (or project)
    too, by all three: the knowledge base records the one it was built from,
    and a table in any other was never exported."""
    if catalog:
        return f"{catalog}.{schema}.{name}" in schema_tables
    if not schema:
        return name in schema_tables
    if f"{schema}.{name}" in schema_tables:
        return True
    return name in schema_tables and not any(entry.endswith(f".{name}") for entry in schema_tables)


def reads_only_known_tables(sql: str, schema_tables: set[str], dialect: str | None = None) -> bool:
    """Whether *sql* is one read-only query over tables in *schema_tables*."""
    return why_not_promotable(sql, schema_tables, dialect) is None


def without_comments(sql: str, dialect: str | None = None) -> str:
    """*sql* re-rendered in *dialect* with its comments removed.

    Example SQL is shown to the model verbatim beneath a one-line comment, and
    sqlglot keeps comments, line breaks and all, so a comment in it would carry
    any text into the prompt. Unparseable SQL is returned as it is; it is
    refused before it is used.
    """
    import sqlglot  # noqa: PLC0415
    from sqlglot.errors import SqlglotError  # noqa: PLC0415

    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    read = _sqlglot_dialect(dialect) if dialect else None
    try:
        return sqlglot.parse_one(sql, read=read).sql(dialect=read, comments=False)
    except (SqlglotError, ValueError, RecursionError):
        return sql


def _has_comments(sql: str, dialect: str | None) -> bool:
    import sqlglot  # noqa: PLC0415
    from sqlglot.errors import SqlglotError  # noqa: PLC0415

    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    read = _sqlglot_dialect(dialect) if dialect else None
    try:
        tree = sqlglot.parse_one(sql, read=read)
    except (SqlglotError, ValueError, RecursionError):
        return False
    return tree.sql(dialect=read) != tree.sql(dialect=read, comments=False)


def prune_verified(agent_id: str, *, dialect: str | None = None) -> int:
    """Delete *agent_id*'s verified pairs that would not be promoted today.

    Pairs promoted before the current rules were checked against looser ones:
    unparseable SQL and SQL over a knowledge base with no schema went through,
    and questions and SQL were stored as typed, comments and line breaks
    included. Each stored pair is checked again, as :func:`promote_feedback`
    checks a new one, and removed if it fails. Returns how many were removed.

    Nothing is removed when the knowledge base has no schema, since then
    nothing can be checked; :func:`promote_feedback` says so.
    """
    from qdrant_client.models import (  # noqa: PLC0415
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
    )

    from nlqueries.embeddings.qdrant_client import build_qdrant_client  # noqa: PLC0415
    from nlqueries.feedback.corrections import MAX_INTENT_CHARS, one_line  # noqa: PLC0415

    schema_tables = _schema_tables(_load_kb(agent_id))
    if not schema_tables:
        return 0
    client = build_qdrant_client()
    collection = _verified_collection(agent_id)
    if collection not in {c.name for c in client.get_collections().collections}:
        return 0

    only_this_agent = Filter(
        must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
    )
    unqualified: list[Any] = []
    scanned = 0
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=only_this_agent,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            scanned += 1
            payload = point.payload or {}
            sql = str(payload.get("sql") or "")
            question = str(payload.get("question") or "")
            reason = _why_not_kept(sql, schema_tables, dialect)
            if reason is None and _has_comments(sql, dialect):
                reason = "it has comments"
            if reason is None and (
                len(question.splitlines()) > 1 or len(question) > MAX_INTENT_CHARS
            ):
                reason = "its question is not one bounded line"
            # Promotion writes the pair normalised, and the point id follows
            # from what is written: a pair stored before that would sit beside
            # its normalised copy, a duplicate taking one of the few example
            # slots. It is the normalised copy that stays.
            if reason is None and (
                question != one_line(question) or sql != without_comments(sql, dialect)
            ):
                reason = "it is stored as it was before promotion normalised pairs"
            if reason is not None:
                # Each pair by name before it goes: the refusal need not be the
                # pair's fault -- a dialect sqlglot has no grammar for, read as
                # Postgres, fails every pair -- and a removed pair is not
                # promoted again, so this line is all that is left of it.
                _log.warning(
                    "prune_verified: removing verified pair %s for %r because %s: "
                    "question %.200r, sql %.500r",
                    point.id,
                    agent_id,
                    reason,
                    question,
                    sql,
                )
                unqualified.append(point.id)
        if offset is None:
            break
    # Every stored pair failing at once says more about the check than about
    # the pairs: a connector whose engine sqlglot has no grammar for is read as
    # Postgres, and then every pair "does not parse". Deleting is irreversible
    # and runs unasked from export-kb, so in that case nothing goes.
    if unqualified and len(unqualified) == scanned:
        _log.warning(
            "prune_verified: every one of the %d verified pair(s) for %r failed the "
            "re-check, so none was removed. When all fail at once the cause is more "
            "likely the check than the pairs -- a dialect read as Postgres because "
            "sqlglot has no grammar for the engine fails every one. Check the "
            "connector's dialect; delete the pairs yourself if they really are stale.",
            scanned,
            agent_id,
        )
        return 0
    if unqualified:
        client.delete(collection_name=collection, points_selector=PointIdsList(points=unqualified))
        _log.warning(
            "prune_verified: removed %d verified pair(s) for %r that no longer qualify.",
            len(unqualified),
            agent_id,
        )
    return len(unqualified)


def known_tables(tables: Iterable[tuple[Any, Any]], database: Any = None) -> set[str]:
    """The tables SQL is checked against, from (schema, name) pairs, lower-cased:
    each by name, as ``schema.table`` as well where a schema is recorded, and
    as ``database.schema.table`` where the *database* is known too.

    Promotion and `export-kb`'s corrections both check against this, so a
    qualified name is compared with its schema in both."""
    found: set[str] = set()
    for schema, name in tables:
        if not name:
            continue
        lowered = str(name).lower()
        found.add(lowered)
        if schema:
            found.add(f"{str(schema).lower()}.{lowered}")
            if database:
                found.add(f"{str(database).lower()}.{str(schema).lower()}.{lowered}")
    return found


def _schema_tables(knowledge_base: dict[str, Any]) -> set[str]:
    """The knowledge base's tables, as :func:`known_tables` gives them."""
    return known_tables(
        (
            (t.get("schema"), t.get("name"))
            for t in knowledge_base.get("schema", {}).get("tables", [])
            if isinstance(t, dict)
        ),
        knowledge_base.get("db_name"),
    )


def _why_not_kept(sql: str, schema_tables: set[str], dialect: str | None) -> str | None:
    """Why a verified pair's *sql* is not promoted or kept, or None.

    The SQL policy first, as for a correction (`select_corrections`): both
    reach the prompt as examples, and :func:`why_not_promotable` alone checks
    the statement's shape and tables, so a call the policy refuses --
    ``read_parquet(...)`` -- was promoted as verified."""
    from nlqueries.sql_policy import evaluate  # noqa: PLC0415

    if not evaluate(sql, dialect or "").allowed:
        return "the SQL policy refuses it"
    return why_not_promotable(sql, schema_tables, dialect)


def _sql_references_known_tables(sql: str, knowledge_base: dict[str, Any]) -> bool:
    """:func:`reads_only_known_tables` against the tables in *knowledge_base*."""
    return reads_only_known_tables(sql, _schema_tables(knowledge_base))


def promote_feedback(
    agent_id: str,
    *,
    dry_run: bool = False,
    include_unattributed: bool = False,
    dialect: str | None = None,
) -> int | list[dict[str, str]]:
    """Promote positively-rated feedback pairs to the verified Qdrant collection.

    Only ``rating == "up"`` records with a non-empty SQL (``corrected_sql``
    preferred over ``generated_sql``) are considered.  Each SQL is validated
    against the current KB schema; pairs referencing removed tables are silently
    dropped.

    The upsert is idempotent — repeated calls with the same data produce the
    same collection state.

    Records whose origin cannot be established are not promoted unless
    *include_unattributed* is set. Promotion makes a (question, SQL) pair an
    exemplar the model is shown for every later question on this agent, and the
    rating that qualifies a record is supplied by whoever submitted it. Over an
    unauthenticated transport that is anyone who can reach the server, so the
    pair they chose would steer generation for everybody (SEC-10).

    Args:
        agent_id: The agent whose feedback file and KB to use.
        include_unattributed: Promote records from `mcp` and older records with
                  no recorded origin as well. For an operator who has reviewed
                  them; the count is reported either way rather than passed over
                  in silence.
        dialect:  The agent's SQL dialect, so its quoting parses when the SQL
                  is checked. None parses as sqlglot's default.
        dry_run:  When True, run only the selection (no embedding, no Qdrant
                  upsert) and return the list of ``{"question", "sql"}`` pairs a
                  real run would promote. It reuses the exact same selection
                  code, so the preview matches what a subsequent real run pushes.

    Returns:
        Normally, the number of (question, sql) pairs upserted in this call.
        When ``dry_run`` is True, the list of pending ``{"question", "sql"}``
        pairs instead.
    """
    from nlqueries.feedback.models import ATTRIBUTED_SOURCES  # noqa: PLC0415
    from nlqueries.feedback.store import load_feedback  # noqa: PLC0415

    records = load_feedback(agent_id)
    positive = [r for r in records if r.rating == "up"]
    if not include_unattributed:
        attributed = [r for r in positive if r.source in ATTRIBUTED_SOURCES]
        skipped = len(positive) - len(attributed)
        if skipped:
            _log.warning(
                "promote_feedback: skipping %d positively-rated record(s) whose "
                "origin cannot be established. Re-run with include_unattributed "
                "to promote them.",
                skipped,
            )
        positive = attributed
    if not positive:
        return [] if dry_run else 0

    from nlqueries.feedback.corrections import one_line  # noqa: PLC0415

    schema_tables = _schema_tables(_load_kb(agent_id))
    collection = _verified_collection(agent_id)
    if not schema_tables:
        # Said on its own: every pair would be refused for it, and the usual
        # cause is a knowledge base never exported, or written elsewhere.
        _log.warning(
            "promote_feedback: nothing promoted for %r because %s; run "
            "`nlqueries export-kb` for it first.",
            agent_id,
            NO_SCHEMA,
        )
        return [] if dry_run else 0

    # Dedup: (normalized_sql → first record seen)
    seen_sql: dict[str, bool] = {}
    candidates: list[tuple[str, str]] = []
    refused: Counter[str] = Counter()
    for rec in positive:
        sql = rec.corrected_sql or rec.generated_sql
        if not sql or not sql.strip():
            continue
        norm = sql.strip().lower()
        if norm in seen_sql:
            continue
        seen_sql[norm] = True
        reason = _why_not_kept(sql, schema_tables, dialect)
        if reason is not None:
            refused[reason] += 1
            continue
        # As shown to the model: the question on one bounded line, the SQL
        # without comments, so neither can carry text into the prompt.
        candidates.append((one_line(rec.question), without_comments(sql, dialect)))
    if refused:
        _log.warning(
            "promote_feedback: skipped %d pair(s) for %r: %s.",
            sum(refused.values()),
            agent_id,
            "; ".join(f"{n} because {why}" for why, n in refused.most_common()),
        )

    if not candidates:
        return [] if dry_run else 0

    if dry_run:
        return [{"question": q, "sql": s} for q, s in candidates]

    # Side effects (embedding + Qdrant) only past this point — imported lazily so
    # a dry-run preview never needs the vector-store dependencies installed.
    from qdrant_client.models import PointStruct  # noqa: PLC0415

    from nlqueries.embeddings.embedder import embed_batch  # noqa: PLC0415
    from nlqueries.embeddings.qdrant_store import ensure_collection  # noqa: PLC0415

    ensure_collection(collection, VERIFIED_VECTOR_SIZE)

    try:
        from nlqueries.embeddings.qdrant_client import build_qdrant_client  # noqa: PLC0415

        client = build_qdrant_client()
    except Exception as exc:  # noqa: BLE001
        _log.warning("promote_feedback: Qdrant unavailable — %s", exc)
        return 0

    # Batch-embed all questions in a single call (one HTTP round-trip to daemon).
    questions = [q for q, _s in candidates]
    try:
        vectors = embed_batch(questions)
    except Exception as exc:  # noqa: BLE001
        _log.warning("promote_feedback: embed_batch failed — %s", exc)
        return 0

    from nlqueries.cache.envelope import sign_pair  # noqa: PLC0415

    points: list[PointStruct] = []
    for (question, sql), vector in zip(candidates, vectors, strict=True):
        point_id = _pair_point_id(question, sql)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                # Signed, so the reader can tell a pair promoted here from one
                # written straight into Qdrant. Unsigned pairs are ignored.
                payload=sign_pair(
                    {
                        "question": question,
                        "sql": sql,
                        "agent_id": agent_id,
                        "verified": True,
                    }
                ),
            )
        )

    if not points:
        return 0

    try:
        client.upsert(collection_name=collection, points=points)
    except Exception as exc:  # noqa: BLE001
        _log.warning("promote_feedback: upsert failed — %s", exc)
        return 0

    return len(points)
