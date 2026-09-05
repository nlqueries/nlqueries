"""
nlqueries.cache.semantic_cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic query cache backed by Qdrant with three lookup tiers.

**Tier 0 — exact-match** (zero embed calls on hit):
    Normalize the question (lowercase, strip punctuation, collapse whitespace),
    derive a deterministic point ID via SHA-256, and call ``client.retrieve()``
    directly.  No embedding is needed.

**Tier 1 — answer cache** (cosine similarity ≥ CACHE_ANSWER_THRESHOLD):
    Embed the question once (or reuse the pre-computed *vector* kwarg) and
    run a nearest-neighbour search filtered to ``kind=answer`` points.
    This is the original Sprint 21 cache, unchanged except for the kind filter.

**Tier 2 — template cache** (cosine similarity ≥ CACHE_TEMPLATE_THRESHOLD):
    Mask entity tokens in the question (dates → ``<DATE>``, numbers →
    ``<NUMBER>`` etc.) and search against stored ``kind=template`` points.
    On a hit, re-extract the concrete entity values from the original question
    and bind them into the parameterized SQL template
    (placeholders look like ``[column_name:DATE]``).  The bound SQL is
    validated syntactically before being returned.

Collection naming convention: ``cache_{agent_id}`` (one per agent, separate
from the ``agent_{id}_schema``, ``agent_{id}_capsules`` and
``doc_{source_id}_chunks`` collections used elsewhere).

Public API
----------
``CacheEntry``
    Dataclass representing a stored cache entry.
``SemanticCache``
    Per-agent cache manager with ``get``, ``put``, ``invalidate``, and
    ``stats`` methods.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import string
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

import sqlglot
from sqlglot import exp

from nlqueries import config
from nlqueries.cache.envelope import (
    SIGNATURE_KEY,
    VERSION_KEY,
    CacheBinding,
    sign,
    verify,
)
from nlqueries.embeddings.embedder import embed_text
from nlqueries.embeddings.qdrant_store import (
    ensure_collection,
    forget_collection_indexes,
)

logger = logging.getLogger(__name__)

CACHE_COLLECTION_PREFIX = "cache_"
#: The embedding model's width, not a number of its own. Declared once in
#: config so this collection cannot be built for one width while the model
#: produces another -- a mismatch writes vectors that do not mean what the
#: collection thinks they mean, and nothing raises.
CACHE_VECTOR_SIZE = config.EMBED_DIMENSIONS
SIMILARITY_THRESHOLD = 0.97  # kept for backward-compat; Tier 1 uses config value

# Qdrant collection names must not contain ":" or other special chars.
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]")

# Module-level set of known-existing Qdrant collection names.
_known_collections: set[str] = set()

# Collections we have looked for and not found, with the time each entry stops
# being trusted.
#
# Without this, only *hits* were cached: an agent whose cache collection does not
# exist yet paid a full get_collections() round trip on every single query,
# forever, until its first cache write created the collection. New agents were
# therefore slower than warm ones for a reason nobody would guess from the code.
#
# The entries expire because a collection created by another process — the CLI,
# a worker, a second API replica — would otherwise stay invisible here. Sixty
# seconds of staleness on a cache lookup is harmless: the consequence is a cache
# miss, which is what was happening anyway.
_missing_collections: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Entity masking helpers (Tier 2 template cache)
# ---------------------------------------------------------------------------

# Regex patterns applied in order (most specific first).
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
# Capture groups: the *value* is bound into SQL, and it must not carry the
# quote characters that delimited it in the question. `sub` still replaces the
# whole match, so `_mask_entities` -- and therefore every cache key ever written
# -- is unaffected by the groups being here.
_DQUOTE_RE = re.compile(r'"([^"]*)"')
_SQUOTE_RE = re.compile(r"'([^']*)'")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Each entry: (pattern, token_type_string)
_ENTITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_ISO_DATE_RE, "<DATE>"),
    (_MONTH_RE, "<MONTH>"),
    (_CURRENCY_RE, "<CURRENCY>"),
    (_DQUOTE_RE, "<STRING>"),
    (_SQUOTE_RE, "<STRING>"),
    (_NUMBER_RE, "<NUMBER>"),
]

# Regex to find parameterized SQL placeholders like [column_name:DATE]
_PLACEHOLDER_RE = re.compile(r"\[([^:\]]+):([^\]]+)\]")

# Maps parameterizer placeholder types → entity type keys used by _extract_entities_by_type
_PARAM_TYPE_TO_ENTITY: dict[str, str] = {
    "INT": "NUMBER",
    "DECIMAL": "NUMBER",
    "DATE": "DATE",
    "TIMESTAMP": "DATE",
    "VARCHAR": "STRING",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_question(question: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for exact-match lookup."""
    q = question.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", q).strip()


def _mask_entities(question: str) -> str:
    """Replace entity tokens in *question* with typed placeholders.

    Patterns are applied in order (most specific first) so that ISO dates are
    captured before the generic number pattern would match the year/month/day
    digits.
    """
    masked = question
    for pattern, token in _ENTITY_PATTERNS:
        masked = pattern.sub(token, masked)
    return masked


def _extract_entities_by_type(question: str) -> dict[str, list[str]]:
    """Return entity values from *question* grouped by type token (no angle brackets).

    Each pattern is applied to *question* and the matched strings are collected.
    A running copy of *question* is masked after each pattern so later patterns
    don't double-count characters that already matched.
    """
    result: dict[str, list[str]] = {}
    remaining = question
    for pattern, token in _ENTITY_PATTERNS:
        type_key = token[1:-1]  # strip < > → "DATE", "NUMBER", etc.
        values = pattern.findall(remaining)
        if values:
            result.setdefault(type_key, []).extend(values)
        remaining = pattern.sub(token, remaining)
    return result


#: Placeholders as they appear in a stored template, with the quotes the
#: parameterizer may have wrapped them in. Both forms occur: a VARCHAR
#: placeholder is emitted inside a string literal, an INT one bare.
_QUOTED_PLACEHOLDER_RE = re.compile(r"'?\[([^:\]]+):([^\]]+)\]'?")

#: What a placeholder becomes before the template is parsed. A bare
#: ``[x:INT]`` is not a literal in any dialect -- Postgres reads it as array
#: indexing, T-SQL as a quoted identifier -- so walking ``exp.Literal`` would
#: silently miss every numeric placeholder and bind only the strings. Turning
#: every placeholder into the same quoted sentinel first gives one code path
#: that finds all of them, in every dialect.
_BIND_SENTINEL_PREFIX = "__nlq_bind_"
_BIND_SENTINEL = _BIND_SENTINEL_PREFIX + "{}__"
_BIND_SENTINEL_RE = re.compile(r"^__nlq_bind_(\d+)__$")

#: Values a placeholder type will accept. A value that does not match is not
#: bound at all: `_bind_entities` returns None and the caller falls through to
#: generation, which is what it already does when an entity is missing.
_COERCE_PATTERNS: dict[str, re.Pattern[str]] = {
    "INT": re.compile(r"^\d+$"),
    "DECIMAL": re.compile(r"^\d+(\.\d+)?$"),
    "DATE": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "TIMESTAMP": re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$"),
}

#: Longer than any real filter value, and the shape a padded injection takes.
_MAX_VARCHAR_ENTITY_CHARS = 512


def _coerce_entity(value: str, param_type: str) -> str | None:
    """Return *value* if it is a legal value for *param_type*, else ``None``.

    The first of two independent defences. This one is cheap and readable and
    stops the obvious thing: a numeric placeholder can only ever receive digits,
    so no amount of SQL in a question can reach one. The second, the structural
    gate in :func:`_bind_entities`, is what holds when this is wrong.

    VARCHAR deliberately accepts anything printable -- a filter value legitimately
    contains apostrophes, spaces and punctuation -- and relies on the AST binding
    to make it inert. It refuses only two things: a NUL, which truncates in some
    drivers rather than erroring, and a value long enough to be padding rather
    than a filter.
    """
    ptype = param_type.upper()
    pattern = _COERCE_PATTERNS.get(ptype)
    if pattern is not None:
        return value if pattern.match(value) else None
    if "\x00" in value or len(value) > _MAX_VARCHAR_ENTITY_CHARS:
        return None
    return value


def _sentinel_template(template_sql: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace every placeholder with a quoted sentinel; return it and their order.

    The order matches ``_PLACEHOLDER_RE.findall``, so sentinel *i* belongs to
    placeholder *i* -- which is what lets a template mentioning the same column
    twice receive two different values, as the previous ``str.replace(..., 1)``
    loop did.
    """
    order: list[tuple[str, str]] = []

    def _sub(match: re.Match[str]) -> str:
        order.append((match.group(1), match.group(2)))
        return "'" + _BIND_SENTINEL.format(len(order) - 1) + "'"

    return _QUOTED_PLACEHOLDER_RE.sub(_sub, template_sql), order


def _literal_shape(tree: exp.Expression) -> str:
    """The statement with every literal blanked, for comparing shapes.

    Two statements with the same shape differ only in the values they compare
    against. That is the whole of what binding is allowed to do.
    """
    clone = tree.copy()
    for node in clone.find_all(exp.Literal):
        node.replace(exp.Literal.string("?"))
    return clone.sql()


def _sqlglot_name(dialect: str | None) -> str | None:
    """Translate a caller's dialect name into one sqlglot answers to.

    The name that reaches the cache is whatever the deployment calls its engine,
    and several of those are not sqlglot dialects: `CAPABILITIES` keys SQL Server
    as `mssql`, the MCP `query` tool documents `mssql` and `mysql`, and a binding
    may carry `postgresql` or `mariadb`. sqlglot rejects all three with
    ``ValueError: Unknown dialect``.

    Untranslated, that error is swallowed by :func:`_parse_or_none` and every
    Tier 2 hit becomes a miss on those deployments -- silently, and including the
    numeric templates that bound correctly before binding moved to the AST.
    `sql_policy.evaluate` translates the same strings for the same reason, and
    this defers to its table rather than keeping a second one.
    """
    if not dialect:
        return None
    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    return _sqlglot_dialect(dialect)


#: Search failures already reported, so an unreachable or too-old Qdrant costs
#: one line rather than one per lookup.
_SEARCH_FAILURES_LOGGED: set[str] = set()


def _report_search_failure(collection: str, tier: str, exc: BaseException) -> None:
    """Say that a lookup *failed* rather than letting it read as a miss.

    A cache miss and a rejected request are the same `None` to the caller, and
    they mean opposite things: the first is the cache working, the second is the
    cache not being consulted at all. Reported once per collection and tier,
    because the condition persists -- an unreachable Qdrant, or a server too old
    for the query API this uses, fails identically on every request.

    That second case is not hypothetical. `query_points` is Qdrant's Universal
    Query API, added in v1.10; against an older server every Tier 1 and Tier 2
    lookup returns 404 and, before this, was indistinguishable from a cache that
    simply had nothing to offer.
    """
    key = f"{collection}:{tier}"
    if key in _SEARCH_FAILURES_LOGGED:
        return
    _SEARCH_FAILURES_LOGGED.add(key)
    logger.warning(
        "Cache %s search on %s failed; treating it as a miss. Every lookup will "
        "miss until this is fixed. If the message below is a 404, the server "
        "predates the query API this uses -- Qdrant v1.10 or newer is required.",
        tier,
        collection,
        exc_info=exc,
    )


#: Unknown dialect names already reported, so an engine nobody registered costs
#: one line rather than one per cache lookup.
_UNKNOWN_DIALECTS_LOGGED: set[str] = set()


def _parse_or_none(sql: str, dialect: str | None) -> exp.Expression | None:
    """Parse *sql* as exactly one statement, or ``None``.

    An unparseable template or bound statement is a cache miss, not an error:
    the caller falls through to generation, which is the outcome that was always
    intended and did not happen while the parse result was discarded inside a
    `contextlib.suppress`.

    `parse`, not `parse_one`, and the count is checked -- the convention
    `sql_policy.evaluate` documents at its own parse call. `parse_one` is
    specified to return the first statement and discard the rest; this sqlglot
    version happens to return a `Block` for multi-statement input instead, whose
    shape does not match a single `Select`, so the gate in `_bind_entities` holds
    either way today. That is incidental rather than intended: this defence
    exists precisely for the case where a value has escaped its literal, so it
    should not rest on which of two behaviours the parser has this release.
    """
    read = _sqlglot_name(dialect)
    try:
        statements = sqlglot.parse(sql, read=read)
    except ValueError as exc:
        # sqlglot raises a plain ValueError for a dialect it does not know.
        # Distinguished from an unparseable statement and logged, because it is
        # a configuration problem an operator can fix and it disables every
        # Tier 2 hit for the life of the deployment. `_is_executable_select`
        # logs the same condition at ERROR for the same reason, and no longer
        # sees it now that binding fails first.
        if dialect and dialect not in _UNKNOWN_DIALECTS_LOGGED:
            _UNKNOWN_DIALECTS_LOGGED.add(dialect)
            logger.warning(
                "Cache template binding is disabled: %r is not a dialect sqlglot "
                "knows (%s). Every Tier 2 template hit will miss until the "
                "connector's db_type is corrected.",
                dialect,
                exc,
            )
        return None
    except Exception:  # noqa: BLE001 -- an unparseable statement is a cache miss
        return None

    parsed = [st for st in statements if st is not None]
    if len(parsed) != 1:
        return None
    # sqlglot declares these as `Expr`, the *parent* of `Expression`, so the
    # declaration is wider than what it returns and mypy will not narrow it.
    # Everything downstream here (`find_all`, `copy`, `sql`) is `Expression`
    # behaviour, and the runtime type always is one.
    return cast("exp.Expression", parsed[0])


def _bind_entities(question: str, template_sql: str, dialect: str | None = None) -> str | None:
    """Substitute question entities into *template_sql* placeholders.

    Placeholders look like ``[column_name:DATE]``.  Entities are matched using
    two strategies applied in order:

    1. **Semantic pre-assignment** (before positional fallback):

       - Placeholders whose name contains ``year`` receive the first 4-digit
         number found in the question (years are nearly always 4 digits).
       - ``param_N`` placeholders (LIMIT / OFFSET generated by the parameterizer
         when no column context is available) receive the number following
         ``top`` or ``first`` in the question (e.g. "top 10 movies" → 10).

    2. **Positional fallback**: remaining placeholders are filled left-to-right
       with remaining entities of the matching type.

    Returns ``None`` when there are fewer available entities than required
    placeholders of a given type (binding would produce an incomplete or
    incorrect query).
    """
    placeholders = _PLACEHOLDER_RE.findall(template_sql)  # [(name, type), ...]
    if not placeholders:
        return template_sql  # no placeholders — use as-is

    entities = _extract_entities_by_type(question)
    # Mutable copy of the number pool so pre-assignments can claim values.
    number_pool: list[str] = list(entities.get("NUMBER", []))

    # --- Semantic pre-assignment 1: year columns → first 4-digit number ---
    pre_assigned: dict[tuple[str, str], str] = {}
    four_digit = [n for n in number_pool if re.match(r"^\d{4}$", n)]
    for name, ptype in placeholders:
        if ptype.upper() in ("INT", "DECIMAL") and "year" in name.lower() and four_digit:
            val = four_digit.pop(0)
            pre_assigned[(name, ptype)] = val
            number_pool.remove(val)

    # --- Semantic pre-assignment 2: param_N (LIMIT) → "top N" / "first N" ---
    top_match = re.search(r"\b(?:top|first)\s+(\d+(?:\.\d+)?)\b", question, re.IGNORECASE)
    if top_match:
        top_val = top_match.group(1)
        for name, ptype in placeholders:
            if (
                re.match(r"^param_\d+$", name.lower())
                and ptype.upper() in ("INT", "DECIMAL")
                and (name, ptype) not in pre_assigned
                and top_val in number_pool
            ):
                pre_assigned[(name, ptype)] = top_val
                number_pool.remove(top_val)
                break  # one LIMIT per "top N" match

    # --- Positional fallback for remaining placeholders ---
    remaining: dict[str, list[str]] = {k: list(v) for k, v in entities.items()}
    remaining["NUMBER"] = number_pool
    type_used: dict[str, int] = {}

    values: list[str] = []
    for name, ptype in placeholders:
        if (name, ptype) in pre_assigned:
            raw = pre_assigned[(name, ptype)]
        else:
            entity_type = _PARAM_TYPE_TO_ENTITY.get(ptype.upper(), "STRING")
            idx = type_used.get(entity_type, 0)
            available = remaining.get(entity_type, [])
            if idx >= len(available):
                return None  # not enough entities -- unsafe to bind
            raw = available[idx]
            type_used[entity_type] = idx + 1

        coerced = _coerce_entity(raw, ptype)
        if coerced is None:
            return None  # not a legal value for this placeholder's type
        values.append(coerced)

    # From here a value never touches SQL text. It becomes a literal node in a
    # parsed tree and sqlglot renders it for the dialect, which makes quoting,
    # doubling and MySQL's backslash escaping its problem rather than ours.
    #
    # `str.replace` could not do that. It quoted values itself, inside a template
    # that had already quoted the placeholder, so `"East"` bound to `''"East"''`
    # and failed to parse on every dialect but MySQL. Two bugs cancelling out --
    # the regex kept the question's quote characters and the template supplied
    # its own -- and correcting either alone would have made the other exploitable.
    normalised, order = _sentinel_template(template_sql)
    template_tree = _parse_or_none(normalised, dialect)
    if template_tree is None or len(order) != len(values):
        return None

    tree = template_tree.copy()
    for node in tree.find_all(exp.Literal):
        match = _BIND_SENTINEL_RE.match(str(node.this))
        if match is None:
            continue
        position = int(match.group(1))
        if position >= len(values):
            return None
        ptype = order[position][1].upper()
        value = values[position]
        node.replace(
            exp.Literal.number(value) if ptype in ("INT", "DECIMAL") else exp.Literal.string(value)
        )

    bound = tree.sql(dialect=_sqlglot_name(dialect))

    # The gate that does not depend on getting the escaping right. A bound
    # statement may differ from its template only in the values of its literals;
    # if the shapes differ then something in a value became syntax, and the
    # result is discarded however it was quoted.
    bound_tree = _parse_or_none(bound, dialect)
    if bound_tree is None or _literal_shape(template_tree) != _literal_shape(bound_tree):
        return None

    # Fail closed if a sentinel survived, which the shape gate cannot see: it
    # blanks every literal, so an unreplaced `'__nlq_bind_0__'` is
    # indistinguishable from a bound value. Such a statement would pass the gate,
    # pass the re-parse, pass the SQL policy and run -- comparing a column
    # against the string `__nlq_bind_0__` and returning nothing, which reads to
    # the caller as "no matching rows" rather than as a fault.
    #
    # It should be unreachable, because every placeholder is normalised to a
    # quoted sentinel that parses as a literal in all five dialects. That is an
    # argument rather than a guarantee, and the check costs one substring scan.
    if _BIND_SENTINEL_PREFIX in bound:
        return None

    return bound


# ---------------------------------------------------------------------------
# Protocol — accepted by SemanticCache.put(); satisfied by AgentQueryResult
# and by the internal _CacheData carrier in multi_agent_orchestrator.
# ---------------------------------------------------------------------------


class _PutResult(Protocol):
    resolved_question: str
    agent_type: str
    answer: str
    sql: str | None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Stored cache entry returned on a semantic cache hit."""

    question: str
    resolved_question: str
    agent_type: str
    answer: str
    sql: str | None
    created_at: datetime
    hit_count: int
    kind: str = field(default="answer")


# ---------------------------------------------------------------------------
# Lazy QdrantClient singleton (separate from qdrant_store's singleton so that
# the cache module can be tested independently without touching capsule/chunk
# collections).
# ---------------------------------------------------------------------------

_cache_client: Any = None


def _get_client() -> Any:
    """Return a shared QdrantClient instance, creating it lazily on first call."""
    global _cache_client  # noqa: PLW0603
    if _cache_client is None:
        # One factory, one authentication rule — see qdrant_client's docstring.
        from nlqueries.embeddings.qdrant_client import build_qdrant_client  # noqa: PLC0415

        _cache_client = build_qdrant_client()
    return _cache_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collection_exists(client: Any, collection: str) -> bool:
    """Return True if *collection* exists in Qdrant.

    Both answers are cached. The positive cache is unbounded, which is fine —
    collection names are bounded by agent count. The negative cache expires
    after ``CACHE_COLLECTION_NEGATIVE_TTL_SECONDS`` so a collection created
    elsewhere becomes visible without a restart.
    """
    if collection in _known_collections:
        return True

    expires_at = _missing_collections.get(collection)
    if expires_at is not None:
        if time.time() < expires_at:
            return False
        del _missing_collections[collection]

    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception:  # noqa: BLE001
        # Qdrant is unreachable, which is not the same as "the collection is
        # missing" — so this is not remembered. Caching it would extend an
        # outage past its end.
        return False

    _known_collections.update(existing)
    if collection in existing:
        return True

    _missing_collections[collection] = time.time() + config.CACHE_COLLECTION_NEGATIVE_TTL_SECONDS
    return False


def _point_id_for_question(question: str, context: dict[str, str] | None = None) -> int:
    """Derive a deterministic unsigned-64-bit Qdrant point ID from a question.

    Uses the first 16 hex characters of SHA-256(question), which gives a
    uniform distribution over 2^64 values — collision probability is
    negligible for any realistic cache size.

    *context* is the caller's cache context, and it has to be here. Without it
    two callers in different contexts asking the same question derive the same
    id and upsert over one another: each write clobbers the other's entry, and
    since the contexts differ neither reads the survivor back. Not a leak -- the
    partition still holds -- but both callers miss indefinitely, and a collapsed
    hit rate is harder to attribute than a wrong answer.

    Appended only when non-empty, so an entry written without a context keeps
    the id it had before this existed. That is the same reasoning as the
    signature in `envelope`, and for the same reason: not to invalidate what is
    already stored.
    """
    if context:
        canonical = json.dumps(context, sort_keys=True, separators=(",", ":"))
        question = f"{question}\n{canonical}"
    digest = hashlib.sha256(question.encode()).hexdigest()
    return int(digest[:16], 16)


#: Payload key holding a digest of the entry's cache context. Written by `put()`
#: so `get()` can push the context into the Qdrant query instead of filtering
#: client-side. Reserved, so it is never mistaken for caller context itself --
#: `_RESERVED_PAYLOAD_KEYS` and `envelope`'s copy both carry it, and a test pins
#: them equal.
CONTEXT_DIGEST_KEY = "_ctx"


#: Payload keys the cache writes itself. Everything else in a stored payload
#: came from `put()`'s `payload_extra`, which is the caller's cache context --
#: so the context can be recovered from an entry without a marker field, and
#: entries written before this existed are read correctly.
_RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "question",
        "resolved_question",
        "agent_type",
        "answer",
        "sql",
        "created_at",
        "hit_count",
        "kind",
        CONTEXT_DIGEST_KEY,
        SIGNATURE_KEY,
        VERSION_KEY,
    }
)


#: Point ids already reported as holding the wrong question. The condition is a
#: planted or corrupted entry, which persists -- so without this the warning
#: fires once per request, for as long as the entry survives, each time carrying
#: the asker's question text into the log.
_MISMATCHED_POINTS_LOGGED: set[int] = set()

#: Contexts already reported as unusable, so a caller with an unfortunate key
#: name costs one line rather than one per lookup.
_RESERVED_CONTEXT_KEYS_LOGGED: set[str] = set()


def _context_names_a_reserved_key(context: dict[str, str] | None, where: str) -> bool:
    """True when *context* uses a key the cache writes itself.

    Such a key never survives: `put()` merges `payload_extra` first and the
    reserved literal that follows overwrites it, so the entry is stored with no
    context at all -- readable by every context-free caller, and not readable by
    the caller that asked to be scoped, for the entry's whole life. That is the
    failure this module exists to close, reached by an unlucky key name rather
    than a forgotten call site, and nothing reported it.

    Refused rather than repaired: silently renaming a caller's key would be its
    own surprise, and there is no correct guess. Both sides refuse, so a context
    that cannot be stored also cannot be used to read.
    """
    if not context:
        return False
    clashing = sorted(set(context) & _RESERVED_PAYLOAD_KEYS)
    if not clashing:
        return False
    token = f"{where}:{','.join(clashing)}"
    if token not in _RESERVED_CONTEXT_KEYS_LOGGED:
        _RESERVED_CONTEXT_KEYS_LOGGED.add(token)
        logger.warning(
            "Cache %s refused: the supplied cache_context uses %s, which the cache "
            "writes itself. Such a key is overwritten on write, so the entry would "
            "be stored unscoped and served to callers with no context at all. "
            "Rename it (for example %r).",
            where,
            ", ".join(repr(k) for k in clashing),
            f"ctx_{clashing[0]}",
        )
    return True


#: When each collection was last swept, so the sweep is amortised rather than
#: paid on every write. Process-local: several processes sharing a collection
#: each sweep on their own schedule, which is harmless -- the delete is
#: idempotent and matches by age, not by identity.
#:
#: It starts empty, so the first write in a process always sweeps. In a
#: long-lived server that is what you want. Under the CLI, where a process
#: answers one question and exits, it means one delete-by-filter per invocation.
#:
#: That is deliberate rather than overlooked. Seeding this so the first sweep
#: waits out an interval would mean a CLI-only deployment never sweeps at all --
#: no process lives long enough -- and those are the deployments whose
#: collections still grow. The cost is bounded the other way instead: the delete
#: is issued with `wait=False` so the caller never waits for it, and
#: `created_at` is indexed as a datetime on every collection created from now
#: on, so what Qdrant does with it is a range query rather than a scan. Only
#: collections predating that change pay a scan, and only until they age out.
_last_prune_at: dict[str, float] = {}


def _prune_expired(client: Any, collection: str, ttl_hours: int, *, wait: bool = False) -> bool:
    """Delete points past the TTL. True when a sweep ran and succeeded.

    `bool`, not a count: Qdrant's delete does not report how many points matched,
    so an int here would have to be a lie or a constant. False covers both "not
    due yet" and "the delete raised"; the latter is logged, and no caller needs
    to tell them apart -- both mean the collection did not shrink this time.

    Nothing else removes anything: the TTL is applied on read, and
    `invalidate()` drops the collection wholesale. That was survivable while a
    repeated question upserted over its own point, and stopped being so when the
    point id gained the cache context -- a context that changes per turn writes
    ids that never recur, are never overwritten and were never removed.

    The cutoff is the same `ttl_hours` the read path applies, so this deletes
    only what a read would already have discarded. It cannot remove an entry that
    is still servable.

    Best-effort: a failed sweep is logged and the write continues. A cache that
    grows is worse than one that does not, but not so much worse that it should
    cost the caller their answer.
    """
    interval = config.CACHE_PRUNE_INTERVAL_SECONDS
    if interval <= 0:
        return False

    now = time.time()
    last = _last_prune_at.get(collection)
    if last is not None and now - last < interval:
        return False
    _last_prune_at[collection] = now

    from qdrant_client.models import (  # noqa: PLC0415
        DatetimeRange,
        FieldCondition,
        Filter,
        FilterSelector,
    )

    # A datetime, not its isoformat string: `DatetimeRange` declares
    # `datetime | date | None`, and while pydantic coerces the string, passing
    # the declared type keeps mypy honest and does not depend on that coercion.
    cutoff = datetime.now(UTC) - timedelta(hours=ttl_hours)
    try:
        client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="created_at", range=DatetimeRange(lt=cutoff))]
                )
            ),
            wait=wait,
        )
    except Exception:  # noqa: BLE001 -- a failed sweep must not fail the write
        logger.warning(
            "Cache sweep of %s could not be issued; it will be retried on the "
            "first write after the interval. Whether the delete reached Qdrant "
            "is not knowable from here -- the request is sent without waiting "
            "for it to be applied.",
            collection,
            exc_info=True,
        )
        return False
    return True


def _context_digest(context: dict[str, str] | None) -> str:
    """A stable digest of *context*, for equality-matching inside Qdrant.

    Qdrant's filter can require a field to equal a value; it cannot require the
    *absence* of fields it was not told about, which is why the context equality
    has to be applied client-side as well. Reducing the whole context to one
    field turns that into something the query can express, so the candidates a
    lookup scans are mostly its own rather than everyone else's.

    Empty and absent contexts share a digest, so an unscoped write and an
    unscoped read agree.

    This value is **not** signed: it is derived from the context rather than
    trusted, and `_payload_matches` still compares the real (signed) context
    before an entry is served. Tampering with it can therefore cause a miss, not
    a wrong hit.
    """
    canonical = json.dumps(
        {str(k): str(v) for k, v in (context or {}).items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _context_of(payload: dict[str, Any]) -> dict[str, str]:
    """The cache context an entry was written with, recovered from its payload."""
    return {str(k): str(v) for k, v in payload.items() if k not in _RESERVED_PAYLOAD_KEYS}


def _payload_matches(payload: dict[str, Any], payload_filter: dict[str, str] | None) -> bool:
    """True when *payload* was written in exactly the caller's cache context.

    The comparison is an equality, not a subset test, and that is the point.
    Asking only whether the payload *contains* the caller's keys made a
    context-free read match an entry written under any context: a caller that
    forgot to pass its context on `get()` was served entries scoped by it, while
    the reverse correctly missed. Since the whole value of `cache_context` is
    that it must be supplied on both sides or not built at all, the direction
    that silently succeeded was the dangerous one.

    A caller with no context therefore reads only entries written with none,
    which is what standalone turns already do with each other, and a follow-up
    turn's context-scoped entry is no longer served to a context-free question.
    """
    return _context_of(payload) == {str(k): str(v) for k, v in (payload_filter or {}).items()}


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


#: Raw `CACHE_ANSWER_TIERS` values already reported as unusable, so a
#: misconfiguration costs one line rather than one per cache lookup.
_UNUSABLE_TIER_VALUES_LOGGED: set[str] = set()


# `_COSINE_CANDIDATES` moved to config.CACHE_COSINE_CANDIDATES so an operator
# can raise it; read at call time so tests and deployments can change it.


def _enabled_tiers() -> frozenset[int]:
    """Parse ``CACHE_ANSWER_TIERS`` into the set of tiers allowed to serve.

    Unrecognised entries are ignored rather than raising: this is read on the
    answer path, and a typo in an operator's environment should not turn every
    question into a 500.

    Ignoring them quietly is a different matter. A value like ``all`` or
    ``0;1;2`` parses to nothing, which disables cache reads entirely -- the hit
    rate goes to zero and latency and model cost rise, with no trace anywhere
    except the absence of hits. Whoever wrote ``all`` meant every tier, not
    none, so a non-empty setting that yields no tiers is reported once. An
    explicitly empty value is left silent: that one does say "serve nothing".
    """
    raw = config.CACHE_ANSWER_TIERS
    tiers = set()
    for part in raw.split(","):
        part = part.strip()
        if part in {"0", "1", "2"}:
            tiers.add(int(part))

    if not tiers and raw.strip() and raw not in _UNUSABLE_TIER_VALUES_LOGGED:
        _UNUSABLE_TIER_VALUES_LOGGED.add(raw)
        logger.warning(
            "The semantic cache is serving nothing: NLQ_CACHE_ANSWER_TIERS=%r "
            "names no usable tier. Expected a comma-separated subset of 0, 1, 2 "
            "(for example %r, the default, or %r for exact matches only).",
            raw,
            "0,1,2",
            "0",
        )
    return frozenset(tiers)


def _write_refusal(question: str, result: _PutResult) -> str | None:
    """Why *question*/*result* must not be cached, or ``None`` to store it.

    Three cheap write-side checks. None of them is a boundary -- an attacker who
    can query the agent can still poison it with a short, plausible question, and
    the fix for that is that the blast radius is other users of the same agent,
    who are already authorised to see its answers. These refuse the shapes that
    are never worth storing, which is where a padded injection happens to sit.

    An over-long question is the interesting one. A padded injection is the
    attacker's question plus an instruction suffix long enough to carry the
    payload but short enough that the whole thing still embeds near a question a
    colleague might ask. Length is the cheapest part of that shape to refuse, and
    questions this long are near-unique anyway: they cost a write and are
    unlikely ever to be hit.

    The other two keep the cache from serving something that was never an
    answer. An empty answer is not worth a hit, and an error frame re-served on a
    hit is the failure that has already happened once here -- a prose refusal was
    stored with its SQL and re-executed against a customer's database on every
    subsequent hit.
    """
    answer = (result.answer or "").strip()
    if not answer:
        return "the answer was empty"
    if _looks_like_an_error(answer):
        return "the answer is an error frame"

    limit = config.CACHE_MAX_QUESTION_CHARS
    if limit > 0 and len(question) > limit:
        return f"the question is {len(question)} chars, over the {limit} limit"
    return None


#: Openings of the error text this codebase produces itself. Matched at the
#: start of the answer only: a legitimate answer may well *discuss* an error,
#: and refusing those would quietly stop caching a whole class of question.
_ERROR_PREFIXES: tuple[str, ...] = (
    "error:",
    "failed to",
    "i encountered an error",
    "i'm sorry, i encountered",
    "sorry, i encountered",
    "an error occurred",
    "query failed",
    "unable to answer",
)


def _looks_like_an_error(answer: str) -> bool:
    """Return ``True`` when *answer* is this system reporting its own failure."""
    head = answer.lstrip().lower()
    return head.startswith(_ERROR_PREFIXES)


class SemanticCache:
    """Semantic query cache for a single agent, backed by a Qdrant collection.

    Usage::

        cache = SemanticCache(agent_id="sales-agent", ttl_hours=24)
        entry = cache.get("How many orders last month?")
        if entry is not None:
            return cached_response(entry)
        result = run_orchestrator(question)
        cache.put("How many orders last month?", result)
    """

    def __init__(
        self,
        agent_id: str,
        ttl_hours: int = 24,
        *,
        binding: CacheBinding | None = None,
    ) -> None:
        """*binding* is required to read or write entries, not to manage them.

        An entry carries SQL that is executed on a hit with no model in front of
        it, so it is signed for the context it was produced in and verified
        before use (SEC-09). Without a binding there is nothing to verify
        against: :meth:`get` reports a miss and :meth:`put` stores nothing.

        :meth:`stats`, :meth:`invalidate` and :meth:`list_entries` do not read
        entry contents and work without one.
        """
        safe_id = _SAFE_ID_RE.sub("_", agent_id)
        self._collection = f"{CACHE_COLLECTION_PREFIX}{safe_id}"
        self._ttl_hours = ttl_hours
        self._binding = binding

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _payload_to_entry(self, payload: dict[str, Any]) -> CacheEntry | None:
        """Deserialize a ``CacheEntry`` from *payload*, checking signature and TTL.

        Returns ``None`` when the entry does not verify, has expired, or the
        timestamp is missing or unparseable.

        Does not verify the signature. Verification belongs to the read path
        that feeds execution -- see :meth:`_verified_entry` -- so that
        :meth:`list_entries` can show an operator what is actually stored,
        including entries that do not verify.
        """
        created_at_raw = payload.get("created_at")
        if not created_at_raw:
            return None
        try:
            created_at = datetime.fromisoformat(str(created_at_raw))
        except ValueError:
            return None
        if datetime.now(UTC) - created_at > timedelta(hours=self._ttl_hours):
            return None
        return CacheEntry(
            question=str(payload.get("question", "")),
            resolved_question=str(payload.get("resolved_question", "")),
            agent_type=str(payload.get("agent_type", "")),
            answer=str(payload.get("answer", "")),
            sql=payload.get("sql") or None,
            created_at=created_at,
            hit_count=int(payload.get("hit_count", 0)),
            kind=str(payload.get("kind", "answer")),
        )

    def _verified_entry(self, payload: dict[str, Any]) -> CacheEntry | None:
        """Deserialize *payload* only if it was signed for this cache's binding.

        An entry that does not verify is a miss rather than an error: entries
        written before signing existed are unsigned, and a question that misses
        is answered by generating SQL again.
        """
        if self._binding is None or not verify(payload, self._binding):
            return None
        return self._payload_to_entry(payload)

    def _increment_hit_count(self, client: Any, point_id: Any, payload: dict[str, Any]) -> int:
        """Increment ``hit_count`` in the Qdrant payload and return the new value."""
        new_count = int(payload.get("hit_count", 0)) + 1
        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=self._collection,
                payload={"hit_count": new_count},
                points=[point_id],
            )
        return new_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        question: str,
        *,
        vector: list[float] | None = None,
        payload_filter: dict[str, str] | None = None,
    ) -> CacheEntry | None:
        """Look up *question* in the cache, trying three tiers in order.

        **Tier 0** — exact-match via SHA-256 hash: zero embed calls on hit.
        **Tier 1** — cosine similarity ≥ ``config.CACHE_ANSWER_THRESHOLD`` against
        ``kind=answer`` points; uses *vector* when supplied to avoid a second
        ``embed_text()`` call.
        **Tier 2** — entity-masked cosine similarity ≥ ``config.CACHE_TEMPLATE_THRESHOLD``
        against ``kind=template`` points; entities from *question* are bound
        into the stored parameterized SQL template.

        Returns ``None`` when the collection does not exist, no tier produces a
        valid non-expired hit, or Tier 2 entity binding fails.

        Args:
            question: The user question.
            vector:   Pre-computed embedding vector.  When provided, the
                      ``embed_text()`` call for Tier 1 is skipped.
            payload_filter: The caller's cache context (paired with
                      :meth:`put`'s ``payload_extra``). Matched as an
                      **equality**, not a subset: a hit is returned only when the
                      stored entry was written under exactly this context, and
                      every tier applies it. Passing ``None`` therefore reads
                      only entries written without a context — it does not match
                      everything. Enterprise uses this to scope a follow-up's
                      entry to one conversation context
                      (``{"context_fingerprint": ...}``).

                      The equality is the point: under a subset rule a caller
                      that forgot to pass its context matched entries scoped by
                      every context, failing open in exactly the case the
                      mechanism exists for. See "Cache partitioning and
                      authorisation" in ``docs/architecture.md`` before building
                      anything that relies on this.
        """
        if _context_names_a_reserved_key(payload_filter, "lookup"):
            return None

        client = _get_client()

        # Lazy import to avoid a cycle: the orchestrator package eagerly imports
        # this module, so importing provenance at module top would be circular.
        from nlqueries.orchestrator.provenance import record_cache  # noqa: PLC0415

        if not _collection_exists(client, self._collection):
            return None

        from qdrant_client.models import (  # noqa: PLC0415
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchAny,
            MatchValue,
            PayloadField,
        )

        wanted_digest = _context_digest(payload_filter)

        def _kind_filter(kind: str) -> Filter:
            """`kind`, plus the caller's context expressed as one exact match.

            Matching each context key individually is a *subset* test -- Qdrant
            can require the keys the caller named but cannot require the absence
            of any it did not -- so entries from other contexts came back and had
            to be discarded client-side, consuming the candidate slots a lookup
            scans. One digest field turns the same condition into something the
            query can express exactly.

            A context-free read is the case needing care: entries written before
            this key existed do not carry it at all, and must still be found. So
            that side is a disjunction -- no digest, or the digest of an empty
            context -- which keeps legacy entries readable rather than requiring
            the whole cache to be rewritten.

            `_payload_matches` still runs on what comes back. This filter is an
            optimisation over the real check, not a replacement for it: the
            digest is derived rather than signed, so it can misdirect a lookup
            but cannot get a foreign entry served.
            """
            must: list[Any] = [FieldCondition(key="kind", match=MatchAny(any=[kind]))]
            if payload_filter:
                must.append(
                    FieldCondition(key=CONTEXT_DIGEST_KEY, match=MatchValue(value=wanted_digest))
                )
                return Filter(must=must)
            return Filter(
                must=must,
                should=[
                    IsEmptyCondition(is_empty=PayloadField(key=CONTEXT_DIGEST_KEY)),
                    FieldCondition(key=CONTEXT_DIGEST_KEY, match=MatchValue(value=wanted_digest)),
                ],
            )

        enabled = _enabled_tiers()

        # --- Tier 0: exact-match hash lookup (zero embed calls on hit) ---
        # Guards the lookup only. A disabled tier is skipped, not a return: the
        # tiers are independent, and `"1,2"` has to leave 1 and 2 working.
        if 0 in enabled:
            normalized = _normalize_question(question)
            tier0_id = _point_id_for_question(normalized, payload_filter)
            with contextlib.suppress(Exception):
                tier0_hits = client.retrieve(
                    collection_name=self._collection,
                    ids=[tier0_id],
                    with_payload=True,
                )
                if tier0_hits:
                    payload = tier0_hits[0].payload or {}
                    # The stored question must be the one asked. Tier 0 trusts an
                    # id, and the id is not part of the signed message -- so an
                    # entry can be *relocated* without being forged. Copying a
                    # genuine, correctly signed entry onto the id another question
                    # hashes to made Tier 0 answer that other question with it:
                    # the signature verifies (the payload really is untouched) and
                    # nothing compared the question. Reproduced before this check
                    # existed: "how many refunds were issued" was answered with
                    # "There were 42 orders."
                    stored_question = _normalize_question(str(payload.get("question") or ""))
                    if stored_question != normalized:
                        # Once per point, like the other two warnings in this
                        # module. The entry persists, so an unguarded warning
                        # would fire on every request that lands on it.
                        if tier0_id not in _MISMATCHED_POINTS_LOGGED:
                            _MISMATCHED_POINTS_LOGGED.add(tier0_id)
                            logger.warning(
                                "Cache point %s holds a different question than the "
                                "one it is keyed by. Ignoring it; this should not "
                                "happen unless the collection has been written to "
                                "directly.",
                                tier0_id,
                            )
                        payload = {}
                    # retrieve() is id-only, so enforce payload_filter here (Tier
                    # 1/2 push it into the Qdrant query filter). A mismatch falls
                    # through to the cosine tiers rather than returning a foreign-
                    # context hit.
                    if payload and _payload_matches(payload, payload_filter):
                        entry = self._verified_entry(payload)
                        if entry is not None:
                            entry.hit_count = self._increment_hit_count(client, tier0_id, payload)
                            record_cache(hit=True, tier="exact")  # provenance (SYL-1.1)
                            return entry

        # --- Tier 1: answer cache (cosine similarity, kind=answer filter) ---
        # Tiers 1 and 2 are what let one user's answer reach another user's
        # differently-worded question. That is their purpose and also the only
        # route by which a poisoned entry travels, so an operator can turn them
        # off per deployment without losing Tier 0.
        if 1 in enabled:
            # Skipped entirely when disabled, which also saves the embed call --
            # Tier 2 embeds the *masked* question separately and does not use `v`.
            v = vector if vector is not None else embed_text(question)
            try:
                response = client.query_points(
                    collection_name=self._collection,
                    query=v,
                    query_filter=_kind_filter("answer"),
                    limit=config.CACHE_COSINE_CANDIDATES,
                )
            except Exception as exc:  # noqa: BLE001
                _report_search_failure(self._collection, "tier 1", exc)
                return None

            # The first candidate clearing both the threshold and the context,
            # not simply the nearest. The Qdrant filter is a subset test -- it
            # can require the caller's keys but cannot require the absence of
            # others -- so the equality is applied here, as at Tier 0, and asking
            # for one point would let a neighbour from another context shadow a
            # same-context entry ranked just below it.
            #
            # Exhausting them falls through to Tier 2 rather than returning: the
            # tiers are independent, and this one being occupied by other
            # contexts says nothing about whether a template exists for this one.
            for hit in response.points:
                if hit.score < config.CACHE_ANSWER_THRESHOLD:
                    break  # ranked by score, so nothing below clears it either
                payload = hit.payload or {}
                if not _payload_matches(payload, payload_filter):
                    continue
                entry = self._verified_entry(payload)
                if entry is not None:
                    entry.hit_count = self._increment_hit_count(client, hit.id, payload)
                    record_cache(hit=True, similarity=float(hit.score), tier="answer")  # SYL-1.1
                    return entry

        # --- Tier 2: template cache (masked cosine similarity, kind=template) ---
        if 2 not in enabled:
            return None
        masked = _mask_entities(question)
        if masked == question:
            # No entities found — template would be identical to answer; skip.
            return None

        try:
            masked_vector = embed_text(masked)
            tmpl_response = client.query_points(
                collection_name=self._collection,
                query=masked_vector,
                query_filter=_kind_filter("template"),
                limit=config.CACHE_COSINE_CANDIDATES,
            )
        except Exception as exc:  # noqa: BLE001
            _report_search_failure(self._collection, "tier 2", exc)
            return None

        # The nearest template belonging to this context, for the same reason as
        # Tier 1: the context equality is applied here rather than pushed into
        # the query, so asking for one point lets another context's template
        # shadow ours.
        # The binding dialect, so values are rendered the way the engine that
        # will run them reads them. Without it sqlglot writes its own dialect and
        # MySQL's backslash escaping in particular is lost.
        dialect = self._binding.dialect if self._binding is not None else None

        # Every check inside the loop, as Tier 1 does, so a candidate that fails
        # one moves to the next rather than ending the lookup. Selecting a
        # candidate first and validating afterwards meant an expired or
        # unverifiable template -- or one whose entities would not bind --
        # returned a miss even with a usable template for a different phrasing
        # ranked just below it and above the threshold. Expired points are never
        # deleted and go on being ranked by the search, so that is not a rare
        # shape; it is the one that accumulates.
        entry = None
        tmpl_score = 0.0
        bound_sql = ""
        for candidate in tmpl_response.points:
            if candidate.score < config.CACHE_TEMPLATE_THRESHOLD:
                break  # ranked by score, so nothing below clears it either
            payload = candidate.payload or {}
            if not _payload_matches(payload, payload_filter):
                continue
            # Verified before its SQL is touched, for two reasons. Expired
            # templates cluster at the front of the ranking -- they are never
            # deleted until the sweep runs -- so binding first spends a full
            # sqlglot parse on each one before discarding it. And it would mean
            # parsing SQL out of a payload whose signature has not been checked,
            # which is the wrong order to do those two things in.
            candidate_entry = self._verified_entry(payload)
            if candidate_entry is None:
                continue  # expired or unverifiable: try the next candidate

            template_sql = str(payload.get("sql") or "")
            if not template_sql:
                continue

            candidate_sql = _bind_entities(question, template_sql, dialect)
            if candidate_sql is None:
                continue

            # A real check, not a suppressed one. This used to parse the bound
            # SQL inside `contextlib.suppress(Exception)` and discard the result,
            # so a statement that did not parse was handed to the orchestrator
            # anyway -- which is how a template hit could return a stale answer
            # alongside "Cached SQL failed revalidation and was not executed".
            # `_bind_entities` already parses; this is the assertion that it did.
            if _parse_or_none(candidate_sql, dialect) is None:
                continue

            entry = candidate_entry
            tmpl_score = float(candidate.score)
            bound_sql = candidate_sql
            break

        if entry is None:
            return None

        entry.question = question
        entry.sql = bound_sql
        entry.hit_count = 0
        entry.kind = "template"
        record_cache(hit=True, similarity=tmpl_score, tier="template")  # SYL-1.1
        return entry

    def put(
        self,
        question: str,
        result: _PutResult,
        *,
        payload_extra: dict[str, str] | None = None,
    ) -> None:
        """Embed *question* and upsert the answer into the cache collection.

        Stores a ``kind="answer"`` point keyed on the **normalized** question so
        that minor variations (capitalisation, punctuation) map to the same entry.

        For SQL results that contain entity literals in the question, also stores
        a ``kind="template"`` point keyed on the masked question embedding, with
        the SQL parameterized via placeholder substitution.  Both points are
        upserted in a single Qdrant call.

        Args:
            question: The (resolved) question string used as the cache key.
            result:   Any object exposing ``resolved_question``, ``agent_type``,
                      ``answer``, and ``sql`` attributes.
            payload_extra: Optional extra string keys merged into every stored
                      point's payload, to be matched later by :meth:`get`'s
                      ``payload_filter``. Reserved payload keys (``question``,
                      ``sql``, ``kind``, ``created_at``, …) take precedence and
                      cannot be overwritten. ``None`` (default) stores exactly as
                      before.
        """
        from qdrant_client.models import PointStruct  # noqa: PLC0415

        if _context_names_a_reserved_key(payload_extra, "write"):
            return

        # Refused before the collection is touched and before the question is
        # embedded, so a rejected write costs nothing.
        refusal = _write_refusal(question, result)
        if refusal is not None:
            logger.debug("Cache write skipped: %s.", refusal)
            return

        # `created_at` is indexed as a datetime because the sweep filters on it
        # with `DatetimeRange`. A keyword index would not accelerate that, and
        # unindexed means Qdrant scans the collection -- on exactly the
        # collections large enough for the sweep to matter.
        ensure_collection(
            self._collection,
            CACHE_VECTOR_SIZE,
            payload_indexes=["kind"],
            datetime_indexes=["created_at"],
        )
        # It exists as of now, so stop remembering that it did not — otherwise
        # the very process that just created it would keep reporting a miss for
        # up to the negative TTL.
        _missing_collections.pop(self._collection, None)
        _known_collections.add(self._collection)

        # Answer entry (Tier 0 exact-match + Tier 1 cosine)
        normalized = _normalize_question(question)
        answer_id = _point_id_for_question(normalized, payload_extra)
        answer_vector = embed_text(question)

        if self._binding is None:
            # Nothing to sign for, so nothing is stored. An unsigned entry would
            # not verify on read and would occupy the collection for its TTL.
            logger.debug("Cache write skipped: no binding was supplied.")
            return

        answer_payload: dict[str, Any] = {
            **(payload_extra or {}),
            CONTEXT_DIGEST_KEY: _context_digest(payload_extra),
            "question": question,
            "resolved_question": result.resolved_question,
            "agent_type": result.agent_type,
            "answer": result.answer,
            "sql": result.sql,
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": "answer",
        }

        answer_payload = sign(answer_payload, self._binding)

        points: list[PointStruct] = [
            PointStruct(id=answer_id, vector=answer_vector, payload=answer_payload)
        ]

        # Template entry (Tier 2) — only for SQL results with entity literals
        if result.agent_type == "sql" and result.sql:
            masked = _mask_entities(question)
            if masked != question:  # entities were found
                try:
                    from nlqueries.processing.parameterizer import (  # noqa: PLC0415
                        _parameterize_sql,
                    )

                    template_sql, placeholders = _parameterize_sql(
                        result.sql, {}, skip_string_literals=True
                    )
                    if placeholders:  # only store if SQL has literal parameters
                        masked_vector = embed_text(masked)
                        tmpl_id = _point_id_for_question(f"tmpl:{masked}", payload_extra)
                        tmpl_payload: dict[str, Any] = {
                            **(payload_extra or {}),
                            CONTEXT_DIGEST_KEY: _context_digest(payload_extra),
                            "question": masked,
                            "resolved_question": result.resolved_question,
                            "agent_type": result.agent_type,
                            "answer": result.answer,
                            "sql": template_sql,
                            "created_at": datetime.now(UTC).isoformat(),
                            "hit_count": 0,
                            "kind": "template",
                        }
                        points.append(
                            PointStruct(
                                id=tmpl_id,
                                vector=masked_vector,
                                payload=sign(tmpl_payload, self._binding),
                            )
                        )
                except Exception:  # noqa: BLE001
                    pass  # template storage is best-effort

        client = _get_client()
        client.upsert(collection_name=self._collection, points=points)

        # After the upsert, so a failed sweep cannot cost the caller the write
        # that prompted it. At most once per collection per interval, so the cost
        # is amortised rather than paid on every put.
        _prune_expired(client, self._collection, self._ttl_hours)

    def invalidate(self, agent_id: str) -> None:  # noqa: ARG002
        """Delete all points in the cache collection (full invalidation).

        The *agent_id* parameter is accepted for API symmetry with CLI callers
        but is not used — the collection to clear is always ``self._collection``.
        """
        _known_collections.discard(self._collection)
        with contextlib.suppress(Exception):
            _get_client().delete_collection(self._collection)

        # After the delete, not before. The record of which payload indexes this
        # collection has must go with the collection -- otherwise the next put()
        # recreates it and skips every index as already done, leaving it without
        # the `kind` keyword index or the `created_at` datetime index the sweep
        # ranges over, for the life of the process.
        #
        # Clearing first left a window: a concurrent put() reaching
        # `ensure_collection` between the two could re-add the keys just before
        # the collection was deleted, reinstating exactly the stale record this
        # is here to remove. Clearing afterwards inverts the race into a harmless
        # one -- a concurrent recreation costs one redundant
        # `create_payload_index`, and creating an index that already exists
        # succeeds rather than raising (measured on v1.9.3 and v1.18.2).
        forget_collection_indexes(self._collection)

    def list_entries(self, limit: int = 100) -> list[CacheEntry]:
        """Return up to *limit* cached entries, ordered by creation time descending.

        Uses Qdrant's ``scroll`` API to page through all points without
        requiring a query vector.  Returns an empty list if the collection
        does not exist or Qdrant is unreachable.
        """
        client = _get_client()
        try:
            records, _ = client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # noqa: BLE001
            return []

        entries: list[CacheEntry] = []
        for record in records:
            payload = record.payload or {}
            entry = self._payload_to_entry(payload)
            if entry is not None:
                entries.append(entry)

        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def stats(self) -> dict[str, Any]:
        """Return basic statistics for this agent's cache collection.

        Returns:
            ``{"total_entries": N, "collection": str}``
        """
        client = _get_client()
        try:
            info = client.get_collection(self._collection)
            total: int = info.points_count or 0
        except Exception:  # noqa: BLE001
            total = 0
        return {"total_entries": total, "collection": self._collection}
