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
from nlqueries.cache.envelope import CacheBinding, sign, verify
from nlqueries.embeddings.embedder import embed_text
from nlqueries.embeddings.qdrant_store import ensure_collection

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


def _point_id_for_question(question: str) -> int:
    """Derive a deterministic unsigned-64-bit Qdrant point ID from a question.

    Uses the first 16 hex characters of SHA-256(question), which gives a
    uniform distribution over 2^64 values — collision probability is
    negligible for any realistic cache size.
    """
    digest = hashlib.sha256(question.encode()).hexdigest()
    return int(digest[:16], 16)


def _payload_matches(payload: dict[str, Any], payload_filter: dict[str, str] | None) -> bool:
    """True when *payload* contains every key of *payload_filter* with an equal value.

    Used to scope a Tier 0 (exact-id) hit by the caller's ``payload_filter`` —
    Tier 1/2 push the same constraint down into the Qdrant query filter instead.
    An empty/``None`` filter matches everything (default behaviour unchanged).
    """
    if not payload_filter:
        return True
    return all(str(payload.get(k)) == str(v) for k, v in payload_filter.items())


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


#: Raw `CACHE_ANSWER_TIERS` values already reported as unusable, so a
#: misconfiguration costs one line rather than one per cache lookup.
_UNUSABLE_TIER_VALUES_LOGGED: set[str] = set()


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
            payload_filter: Optional exact-match constraints on stored payload
                      keys (paired with :meth:`put`'s ``payload_extra``). A hit is
                      only returned when the stored point carries every key/value
                      here — every tier applies it. Enterprise uses this to scope
                      a follow-up's cache entry to one conversation context
                      (``{"context_fingerprint": ...}``); ``None`` (default)
                      leaves lookups exactly as before.
        """
        client = _get_client()

        # Lazy import to avoid a cycle: the orchestrator package eagerly imports
        # this module, so importing provenance at module top would be circular.
        from nlqueries.orchestrator.provenance import record_cache  # noqa: PLC0415

        if not _collection_exists(client, self._collection):
            return None

        from qdrant_client.models import (  # noqa: PLC0415
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        def _kind_filter(kind: str) -> Filter:
            """kind== plus every payload_filter key as an exact-match must-condition."""
            must: list[Any] = [FieldCondition(key="kind", match=MatchAny(any=[kind]))]
            if payload_filter:
                must += [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in payload_filter.items()
                ]
            return Filter(must=must)

        enabled = _enabled_tiers()

        # --- Tier 0: exact-match hash lookup (zero embed calls on hit) ---
        # Guards the lookup only. A disabled tier is skipped, not a return: the
        # tiers are independent, and `"1,2"` has to leave 1 and 2 working.
        if 0 in enabled:
            tier0_id = _point_id_for_question(_normalize_question(question))
            with contextlib.suppress(Exception):
                tier0_hits = client.retrieve(
                    collection_name=self._collection,
                    ids=[tier0_id],
                    with_payload=True,
                )
                if tier0_hits:
                    payload = tier0_hits[0].payload or {}
                    # retrieve() is id-only, so enforce payload_filter here (Tier
                    # 1/2 push it into the Qdrant query filter). A mismatch falls
                    # through to the cosine tiers rather than returning a foreign-
                    # context hit.
                    if _payload_matches(payload, payload_filter):
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
                    limit=1,
                )
            except Exception:  # noqa: BLE001
                return None

            if response.points:
                hit = response.points[0]
                if hit.score >= config.CACHE_ANSWER_THRESHOLD:
                    payload = hit.payload or {}
                    entry = self._verified_entry(payload)
                    if entry is not None:
                        entry.hit_count = self._increment_hit_count(client, hit.id, payload)
                        record_cache(
                            hit=True, similarity=float(hit.score), tier="answer"
                        )  # SYL-1.1
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
                limit=1,
            )
        except Exception:  # noqa: BLE001
            return None

        if not tmpl_response.points:
            return None

        tmpl_hit = tmpl_response.points[0]
        if tmpl_hit.score < config.CACHE_TEMPLATE_THRESHOLD:
            return None

        tmpl_payload = tmpl_hit.payload or {}
        template_sql = str(tmpl_payload.get("sql") or "")
        if not template_sql:
            return None

        # The binding dialect, so values are rendered the way the engine that
        # will run them reads them. Without it sqlglot writes its own dialect and
        # MySQL's backslash escaping in particular is lost.
        dialect = self._binding.dialect if self._binding is not None else None
        bound_sql = _bind_entities(question, template_sql, dialect)
        if bound_sql is None:
            return None

        # A real check, not a suppressed one. This used to parse the bound SQL
        # inside `contextlib.suppress(Exception)` and discard the result, so a
        # statement that did not parse was handed to the orchestrator anyway --
        # which is how a template hit could return a stale answer alongside
        # "Cached SQL failed revalidation and was not executed". `_bind_entities`
        # already parses; this is the assertion that it did.
        if _parse_or_none(bound_sql, dialect) is None:
            return None

        entry = self._verified_entry(tmpl_payload)
        if entry is None:
            return None

        entry.question = question
        entry.sql = bound_sql
        entry.hit_count = 0
        entry.kind = "template"
        record_cache(hit=True, similarity=float(tmpl_hit.score), tier="template")  # SYL-1.1
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

        # Refused before the collection is touched and before the question is
        # embedded, so a rejected write costs nothing.
        refusal = _write_refusal(question, result)
        if refusal is not None:
            logger.debug("Cache write skipped: %s.", refusal)
            return

        ensure_collection(self._collection, CACHE_VECTOR_SIZE, payload_indexes=["kind"])
        # It exists as of now, so stop remembering that it did not — otherwise
        # the very process that just created it would keep reporting a miss for
        # up to the negative TTL.
        _missing_collections.pop(self._collection, None)
        _known_collections.add(self._collection)

        # Answer entry (Tier 0 exact-match + Tier 1 cosine)
        normalized = _normalize_question(question)
        answer_id = _point_id_for_question(normalized)
        answer_vector = embed_text(question)

        if self._binding is None:
            # Nothing to sign for, so nothing is stored. An unsigned entry would
            # not verify on read and would occupy the collection for its TTL.
            logger.debug("Cache write skipped: no binding was supplied.")
            return

        answer_payload: dict[str, Any] = {
            **(payload_extra or {}),
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
                        tmpl_id = _point_id_for_question(f"tmpl:{masked}")
                        tmpl_payload: dict[str, Any] = {
                            **(payload_extra or {}),
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

    def invalidate(self, agent_id: str) -> None:  # noqa: ARG002
        """Delete all points in the cache collection (full invalidation).

        The *agent_id* parameter is accepted for API symmetry with CLI callers
        but is not used — the collection to clear is always ``self._collection``.
        """
        _known_collections.discard(self._collection)
        with contextlib.suppress(Exception):
            _get_client().delete_collection(self._collection)

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
