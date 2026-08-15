"""
nlqueries.orchestrator.prompt_assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Builds prompts from a YAML knowledge base.

Public API
----------
``AssembledPrompt``
    Dataclass splitting the prompt into a cacheable static block and a
    per-question dynamic block (see P2/P3 in performance-improvement-plan.md).

``assemble_prompt(question, knowledge_base, top_k_capsules, *, collection)``
    Returns an :class:`AssembledPrompt`.  The ``static_system`` field is
    byte-identical across requests for the same agent (enabling Anthropic
    prompt caching).  Dynamic retrieval results go into ``dynamic_context``.
    Runs its Qdrant searches sequentially; use this from sync call sites
    (CLI, tests).

``assemble_prompt_async(...)``
    Concurrent variant of ``assemble_prompt`` for async call sites (Phase 6C).
    Identical output, but the schema-hint, verified-example, and capsule
    searches run via ``asyncio.gather`` instead of one after another, so
    per-request latency is roughly the slowest single search rather than
    the sum of all three.

``assemble_prompt_with_history(...)``
    Multi-turn variant; returns ``(static_system_blocks, user_prompt, prior_messages)``.

``assemble_document_prompt(question, retrieval_result)``
    Builds ``(system_prompt, user_prompt)`` for document Q&A.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nlqueries.knowledge.concept_hierarchy import GlossaryHierarchy, build_glossary_hierarchy
from nlqueries.orchestrator.provenance import record_capsule, record_prompt_section

if TYPE_CHECKING:
    from nlqueries.orchestrator.document_retrieval import DocumentRetrievalResult

# ---------------------------------------------------------------------------
# SQL format instructions shared by static_system and the single-call path
# (P3: the LLM is instructed to emit reasoning then <sql>...</sql>).
# ---------------------------------------------------------------------------

_ROLE_PREAMBLE = (
    "You are a SQL generation assistant for a natural-language query interface.\n"
    "Use the schema and example queries below to translate the user's question into valid SQL."
)

_SQL_FORMAT_RULES = """\
## Instructions
- Generate only a single SELECT SQL statement.
- Use only tables and columns present in the schema above.
- First, briefly explain your reasoning in 2-4 sentences (plain text).
- Then output the SQL between EXACTLY these markers — nothing before or after:

<sql>
SELECT ...
</sql>"""


# ---------------------------------------------------------------------------
# AssembledPrompt
# ---------------------------------------------------------------------------


@dataclass
class AssembledPrompt:
    """Prompt split into a stable cacheable block and a per-question block.

    ``static_system``
        Role instructions + FULL schema (all tables, deterministic order) +
        business context + SQL format rules.  Byte-identical across questions
        for the same agent KB — suitable for Anthropic prompt caching.

    ``dynamic_context``
        Retrieved capsules + relevant-table hint.  Changes per question.

    ``user_question``
        The (possibly resolved) user question.
    """

    static_system: str
    dynamic_context: str
    user_question: str

    def system_blocks(self, *, cache: bool = True) -> list[dict[str, Any]]:
        """Return the system parameter as a list of typed blocks.

        When *cache* is True and the static block is non-empty, attach
        ``cache_control: {"type": "ephemeral"}`` so Anthropic's prompt-cache
        API stores the prefix for up to 5 minutes.
        """
        blocks: list[dict[str, Any]] = []
        if self.static_system:
            block: dict[str, Any] = {"type": "text", "text": self.static_system}
            if cache:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        if self.dynamic_context:
            blocks.append({"type": "text", "text": self.dynamic_context})
        return blocks

    def user_content(self) -> str:
        """Return the user turn content — just the question.

        Dynamic context (capsules, table hints) lives in the second system
        block returned by ``system_blocks()``; it must not be duplicated here.
        """
        return self.user_question


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def _append_extra_context(dynamic_context: str, extra: str | None) -> str:
    """Append caller-supplied *extra* guidance to the dynamic-context block.

    Extra context (e.g. an enterprise Nexus join-paths / metric section) is
    appended to the *dynamic* block, never the cached static prefix, so it never
    busts the Anthropic prompt cache on the schema. ``None``/empty is a no-op.
    """
    if not extra:
        return dynamic_context
    return f"{dynamic_context}\n\n{extra}" if dynamic_context else extra


def _append_question_glossary(
    dynamic_context: str, question: str, knowledge_base: dict[str, Any]
) -> str:
    """Append the question-scoped glossary block when the flag is on (CG-2.2).

    A no-op unless ``GLOSSARY_QUESTION_SCOPED`` is set (default off), in which
    case the glossary was omitted from the static block and is rendered here,
    filtered to the terms this question is about plus their hierarchy neighbours.
    """
    from nlqueries import config as _cfg  # noqa: PLC0415

    if not _cfg.GLOSSARY_QUESTION_SCOPED:
        return dynamic_context
    section = _build_question_glossary_section(question, knowledge_base)
    return _append_extra_context(dynamic_context, section)


def assemble_prompt(
    question: str,
    knowledge_base: dict[str, Any],
    top_k_capsules: int = 5,
    *,
    collection: str | None = None,
    vector: list[float] | None = None,
    extra_dynamic_context: str | None = None,
) -> AssembledPrompt:
    """Assemble an :class:`AssembledPrompt` for *question*.

    The ``static_system`` block is built entirely from the knowledge-base dict
    (deterministic — all tables in KB order, no Qdrant calls) so it remains
    byte-identical across requests for the same agent, enabling Anthropic
    prompt caching.

    The ``dynamic_context`` block contains per-question Qdrant results:
    a relevant-tables hint (comma list of top-10 table names) and the top-k
    most similar query capsules.

    The question is embedded **once** and the pre-computed vector is reused
    for both Qdrant searches.

    Args:
        question:       The user's natural-language question.
        knowledge_base: Parsed YAML KB dict.
        top_k_capsules: Number of example capsules in the dynamic block.
        collection:     Optional Qdrant collection name for semantic ranking.
        extra_dynamic_context: Optional caller-supplied guidance appended to the
                        dynamic block (e.g. an enterprise Nexus join-paths /
                        Beacons section). Core stays agnostic to its content.

    Returns:
        :class:`AssembledPrompt`
    """
    static_system = _build_static_system(knowledge_base)

    # Use caller-supplied vector when available (avoids a redundant embed_text
    # call when the multi-agent orchestrator has already embedded the question
    # for the semantic cache lookup).  Fall back to a fresh embed otherwise.
    question_vector: list[float] | None = vector
    if collection and question_vector is None:
        try:
            from nlqueries.embeddings.embedder import embed_text

            question_vector = embed_text(question)
        except Exception:  # noqa: BLE001
            pass

    dynamic_context = _build_dynamic_context(
        question, knowledge_base, top_k_capsules, collection, question_vector
    )
    dynamic_context = _append_question_glossary(dynamic_context, question, knowledge_base)
    dynamic_context = _append_extra_context(dynamic_context, extra_dynamic_context)

    return AssembledPrompt(
        static_system=static_system,
        dynamic_context=dynamic_context,
        user_question=question,
    )


async def assemble_prompt_async(
    question: str,
    knowledge_base: dict[str, Any],
    top_k_capsules: int = 5,
    *,
    collection: str | None = None,
    vector: list[float] | None = None,
    extra_dynamic_context: str | None = None,
) -> AssembledPrompt:
    """Concurrent variant of :func:`assemble_prompt` (Phase 6C).

    Produces identical output to :func:`assemble_prompt`, but the embedding
    call (when *vector* is not supplied) and the three dynamic-context Qdrant
    searches are offloaded to threads and run via ``asyncio.gather`` instead
    of sequentially — so a request no longer blocks the event loop for the
    sum of all Qdrant round trips, only for the slowest one.

    Use this from async call sites (e.g.
    :class:`~nlqueries.orchestrator.orchestrator.Orchestrator`); use the sync
    :func:`assemble_prompt` from CLI / non-async paths.

    Args:
        question:       The user's natural-language question.
        knowledge_base: Parsed YAML KB dict.
        top_k_capsules: Number of example capsules in the dynamic block.
        collection:     Optional Qdrant collection name for semantic ranking.
        vector:         Pre-computed question embedding, reused instead of
                        embedding again.
        extra_dynamic_context: Optional caller-supplied guidance appended to the
                        dynamic block (e.g. an enterprise Nexus join-paths /
                        Beacons section). Core stays agnostic to its content.

    Returns:
        :class:`AssembledPrompt`
    """
    static_system = _build_static_system(knowledge_base)

    question_vector: list[float] | None = vector
    if collection and question_vector is None:
        try:
            from nlqueries.embeddings.embedder import embed_text

            question_vector = await asyncio.to_thread(embed_text, question)
        except Exception:  # noqa: BLE001
            pass

    dynamic_context = await _build_dynamic_context_async(
        question, knowledge_base, top_k_capsules, collection, question_vector
    )
    dynamic_context = _append_question_glossary(dynamic_context, question, knowledge_base)
    dynamic_context = _append_extra_context(dynamic_context, extra_dynamic_context)

    return AssembledPrompt(
        static_system=static_system,
        dynamic_context=dynamic_context,
        user_question=question,
    )


# ---------------------------------------------------------------------------
# Static-system builder (byte-identical per agent KB)
# ---------------------------------------------------------------------------


def _render_m_schema(knowledge_base: dict[str, Any]) -> str:
    """Render the KB as a compact M-Schema string (Phase 6B).

    Format::

        【DB_ID】 sales
        【Table】 orders — one row per order
        (order_id:BIGINT, PK), (customer_id:BIGINT, FK->customers.customer_id),
        (status:TEXT, samples: ['pending', 'shipped', 'cancelled']), ...
        【Foreign keys】
        orders.customer_id = customers.customer_id

    Falls back gracefully when KB lacks v2 fields (``is_primary_key``,
    ``is_foreign_key``, ``references``, ``samples``) — columns are rendered
    as ``(name:TYPE)`` without flags.
    """
    db_name = knowledge_base.get("db_name", "database")
    tables: list[dict[str, Any]] = knowledge_base.get("schema", {}).get("tables", [])
    foreign_keys: list[dict[str, str]] = knowledge_base.get("schema", {}).get("foreign_keys", [])

    if not tables:
        return ""

    lines: list[str] = [f"【DB_ID】 {db_name}", ""]

    for table in tables:
        name = table.get("name", "")
        desc = table.get("description", "")
        header = f"【Table】 {name}"
        if desc:
            header += f" — {desc}"
        lines.append(header)
        if name:
            record_prompt_section(f"table_desc:{name}")  # provenance (SYL-1.1)

        col_parts: list[str] = []
        for col in table.get("columns", []):
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            flags: list[str] = []

            if col.get("is_primary_key"):
                flags.append("PK")
            if col.get("is_foreign_key") and col.get("references"):
                flags.append(f"FK->{col['references']}")

            samples: list[str] = col.get("samples", [])
            if samples:
                sample_str = ", ".join(f"'{s}'" for s in samples[:5])
                flags.append(f"samples: [{sample_str}]")

            inner = f"{col_name}:{col_type}"
            if flags:
                inner += f", {', '.join(flags)}"
            col_parts.append(f"({inner})")

        if col_parts:
            lines.append(", ".join(col_parts))
        lines.append("")

    if foreign_keys:
        lines.append("【Foreign keys】")
        for fk in foreign_keys:
            lines.append(f"{fk['from']} = {fk['to']}")
        lines.append("")

    return "\n".join(lines)


def _build_static_system(knowledge_base: dict[str, Any]) -> str:
    """Build the cacheable static system prompt from *knowledge_base*.

    Contains: role preamble + ALL schema tables (deterministic KB order) +
    business context + SQL format rules.  No question-dependent content.

    Schema format is controlled by ``config.SCHEMA_FORMAT``:
    - ``"compact"`` (default): M-Schema 【Table】 format — fewer tokens.
    - ``"verbose"``: Full markdown ``### Table:`` format — backward compatible.
    """
    from nlqueries import config as _cfg  # noqa: PLC0415

    parts: list[str] = [_ROLE_PREAMBLE, ""]

    if _cfg.SCHEMA_FORMAT == "verbose":
        schema_section = _build_full_schema_section(knowledge_base)
    else:
        schema_section = _render_m_schema(knowledge_base)

    if schema_section:
        parts.append(schema_section)
    # In question-scoped mode (CG-2.2) the glossary moves to the per-question
    # dynamic block; keep rules here. Off by default → full glossary stays static.
    business_section = _build_business_context_section(
        knowledge_base, include_glossary=not _cfg.GLOSSARY_QUESTION_SCOPED
    )
    if business_section:
        parts.append(business_section)
    parts.append(_SQL_FORMAT_RULES)
    return "\n".join(parts)


def _build_full_schema_section(knowledge_base: dict[str, Any]) -> str:
    """Render ALL tables from the KB in deterministic (KB YAML) order."""
    tables: list[dict[str, Any]] = knowledge_base.get("schema", {}).get("tables", [])
    if not tables:
        return ""

    lines: list[str] = ["## Database Schema", ""]
    for table in tables:
        name = table.get("name", "")
        desc = table.get("description", "")
        lines.append(f"### Table: {name}")
        if name:
            record_prompt_section(f"table_desc:{name}")  # provenance (SYL-1.1)
        if desc:
            lines.append(f"Description: {desc}")
        row_count = table.get("row_count")
        if row_count is not None:
            lines.append(f"Rows: {row_count:,}")
        lines.append("Columns:")
        for col in table.get("columns", []):
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            col_desc = col.get("description", "")
            if col_desc:
                lines.append(f"  - {col_name} ({col_type}): {col_desc}")
            else:
                lines.append(f"  - {col_name} ({col_type})")
        lines.append("")
    return "\n".join(lines)


def _entry_term(entry: Any) -> str | None:
    """The term name of a glossary entry (dict with ``term`` or a bare string)."""
    if isinstance(entry, dict):
        term = str(entry.get("term", "")).strip()
        return term or None
    if isinstance(entry, str):
        return entry.strip() or None
    return None


def _render_glossary_line(
    entry: Any, hierarchy: GlossaryHierarchy, *, provenance_prefix: str = "glossary"
) -> str:
    """One glossary bullet, with parent context, recording a provenance tag."""
    if isinstance(entry, dict):
        term = str(entry.get("term", ""))
        parent = hierarchy.parent(term) if term else None
        label = f"{term} (a kind of {parent})" if parent else term
        if term:
            record_prompt_section(f"{provenance_prefix}:{term}")  # provenance (SYL-1.1)
        return f"- {label}: {entry.get('definition', '')}"
    return f"- {entry}"


def _build_business_context_section(
    knowledge_base: dict[str, Any], *, include_glossary: bool = True
) -> str:
    """Return a formatted Markdown section for glossary and business rules.

    When *include_glossary* is False (question-scoped mode, CG-2.2), the glossary
    is omitted here and rendered per-question in the dynamic block instead; rules
    are always included.
    """
    biz: dict[str, Any] = knowledge_base.get("business_context", {})
    glossary: list[Any] = biz.get("glossary", []) if include_glossary else []
    rules: list[Any] = biz.get("rules", [])
    if not glossary and not rules:
        return ""

    lines: list[str] = ["## Business Context", ""]
    if glossary:
        # Cycle-safe: a malformed hierarchy renders flat, never raises at query time.
        hierarchy = build_glossary_hierarchy(glossary, validate=False)
        lines.append("### Glossary")
        for entry in glossary:
            lines.append(_render_glossary_line(entry, hierarchy))
        lines.append("")
    if rules:
        lines.append("### Business Rules")
        for rule in rules:
            lines.append(f"- {rule}")
            record_prompt_section(f"rule:{str(rule)[:60]}")  # provenance (SYL-1.1)
        lines.append("")
    return "\n".join(lines)


def _term_in_question(term: str, question_lower: str) -> bool:
    """True when *term* appears in the (lower-cased) question on word boundaries."""
    if not term:
        return False
    return re.search(rf"\b{re.escape(term.lower())}\b", question_lower) is not None


def _build_question_glossary_section(question: str, knowledge_base: dict[str, Any]) -> str:
    """Question-scoped glossary block (CG-2.2).

    Selects glossary terms the question mentions, then expands with their
    hierarchy **ancestors** (broader context) and **descendants** (narrower
    candidates, depth 3). Directly-matched terms record ``glossary:<term>``
    provenance; terms pulled in via the hierarchy record ``glossary_hier:<term>``
    so the explanation tree can distinguish them. Returns "" when the question
    mentions no glossary term (nothing relevant to inject).
    """
    biz: dict[str, Any] = knowledge_base.get("business_context", {})
    glossary: list[Any] = biz.get("glossary", [])
    if not glossary:
        return ""

    hierarchy = build_glossary_hierarchy(glossary, validate=False)
    known_terms = {t for t in (_entry_term(e) for e in glossary) if t}
    question_lower = question.lower()
    matched = {t for t in known_terms if _term_in_question(t, question_lower)}
    if not matched:
        return ""

    # term -> directly matched? (hierarchy-surfaced terms default to False)
    selected: dict[str, bool] = {t: True for t in matched}
    for t in matched:
        for rel in (*hierarchy.ancestors(t), *hierarchy.descendants(t, max_depth=3)):
            selected.setdefault(rel, False)

    lines: list[str] = ["## Business Context (relevant to your question)", "", "### Glossary"]
    for entry in glossary:  # keep KB order for stability
        term = _entry_term(entry)
        if term is None or term not in selected:
            continue
        prefix = "glossary" if selected[term] else "glossary_hier"
        lines.append(_render_glossary_line(entry, hierarchy, provenance_prefix=prefix))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic-context builder (per-question)
# ---------------------------------------------------------------------------


def _format_dynamic_context(
    knowledge_base: dict[str, Any],
    top_k: int,
    hit_names: list[str],
    verified_hits: list[dict[str, Any]],
    capsule_hits: list[Any],
) -> str:
    """Render the dynamic-context text from already-fetched search results.

    Pure formatting, no I/O — shared by the sequential
    (:func:`_build_dynamic_context`) and concurrent
    (:func:`_build_dynamic_context_async`) search paths so the two can never
    drift apart in output.
    """
    parts: list[str] = []

    if hit_names:
        parts.append("## Most relevant tables for this question\n" + ", ".join(hit_names))

    if verified_hits or capsule_hits:
        cap_lines: list[str] = [
            "## Example Queries",
            "",
            "The following queries represent common patterns in this database:",
            "",
        ]
        idx = 1
        for v in verified_hits:
            q = v.get("question", "")
            sql = v.get("sql", "")
            if not q and not sql:
                continue
            cap_lines.append(f"{idx}. Intent: {q}")
            if sql:
                cap_lines.append(
                    f"   SQL Template: {sql}  -- verified example (user-confirmed correct)"
                )
            cap_lines.append("")
            idx += 1
        for cap in capsule_hits:
            intent = cap.intent or cap.auto_description
            template = cap.template_sql
            if not intent and not template:
                continue
            cap_lines.append(f"{idx}. Intent: {intent}")
            if template:
                cap_lines.append(f"   SQL Template: {template}")
            cap_lines.append("")
            idx += 1
            # provenance (SYL-1.1): capsules carry no persisted id in the Qdrant
            # payload, so the intent/description label is the stable handle we have.
            record_capsule(intent or template[:60], intent)
        parts.append("\n".join(cap_lines))

    if not parts:
        # Fall back to raw KB capsules when Qdrant is unavailable
        raw_capsules: list[dict[str, Any]] = knowledge_base.get("query_capsules", [])
        if raw_capsules:
            cap_lines = [
                "## Example Queries",
                "",
                "The following queries represent common patterns in this database:",
                "",
            ]
            for i, cap in enumerate(raw_capsules[:top_k], 1):
                intent = cap.get("intent", "")
                template = cap.get("template", "")
                if not intent and not template:
                    continue
                cap_lines.append(f"{i}. Intent: {intent}")
                if template:
                    cap_lines.append(f"   SQL Template: {template}")
                cap_lines.append("")
                record_capsule(intent or str(template)[:60], intent)  # provenance (SYL-1.1)
            parts.append("\n".join(cap_lines))

    return "\n\n".join(parts)


def _build_dynamic_context(
    question: str,
    knowledge_base: dict[str, Any],
    top_k: int,
    collection: str | None,
    question_vector: list[float] | None,
) -> str:
    """Build the per-question dynamic context block.

    Contains a relevant-tables hint (if Qdrant available), verified examples
    from user-confirmed feedback (Phase 5B), and top-k capsule examples.

    Verified examples are searched first and consume slots from *top_k* so the
    total number of examples presented to the LLM stays bounded. Searches run
    sequentially — use :func:`_build_dynamic_context_async` from async call
    sites for concurrent searches.
    """
    hit_names: list[str] = []
    verified_hits: list[dict[str, Any]] = []
    capsule_hits: list[Any] = []

    if collection:
        # Relevant-tables hint (short list, not full schema)
        try:
            from nlqueries.embeddings.qdrant_store import search_schema

            hits = search_schema(collection, question, top_k=10, vector=question_vector)
            hit_names = sorted({str(h["table_name"]) for h in hits if h.get("table_name")})
        except Exception:  # noqa: BLE001
            pass

        # Phase 5B: Verified examples (user-confirmed correct SQL, threshold 0.75)
        verified_hits = _search_verified(collection, question, top_k=3, vector=question_vector)
        remaining_slots = top_k - len(verified_hits)

        # Capsule examples (fill remaining slots after verified ones)
        try:
            from nlqueries.embeddings.qdrant_store import search

            capsule_hits = search(
                collection, question, top_k=max(1, remaining_slots), vector=question_vector
            )
        except Exception:  # noqa: BLE001
            pass

    return _format_dynamic_context(knowledge_base, top_k, hit_names, verified_hits, capsule_hits)


async def _build_dynamic_context_async(
    question: str,
    knowledge_base: dict[str, Any],
    top_k: int,
    collection: str | None,
    question_vector: list[float] | None,
) -> str:
    """Concurrent variant of :func:`_build_dynamic_context` (Phase 6C).

    Runs the schema-hint, verified-example, and capsule searches as three
    independent Qdrant round trips via ``asyncio.gather`` (each offloaded to
    a thread, since the Qdrant client is synchronous) instead of one after
    another. The capsule search always requests *top_k* results rather than
    ``top_k - len(verified_hits)`` — since verified-hit count isn't known
    until after the gather completes — and the excess is trimmed afterward;
    this is equivalent because results are returned in relevance order, so
    requesting more and truncating yields the same prefix as requesting the
    smaller count directly.
    """
    hit_names: list[str] = []
    verified_hits: list[dict[str, Any]] = []
    capsule_hits: list[Any] = []

    if collection:
        from nlqueries.embeddings.qdrant_store import search, search_schema

        schema_result, verified_result, capsule_result = await asyncio.gather(
            asyncio.to_thread(
                search_schema, collection, question, top_k=10, vector=question_vector
            ),
            asyncio.to_thread(
                _search_verified, collection, question, top_k=3, vector=question_vector
            ),
            asyncio.to_thread(search, collection, question, top_k=top_k, vector=question_vector),
            return_exceptions=True,
        )

        if not isinstance(schema_result, BaseException):
            hit_names = sorted({str(h["table_name"]) for h in schema_result if h.get("table_name")})
        if not isinstance(verified_result, BaseException):
            verified_hits = verified_result
        if not isinstance(capsule_result, BaseException):
            remaining_slots = max(1, top_k - len(verified_hits))
            capsule_hits = capsule_result[:remaining_slots]

    return _format_dynamic_context(knowledge_base, top_k, hit_names, verified_hits, capsule_hits)


def _search_verified(
    collection: str,
    question: str,
    top_k: int = 3,
    *,
    vector: list[float] | None = None,
    threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Search the ``agent_{id}_verified`` collection for similar confirmed examples.

    Derives the verified collection name from *collection* (expected to be
    ``agent_{id}_schema``; verified collection is ``agent_{id}_verified``).

    Returns a list of payload dicts for hits above *threshold*, or ``[]`` when
    the collection does not exist or Qdrant is unreachable.
    """
    # Derive the verified collection name from the schema collection name.
    # schema collection:   agent_{safe_id}_schema
    # verified collection: agent_{safe_id}_verified
    if not collection.endswith("_schema"):
        return []
    verified_collection = collection[: -len("_schema")] + "_verified"

    try:
        from qdrant_client import QdrantClient  # noqa: PLC0415
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        from nlqueries import config as _cfg  # noqa: PLC0415

        client = QdrantClient(url=_cfg.QDRANT_URL, api_key=_cfg.QDRANT_API_KEY or None)
        existing = {c.name for c in client.get_collections().collections}
        if verified_collection not in existing:
            return []

        if vector is None:
            from nlqueries.embeddings.embedder import embed_text  # noqa: PLC0415

            vector = embed_text(question)

        response = client.query_points(
            collection_name=verified_collection,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="verified", match=MatchValue(value=True))]
            ),
            limit=top_k,
        )
        return [hit.payload for hit in response.points if hit.score >= threshold and hit.payload]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Multi-turn prompt assembly with conversation history (Task 18.1)
# ---------------------------------------------------------------------------


def assemble_prompt_with_history(
    question: str,
    knowledge_base: dict[str, Any],
    history: list[dict[str, str]],
    top_k_capsules: int = 5,
) -> tuple[list[dict[str, Any]], str, list[dict[str, str]]]:
    """Assemble a (system_blocks, user_prompt, prior_messages) triple.

    Returns the system as a list of typed blocks (with cache_control on the
    static block) ready to pass to the Anthropic API.  ``prior_messages`` is
    the ``history`` list passed through unchanged.

    Args:
        question:       The current user question.
        knowledge_base: Parsed YAML KB dict.
        history:        Prior turns from ConversationSession.to_prompt_messages().
        top_k_capsules: Number of capsules to embed.

    Returns:
        ``(system_blocks, current_user_content, prior_messages_for_api)``
    """
    prompt = assemble_prompt(question, knowledge_base, top_k_capsules)
    return prompt.system_blocks(), prompt.user_content(), history


# ---------------------------------------------------------------------------
# Document Q&A prompt assembly (Task 11.1) — unchanged
# ---------------------------------------------------------------------------


def assemble_document_prompt(
    question: str,
    retrieval_result: DocumentRetrievalResult,
) -> tuple[str, str]:
    """Build ``(system_prompt, user_prompt)`` for document Q&A.

    The system prompt instructs the LLM to answer using only the provided
    document chunks and to cite sources inline as ``[source_name, page N]``
    (or ``[source_name]`` when no page number is available).

    Args:
        question:         The user's natural-language question.
        retrieval_result: Result from ``retrieve_for_question``.

    Returns:
        ``(system_prompt, user_prompt)`` — both are plain strings.
    """
    system_prompt = (
        "You are a document question-answering assistant.\n"
        "Answer the user's question using ONLY the document excerpts provided below.\n"
        "You MUST cite sources inline using the format [source_name, page N] when a page "
        "number is available, or [source_name] when it is not.\n"
        "If the answer cannot be found in the provided excerpts, say so explicitly — "
        "do not fabricate information.\n"
        "Do not reference any knowledge outside the provided document context."
    )

    context_lines: list[str] = ["## Document Context", ""]
    for i, (chunk, citation) in enumerate(
        zip(retrieval_result.chunks, retrieval_result.citations, strict=True),
        start=1,
    ):
        if citation.page_number is not None:
            label = f"[{citation.source_name}, page {citation.page_number}]"
        else:
            label = f"[{citation.source_name}]"
        context_lines.append(f"### Excerpt {i} — {label}")
        context_lines.append(chunk.text)
        context_lines.append("")

    user_prompt = f"{question}\n\n" + "\n".join(context_lines)
    return system_prompt, user_prompt
