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

#: What the answer looks like: which columns, in what order, and the idioms a
#: correct result is expected in. Each rule comes from re-running a text-to-SQL
#: benchmark's misses (2026-10-09), where the statement ran and was defensible
#: but answered in another shape. The largest classes were extra or missing
#: columns (a fifth of the misses), COUNT(DISTINCT) where the joined rows were
#: meant, a MAX/MIN subquery where ORDER BY ... LIMIT 1 was, and a hint's
#: literal not used as written.
#:
#: Measured, not assumed. A rule against filters nobody asked for (IS NOT NULL)
#: lost twice as many correct answers as it gained and was taken out; the
#: YES/NO rule is limited to questions that offer the choice, because applied
#: to every "is it ...?" it replaced the value the data holds with 'YES'.
#:
#: The rule on joins that repeat rows is for a reporting question's commonest
#: silent error: a total over one table, joined to another that has several
#: rows per key, is multiplied by the join and still looks plausible.
#:
#: The join-type rule: an inner join is the rule and an outer join the listed
#: exception. Written as a choice between the two, the same wording went one way
#: in one question and the other way in the next, and an outer join added
#: zero-count rows nobody asked for.
#:
#: The display-column rule: an answer of ids is right and useless to the person who
#: asked which customer it was. Instead of the id, not beside it.
#:
#: The column-order clause: with columns already to be returned in the order asked
#: for, an id the question named later still came back first, the usual shape of
#: a key leading its row. Its net effect is not yet measured; the re-run after
#: these rule changes is that measurement.
#:
#: The key-column rule: a key is added only to a figure given per entity when
#: nothing the question names tells the entities apart, and that key once, with
#: no other id. Added anywhere else, an id crowded out the columns asked for, or a
#: url came back in place of the name the question wanted. "Once" counts keys, not
#: columns: half of a two-column key groups distinct entities together. The first
#: rule names this as its one exception, so the two are not weighed afresh each time.
#:
#: The definitions rule: a user who spells out how a metric is worked out wants
#: that calculation, not the textbook one the term usually names.
#:
#: The coded-values rule: a code returned as a word, or cast to a number, is not
#: the value stored ('65+' cast to a number is 65, and no longer equals '65+'). A
#: mapping that explains a code says which code to filter on, not that the output
#: should be relabelled.
#:
#: The numeric-text rules: a text column holding numbers is ordered by its cast
#: but returned as stored, since a converted copy is not the value the person
#: asked for. Stripping a decoration (a currency sign, a percent sign, a unit) is
#: asked for only where the knowledge base's sample values show one on that
#: column (see :func:`answer_rules_for`); stated for every column, it rewrote
#: plain values and built parsers for dates written as free text.
#:
#: The running-total rule: a column that already holds a running total (a
#: standings table's points, a balance, cumulative_*) is summed to a figure many
#: times too large. It is named only where a column's name says so. The latest row
#: is the total; MAX equals it only where the total never falls, which a balance
#: can, and a total across entities still adds up each one's latest value.
#:
#: The correlated-subquery rule: a statement that is right but, as some databases
#: run it, repeats a subquery for every row of a large table does not finish within
#: a request's timeout.
#:
#: One block, kept short: it sits in the prompt-cached Instructions and costs
#: tokens on every call. Shared with `sql_generation`, whose prompt also drives
#: the repair step, so a repaired statement cannot undo it.
_ANSWER_RULES = """\
- SELECT only the columns the question asks for, in the order the question or its \
hint lists them: "street, city, state and zip" is four columns in that order. A key or \
id column stays where the question names it, never moved to the front: "the city, the \
store id and the opening date" puts the id second. Do not add the column you sorted, \
filtered or ranked by, an id, or a name, label or count column for context; the one \
exception, an entity's key for a figure given per entity, is set out below.
- A field, count, total, average or rank the question names is a requested column: \
"the top five cities and their number of stores" returns the city and the count.
- "How many", or "what is the total" or "the average", with no breakdown asked for \
returns a single number with no grouping column.
- Add an entity's key only when the question asks for a figure for each entity ("how \
many orders each customer placed") and names no column for it, such as a name, title, \
label or description ("in each city" names the city column); then add that entity's \
key once (every column of it, where it has more than one) and no other id. Never add \
an id beside a name the question asked for, and never return a url or a code in place \
of a name it asked for: "who" asks for the name.
- A person's full name is the first name and the last name, as two columns in that \
order. "The name" means one set of name columns, not every numbered variant (Name1, \
Name2, Name3).
- When the question asks which customer, product or other thing, return its name or \
title column instead of the id, or a description where it has neither, unless the \
question asks for the id.
- Count the rows the joins produce with COUNT(column). Use COUNT(DISTINCT ...) only \
when the question or hint says distinct, unique or different.
- When a join repeats the rows whose values you total or count (a customer's invoices \
joined to the same customer's shipments repeat each invoice once per shipment), total \
or count them per join key in a subquery first, then join the result.
- Join with an inner join. A bare "for each X, how many Y" or "the total Y per X" is an \
inner join too: an X with no Y is left out. Use LEFT JOIN only when the question says \
"including those with none", "zero", "at most", "fewer than", "no" or "without", asks \
about the X that have no Y ("never ordered", "missing a manager"), or asks for every \
member of a dimension table whether or not it has a match; a bare "for each X" is not \
such a question unless it says the unmatched ones are wanted. There a row with no match \
counts as 0 where a count is asked for. If unsure, use an inner join.
- When the question orders, compares or adds a text column that holds numbers, use its \
cast to a numeric type the dialect accepts (REAL in SQLite, FLOAT64 in BigQuery): as \
text, '9' sorts above '10'. Return the column as stored: for its largest value, ORDER BY \
the cast DESC LIMIT 1, not MAX(CAST(...)), since a converted copy is not the value asked \
for. A plain cast reads only digits and a decimal point: where the values shown carry \
anything else, remove it inside the cast (a decimal comma becomes a point, not removed: \
'1,5' is 1.5), or the cast gives 0 or the leading digits on \
some databases and fails on others. When the question only asks for the value, do not \
cast it.
- Do not parse dates written as free text (month names, mixed formats) with CASE ladders \
or chains of SUBSTR. Unless the knowledge base documents the format, order by the column \
as stored, and where you explain your reasoning, say so.
- Prefer joins, GROUP BY and window functions to a subquery that refers to the outer \
row, which some databases run once per row of the outer query: on a large table that \
can take minutes.
- For "the highest", "lowest", "most", "least", "earliest" or "latest", use ORDER BY \
... LIMIT 1, not a comparison with a MAX or MIN subquery, unless the question asks \
for every tie.
- A quoted value in the hint is exact: use it as written, in the column the hint names.
- Coded values ('Y', 'N', '+', '-', '65+', a status code) stay exactly as stored, in \
SELECT and in WHERE: do not relabel them (no CASE turning 'Y' into a word) and do not \
CAST a coded column ('65+' is an age band, not the number 65; a column listed below \
as holding numbers written with a decoration is not coded). A mapping the question, \
the hint or a document gives to explain a code ("returned refers to status = 'R'") \
tells you which code to filter on; it is not a request to relabel the output. Relabel \
only when the question asks for the relabelled output itself, as a "yes or no?" \
question does.
- A column the hint names for something in the question is the column to use for it \
("the cheapest refers to MIN(price)" means order by price), even where another \
column seems to hold the same thing in another form.
- A definition given in the question, the Business Rules or the Glossary (how a \
segment, a rate or a period is worked out) is exact: apply every condition and \
boundary as written, not the usual meaning of the term.
- Only when the question itself offers the choice, as in "yes or no?" or "true or \
false?", answer with the text 'YES' or 'NO', or 'True' or 'False', not a boolean \
expression. A question that only asks whether something holds ("is the order \
shipped?") returns the value the data stores for it."""

#: A number with at most a short prefix, a short suffix ('%', ' min', '/yr') and
#: separators inside its digits ('1,200', '1.234,56'). A prefix is a currency sign
#: ('$', '€') or up to four letters set off by a space or a period ('I 2008',
#: 'No. 5'): letters joined to the digits ('A1', 'SKU12', 'INV-1001') make a code,
#: not a decorated number. So does a bare '+' or '-' after the digits ('65+' is
#: an age band, '10-' a grade): the coded-values rule keeps those as stored, and
#: listing them here would tell the model to strip and cast them. Times and dates
#: do not match: their separators sit between digits outside the number.
_DECORATED_NUMBER = re.compile(
    r"^\s*(?P<pre>[$€£¥₹]|[A-Za-z]{1,4}(?:\.\s?|\s))?\s?"
    r"(?P<num>[+-]?(?:\d{1,3}(?:[,.\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?))"
    r"\s?(?P<post>(?![+-]\s*$)[^\d\s][^\d]{0,7})?\s*$"
)

#: Most columns a knowledge-base rule names; a longer list is cut and says so, so
#: a schema of text columns does not fill the cached Instructions with names.
_MAX_RULE_COLUMNS = 10

#: Text column types, as the knowledge base records them.
_TEXT_MARKERS = ("CHAR", "TEXT", "STRING", "CLOB")


def _decorated_number_columns(
    knowledge_base: dict[str, Any],
) -> list[tuple[str, str, list[str]]]:
    """``(table, column, examples)`` for each text column whose sample values are
    numbers written with one fixed decoration.

    Every sample must parse as a number with a decoration around or within it
    (see :data:`_DECORATED_NUMBER`), at least one must carry one, and the
    decorations must agree: one prefix and one suffix at most across the
    samples. So '45%', '7%' and '$1,200', '$300' count; codes such as 'A1',
    'A2' or 'INV-1001', and plain numbers stored as text, do not. Two samples
    at least, since one is not evidence of a pattern.
    """
    found: list[tuple[str, str, list[str]]] = []
    for table in knowledge_base.get("schema", {}).get("tables", []) or []:
        for column in table.get("columns", []) or []:
            col_type = str(column.get("type") or "").upper()
            if col_type and not any(marker in col_type for marker in _TEXT_MARKERS):
                continue
            samples = [str(v) for v in column.get("samples") or [] if str(v).strip()]
            if len(samples) < 2:
                continue
            parsed = [m for m in map(_DECORATED_NUMBER.match, samples) if m is not None]
            if len(parsed) < len(samples):
                continue
            prefixes = {m["pre"].strip().lower() for m in parsed if m["pre"]}
            suffixes = {m["post"].strip().lower() for m in parsed if m["post"]}
            separated = any(re.search(r"[,\s]|\..*\.", m["num"]) for m in parsed)
            if len(prefixes) > 1 or len(suffixes) > 1 or not (prefixes or suffixes or separated):
                continue
            name = _table_ref(table) or str(table.get("name"))
            found.append((name, str(column.get("name")), samples))
    return found


#: Column names that say the column holds a running total wherever it appears.
#: "running" and "balance" alone are not enough: running_time is a duration
#: and balance_due or minimum_balance an amount on each row, all of which add
#: up. So running_ needs a total after it, and a balance is the bare word or
#: one of the forms a ledger carries forward.
_RUNNING_TOTAL_NAME = re.compile(
    r"^(?:cumulative|cum|ytd)_"
    r"|_(?:cumulative|to_date|ytd)$"
    r"|(?:^|_)running_(?:total|sum|balance|count)(?:$|_)"
    r"|^(?:opening_|closing_|account_)?balance(?:_after)?$",
    re.IGNORECASE,
)
#: Column names that hold a running total in a standings table: a team's wins or
#: points so far in the season, restated on every row.
_STANDINGS_TOTALS = frozenset({"wins", "losses", "draws", "points", "pts"})


def _running_total_columns(knowledge_base: dict[str, Any]) -> list[tuple[str, str]]:
    """``(table, column)`` for each column whose name says it holds a running
    total: ``cumulative_*``, ``cum_*``, ``ytd_*``, ``*_ytd``, ``*_to_date``,
    ``running_total``, ``running_sum``, ``running_balance`` or ``running_count``,
    a balance carried forward (``balance``, ``opening_balance``,
    ``closing_balance``, ``account_balance``, ``balance_after``), or wins,
    losses, draws or points in a table named as standings. Names only: sample
    values cannot show that a column accumulates."""
    found: list[tuple[str, str]] = []
    for table in knowledge_base.get("schema", {}).get("tables", []) or []:
        table_name = str(table.get("name") or "")
        standings = "standing" in table_name.lower()
        for column in table.get("columns", []) or []:
            name = str(column.get("name") or "")
            if _RUNNING_TOTAL_NAME.search(name) or (
                standings and name.lower() in _STANDINGS_TOTALS
            ):
                found.append((_table_ref(table) or table_name, name))
    return found


def answer_rules_for(knowledge_base: dict[str, Any]) -> str:
    """The answer rules, with those this knowledge base's evidence calls for.

    :data:`_ANSWER_RULES` holds for every database. A rule that depends on what
    a column holds is added only where the knowledge base shows it, so it does
    not fire on columns it was never about. Depends on the knowledge base
    alone, so the cached block stays identical across questions.
    """
    extra: list[str] = []
    decorated = _decorated_number_columns(knowledge_base)
    if decorated:
        listed = "; ".join(
            f"{table}.{column} ({', '.join(repr(v[:20]) for v in examples[:2])})"
            for table, column, examples in decorated[:_MAX_RULE_COLUMNS]
        )
        if len(decorated) > _MAX_RULE_COLUMNS:
            listed += f"; and {len(decorated) - _MAX_RULE_COLUMNS} more like them, not listed"
        extra.append(
            "- These text columns hold numbers written with a decoration: "
            f"{listed}. Only when the question orders, compares or adds one of them, "
            "strip the decoration inside the cast to a numeric type "
            "(CAST(REPLACE(column, '%', '') AS REAL) in SQLite), "
            "keeping the decimal mark as a point: where the values use a decimal comma, it "
            "becomes a point rather than being removed ('1,5' is 1.5, not 15), while a "
            "thousands separator is removed ('1,200' is 1200); a plain CAST fails on some "
            "databases and on "
            "others keeps only the leading digits or gives 0. When the question just asks for "
            "the value, return the column as stored."
        )
    running = _running_total_columns(knowledge_base)
    if running:
        listed = ", ".join(f"{table}.{column}" for table, column in running[:_MAX_RULE_COLUMNS])
        if len(running) > _MAX_RULE_COLUMNS:
            listed += f", and {len(running) - _MAX_RULE_COLUMNS} more like them, not listed"
        extra.append(
            f"- These columns hold running totals: {listed}. Each row holds the total so far, "
            "so an entity's total is the value on its latest row, not the SUM of its rows (MAX "
            "gives the same for a total that never falls, such as wins or points, but not for "
            "a balance); a total across entities adds up each one's latest value. Where you "
            "explain your reasoning, say which you used."
        )
    return "\n".join([_ANSWER_RULES, *extra])


def _sql_format_rules(answer_rules: str) -> str:
    """The Instructions block, around *answer_rules*."""
    return f"""\
## Instructions
- Generate only a single SELECT SQL statement.
- Use only tables and columns present in the schema above.
{answer_rules}
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


#: Appended to a table whose KB entry sets ``columns_omitted``.
#:
#: A knowledge base may list only some of a table's columns — a deployment can
#: restrict which of them a given agent may read. The model cannot tell a
#: partial list from a complete one, so it writes ``SELECT *`` meaning
#: "everything about this table" and either receives columns the KB never showed
#: it or has the statement refused by whatever enforced the restriction. Saying
#: so costs a few tokens and avoids both.
_PARTIAL_COLUMNS_NOTE = (
    "only the columns listed are available for this table; select them by name, never with *"
)


def _columns_omitted(table: dict[str, Any]) -> bool:
    """Whether *table*'s KB entry says its column list is incomplete."""
    return bool(table.get("columns_omitted"))


#: Longest column description the compact schema renders; a longer one is cut
#: at a word boundary. A description says what an abbreviated or ambiguous name
#: cannot, but a paragraph for every column would crowd out the schema itself.
_MAX_COLUMN_DESCRIPTION_CHARS = 160


def _column_description(col: dict[str, Any]) -> str:
    """*col*'s description on one line for the compact schema, or ``""`` when it
    has none or only repeats the column's name ("points" for ``points``)."""
    text = " ".join(str(col.get("description") or "").split())
    if not text or _letters_and_digits(text) == _letters_and_digits(str(col.get("name", ""))):
        return ""
    if len(text) > _MAX_COLUMN_DESCRIPTION_CHARS:
        cut = text[:_MAX_COLUMN_DESCRIPTION_CHARS].rsplit(" ", 1)[0]
        text = cut.rstrip(" ,;:") + "..."
    return text


def _letters_and_digits(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _render_m_schema(knowledge_base: dict[str, Any]) -> str:
    """Render the KB as a compact M-Schema string (Phase 6B).

    Format::

        【DB_ID】 sales
        【Table】 orders — one row per order
        (order_id:BIGINT, PK), (customer_id:BIGINT, FK->customers.customer_id),
        (st:TEXT, order status, samples: ['pending', 'shipped', 'cancelled']), ...
        【Foreign keys】
        orders.customer_id = customers.customer_id

    A column's description, when the KB has one that says more than the
    column's name, follows its keys and precedes its values (see
    :func:`_column_description`). Without it the model has only the name to go
    on, and a name such as ``CRE`` or ``position`` (in two tables) does not say
    which column a question means.

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
        header = f"【Table】 {_table_ref(table)}"
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
            description = _column_description(col)
            if description:
                flags.append(description)

            samples: list[str] = col.get("samples", [])
            if samples:
                # "values" when they are every value the column holds, so the
                # model can rely on the spelling; "samples" when they are a few.
                if col.get("values_complete"):
                    value_str = ", ".join(f"'{s}'" for s in samples)
                    flags.append(f"values: [{value_str}]")
                else:
                    sample_str = ", ".join(f"'{s}'" for s in samples[:5])
                    flags.append(f"samples: [{sample_str}]")

            inner = f"{col_name}:{col_type}"
            if flags:
                inner += f", {', '.join(flags)}"
            col_parts.append(f"({inner})")

        if col_parts:
            lines.append(", ".join(col_parts))
        if _columns_omitted(table):
            # `Note:` prefixed, not parenthesised. M-Schema renders columns as
            # parenthesised tuples -- `(id:BIGINT)` -- so a bare `(...)` line
            # here is not syntactically distinguishable from a further column
            # entry, and the whole value of this line rests on the model reading
            # it as prose rather than as schema.
            lines.append(f"Note: {_PARTIAL_COLUMNS_NOTE}.")
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
    parts.append(_sql_format_rules(answer_rules_for(knowledge_base)))
    return "\n".join(parts)


def _table_ref(table: dict[str, Any]) -> str:
    """How a table should be NAMED to the model: ``schema.table``, or bare.

    Table HEADERS only. The foreign-key block and the per-column ``FK->``
    flags still read bare, and that is a known gap rather than an oversight:
    `kb_generator` builds `foreign_keys` from `col.references`, which the
    connectors report unqualified, so there is nothing here to qualify them
    WITH. A model mirroring one of those lines into a join therefore loses the
    schema, and on two same-named tables in different schemas the block is
    ambiguous. Closing it means resolving reference targets back to their
    tables at generation time, which is a change to the knowledge base rather
    than to this renderer.

    The knowledge base used to record only the bare name, so a prompt could not
    tell the model where a table lived and generated SQL referred to something
    the server could not resolve -- invisible on a connection whose default
    schema already held everything, and a hard failure anywhere else.

    Every connector reports a schema -- `TableSpec.schema` is a required `str`
    -- so the bare fallback is for knowledge bases written before this field
    existed and for hand-edited entries, not for any connector. A regenerated
    SQLite base renders `main.orders`, which resolves.
    """
    name = str(table.get("name", ""))
    schema = str(table.get("schema") or "").strip()
    return f"{schema}.{name}" if schema and name else name


def _build_full_schema_section(knowledge_base: dict[str, Any]) -> str:
    """Render ALL tables from the KB in deterministic (KB YAML) order."""
    tables: list[dict[str, Any]] = knowledge_base.get("schema", {}).get("tables", [])
    if not tables:
        return ""

    lines: list[str] = ["## Database Schema", ""]
    for table in tables:
        name = table.get("name", "")
        desc = table.get("description", "")
        lines.append(f"### Table: {_table_ref(table)}")
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
        # Both renderers carry the note, because `SCHEMA_FORMAT` chooses between
        # them at runtime: marking only the default would leave every `verbose`
        # deployment with the restriction unstated and no way to tell.
        if _columns_omitted(table):
            lines.append(f"Note: {_PARTIAL_COLUMNS_NOTE}.")
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
        # Qualified through the same helper as the schema block.
        # `upsert_schema` writes `table_name` into the Qdrant payload
        # unqualified, so without this a single prompt could offer
        # `sales.orders` in the schema block and `orders` in the hint list
        # directly above it -- two names for one table, in the place a
        # reader of `_table_ref` is most likely to assume was covered.
        #
        # A hit naming a table the KB does not hold keeps its bare name:
        # better a name the model can still match than one invented here.
        # Qualified only when the bare name maps to exactly ONE table.
        #
        # A knowledge base holding both `public.orders` and `sales.orders` is
        # the ordinary multi-schema case, and the Qdrant payload carries no
        # schema -- so there is nothing here to say which was meant. Naming one
        # would point the model at a definite schema that may be the wrong one,
        # and a query against the wrong table returns a plausible answer rather
        # than an error. The bare name leaves the schema block to disambiguate,
        # which is what it did before this.
        by_name: dict[str, list[dict[str, Any]]] = {}
        for t in knowledge_base.get("schema", {}).get("tables", []):
            by_name.setdefault(str(t.get("name", "")), []).append(t)
        qualified = [
            _table_ref(by_name[n][0]) if len(by_name.get(n, ())) == 1 else n for n in hit_names
        ]
        parts.append("## Most relevant tables for this question\n" + ", ".join(qualified))

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
