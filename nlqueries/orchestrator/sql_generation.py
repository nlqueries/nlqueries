"""
nlqueries.orchestrator.sql_generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM-driven SQL generation with sqlglot AST validation.

Public API
----------
``SQLGenerationResult``
    Dataclass carrying the generated SQL, validity flag, and metadata.

``validate_and_repair(sql, knowledge_base, dialect, llm, system)``
    Validate extracted SQL and repair it if needed (mechanical first, then
    LLM repair reusing the prompt-cached system prefix).  Async.

``generate_sql(question, knowledge_base, dialect)``
    Legacy sync entry point that drives two LLM calls itself.  Retained for
    CLI paths and tests; new code should use ``validate_and_repair`` instead.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp

from nlqueries.llm import get_llm_client

if TYPE_CHECKING:
    from nlqueries.llm.client import LLMClient


@dataclass
class SQLGenerationResult:
    """Result of a SQL generation or validation attempt.

    Attributes:
        sql:              The SQL statement (may be invalid if ``is_valid`` is
                          ``False``).
        is_valid:         ``True`` when the statement passed all validation
                          checks.
        validation_error: Human-readable error message, or ``None`` when
                          valid.
        dialect:          The SQL dialect used for generation and validation.
        attempt_count:    ``1`` when the first attempt succeeded; ``2`` after
                          a retry.
    """

    sql: str
    is_valid: bool
    validation_error: str | None
    dialect: str
    attempt_count: int


# ---------------------------------------------------------------------------
# Phase 3B: validate_and_repair (async)
# ---------------------------------------------------------------------------


async def validate_and_repair(
    sql: str,
    knowledge_base: dict[str, Any],
    dialect: str,
    llm: LLMClient,
    system: str | list[dict[str, Any]] | None = None,
    connector: Any = None,
    explain_check: bool = False,
) -> SQLGenerationResult:
    """Validate *sql* extracted from a streaming response; repair if invalid.

    Repair order (cheapest first):

    1. **Mechanical repair**: sqlglot transpile + case-insensitive table-name
       correction + multi-statement stripping — no LLM call consumed on success.
    2. **LLM repair**: one ``acomplete()`` call with the same cached system
       prefix (*system*) so Anthropic prompt-cache tokens are reused.
    3. **EXPLAIN gate** (optional): runs ``EXPLAIN <sql>`` via *connector*
       when *explain_check* is ``True``; marks the result invalid on failure.
       Only executed when the SQL has already passed static validation.

    Args:
        sql:            Extracted SQL text (may be empty on fallback).
        knowledge_base: Parsed YAML KB dict for schema validation.
        dialect:        SQL dialect (``"postgres"``, ``"snowflake"``, …).
        llm:            LLM client instance (for the repair call).
        system:         System prompt passed to the repair ``acomplete()``
                        call.  When ``None``, a minimal fallback system
                        string is used (prompt-cache miss, but correct).
        connector:      Optional DB connector with an ``execute(sql)`` method.
                        Required when *explain_check* is ``True``; ignored
                        otherwise.
        explain_check:  When ``True`` (and *connector* is not ``None``), run
                        ``EXPLAIN`` on the final SQL to catch plan-time errors.
                        Defaults to ``False``.

    Returns:
        :class:`SQLGenerationResult`.
    """
    error = _validate_sql(sql, knowledge_base, dialect)
    if error is None:
        result = SQLGenerationResult(
            sql=sql,
            is_valid=True,
            validation_error=None,
            dialect=dialect,
            attempt_count=1,
        )
        return await _apply_explain_gate(result, connector, explain_check, dialect)

    # --- Mechanical repair (no LLM) ------------------------------------------
    repaired, mech_error = _try_mechanical_repair(sql, knowledge_base, dialect)
    if mech_error is None:
        result = SQLGenerationResult(
            sql=repaired,
            is_valid=True,
            validation_error=None,
            dialect=dialect,
            attempt_count=1,
        )
        return await _apply_explain_gate(result, connector, explain_check, dialect)

    # --- LLM repair (reuses cached system prefix) ----------------------------
    if system is None:
        system = _build_sql_system_prompt(knowledge_base, dialect)

    correction_user = (
        "Your SQL had a validation error and needs correction.\n\n"
        f"Error: {error}\n"
        f"SQL with error:\n{sql}\n\n"
        f"Please generate a corrected {dialect} SELECT statement. "
        "Wrap the SQL in <sql>...</sql> markers."
    )

    # --- Phase 6A: self-consistency for hard queries -----------------------
    from nlqueries import config as _cfg  # noqa: PLC0415
    from nlqueries.orchestrator.candidates import (  # noqa: PLC0415
        _is_hard,
        generate_candidates,
        select_best,
    )

    sc_mode = _cfg.SELF_CONSISTENCY
    use_consistency = sc_mode == "all" or (
        sc_mode == "hard" and _is_hard(correction_user, knowledge_base)
    )

    if use_consistency:
        candidates = await generate_candidates(llm, system, correction_user, n=3)
        if candidates:
            repaired_sql = select_best(candidates, knowledge_base, dialect)
        else:
            raw = await llm.acomplete(system, correction_user, max_tokens=512)
            repaired_sql = _extract_sql(raw)
    else:
        raw = await llm.acomplete(system, correction_user, max_tokens=512)
        repaired_sql = _extract_sql(raw)

    repair_error = _validate_sql(repaired_sql, knowledge_base, dialect)
    result = SQLGenerationResult(
        sql=repaired_sql,
        is_valid=repair_error is None,
        validation_error=repair_error,
        dialect=dialect,
        attempt_count=2,
    )
    return await _apply_explain_gate(result, connector, explain_check, dialect)


# ---------------------------------------------------------------------------
# Legacy sync entry point (CLI, existing tests)
# ---------------------------------------------------------------------------


def generate_sql(
    question: str,
    knowledge_base: dict[str, Any],
    dialect: str,
) -> SQLGenerationResult:
    """Generate and validate a SQL SELECT statement from a natural-language question.

    Calls the LLM with a SQL-focused system prompt, extracts and validates
    the output with sqlglot, and retries once if validation fails.

    Validation checks (in order):
    1. The response must be parseable by sqlglot for *dialect*.
    2. The top-level statement must be a ``SELECT``.
    3. Every table name referenced must appear in *knowledge_base* schema
       (CTE aliases are excluded from this check).

    On first failure the LLM is called a second time with a correction prompt
    that includes the validator's error message.  The second attempt's result
    is returned regardless of whether it passes validation.

    Args:
        question:       Natural-language question to translate to SQL.
        knowledge_base: Parsed YAML KB dict (schema + capsules context).
        dialect:        Target SQL dialect: ``"postgres"``, ``"snowflake"``,
                        or ``"bigquery"``.

    Returns:
        ``SQLGenerationResult`` — always returned, never raises on LLM or
        validation failure.
    """
    llm = get_llm_client()
    system_prompt = _build_sql_system_prompt(knowledge_base, dialect)

    # Attempt 1 ---------------------------------------------------------------
    raw1 = llm.complete(system_prompt, question)
    sql1 = _extract_sql(raw1)
    error1 = _validate_sql(sql1, knowledge_base, dialect)

    if error1 is None:
        return SQLGenerationResult(
            sql=sql1,
            is_valid=True,
            validation_error=None,
            dialect=dialect,
            attempt_count=1,
        )

    # Attempt 2: error-correction retry ---------------------------------------
    correction_user = (
        f"{question}\n\n"
        f"Your previous SQL had an error: {error1}\n"
        f"Please generate a corrected {dialect} SQL SELECT statement. "
        "Output only the raw SQL, no explanation or markdown."
    )
    raw2 = llm.complete(system_prompt, correction_user)
    sql2 = _extract_sql(raw2)
    error2 = _validate_sql(sql2, knowledge_base, dialect)

    return SQLGenerationResult(
        sql=sql2,
        is_valid=error2 is None,
        validation_error=error2,
        dialect=dialect,
        attempt_count=2,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# -- Phase 4B: schema dict for column validation ---------------------------


def _kb_to_sqlglot_schema(knowledge_base: dict[str, Any]) -> dict[str, Any]:
    """Build a sqlglot-compatible schema dict: ``{table: {column: type}}``."""
    schema: dict[str, dict[str, str]] = {}
    for table in knowledge_base.get("schema", {}).get("tables", []):
        name = table.get("name")
        if not name:
            continue
        schema[name] = {
            col["name"]: col.get("type", "TEXT")
            for col in table.get("columns", [])
            if col.get("name")
        }
    return schema


def _validate_numeric_clauses(statement: exp.Expression) -> str | None:
    """Check that LIMIT/OFFSET/FETCH carry integer literals, not string literals.

    LLMs sometimes produce ``LIMIT '10'`` when the question text contains a
    quoted number.  Most databases reject or silently mis-handle this.
    """
    for node in statement.find_all(exp.Literal):
        if not node.is_string:
            continue
        parent = node.parent
        if isinstance(parent, (exp.Limit, exp.Offset, exp.Fetch)):
            return (
                f"LIMIT/OFFSET value must be an integer literal, not a string: "
                f"'{node.this}' — use {node.this} (no quotes)"
            )
    return None


def _validate_columns(
    statement: exp.Expression,
    knowledge_base: dict[str, Any],
    dialect: str,
) -> str | None:
    """Validate explicit column references via ``sqlglot.optimizer.qualify``.

    Uses ``validate_qualify_columns=True`` and ``expand_stars=False`` so that
    ``SELECT *`` is always accepted while explicit unknown column names are
    caught.  Returns ``None`` (pass) when the schema is empty or when qualify
    raises unexpected errors, to avoid false rejections on complex queries.
    """
    from sqlglot.optimizer.qualify import qualify  # noqa: PLC0415

    schema = _kb_to_sqlglot_schema(knowledge_base)
    if not schema:
        return None
    try:
        qualify(
            statement.copy(),
            schema=schema,
            expand_stars=False,
            qualify_columns=True,
            validate_qualify_columns=True,
            dialect=dialect,
        )
        return None
    except sqlglot.errors.OptimizeError as exc:
        return f"Column validation error: {exc}"
    except Exception:  # noqa: BLE001
        return None  # be permissive — column validation is best-effort


# -- Phase 4B: EXPLAIN gate ------------------------------------------------


async def _run_explain_gate(sql: str, connector: Any, timeout: float = 5.0) -> str | None:
    """Run ``EXPLAIN <sql>`` via *connector.execute()*; return error or ``None``."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(connector.execute, f"EXPLAIN {sql}"),
            timeout=timeout,
        )
        return None
    except TimeoutError:
        return f"EXPLAIN timed out after {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return f"EXPLAIN failed: {exc}"


async def _apply_explain_gate(
    result: SQLGenerationResult,
    connector: Any,
    explain_check: bool,
    dialect: str,
) -> SQLGenerationResult:
    """Run the EXPLAIN gate on *result* when conditions are met.

    A no-op when *result* is already invalid, *connector* is ``None``, or
    *explain_check* is ``False``.
    """
    if not result.is_valid or connector is None or not explain_check:
        return result
    explain_error = await _run_explain_gate(result.sql, connector)
    if explain_error is None:
        return result
    return SQLGenerationResult(
        sql=result.sql,
        is_valid=False,
        validation_error=explain_error,
        dialect=dialect,
        attempt_count=result.attempt_count,
    )


def _build_sql_system_prompt(knowledge_base: dict[str, Any], dialect: str) -> str:
    """Build the system prompt for SQL generation from *knowledge_base*."""
    schema_ctx = _format_schema_for_prompt(knowledge_base)
    return (
        f"You are a {dialect} SQL generation assistant.\n"
        "Generate ONLY a single valid SQL SELECT statement that answers the user's question.\n"
        "Do not include any explanation or comments.\n"
        "Do not use markdown code blocks.\n"
        "Output only the raw SQL statement.\n\n"
    ) + schema_ctx


def _format_schema_for_prompt(knowledge_base: dict[str, Any]) -> str:
    """Render schema tables and sample capsules as a compact context block."""
    tables: list[dict[str, Any]] = knowledge_base.get("schema", {}).get("tables", [])
    lines: list[str] = []

    if tables:
        lines.append("## Database Schema")
        lines.append("")
        for table in tables:
            name = table.get("name", "")
            desc = table.get("description", "")
            header = f"Table: {name}"
            if desc:
                header += f" — {desc}"
            lines.append(header)
            for col in table.get("columns", []):
                lines.append(f"  - {col.get('name', '')} ({col.get('type', '')})")
            lines.append("")

    capsules: list[dict[str, Any]] = knowledge_base.get("query_capsules", [])
    if capsules:
        lines.append("## Example Queries")
        lines.append("")
        for cap in capsules[:3]:
            intent = cap.get("intent", "")
            template = cap.get("template", "")
            if intent and template:
                lines.append(f"-- {intent}")
                lines.append(template)
                lines.append("")

    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    """Extract the SQL statement from an LLM response.

    Handles three output formats (in priority order):

    1. ``<sql>…</sql>`` sentinels — the single-call streaming format.
    2. Markdown code fences (``\\`\\`\\`sql … \\`\\`\\``).
    3. First ``SELECT`` / ``WITH`` keyword scan (plain-text fallback).
    """
    text = text.strip()

    # 1. <sql>...</sql> sentinels (Phase 3B single-call format)
    sql_tag_match = re.search(r"<sql>\s*([\s\S]*?)\s*</sql>", text, re.IGNORECASE)
    if sql_tag_match:
        return sql_tag_match.group(1).strip()

    # 2. Markdown code fences
    if "```" in text:
        match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 3. Find whichever SQL-starting keyword (SELECT or WITH) appears earliest
    upper = text.upper()
    candidates: list[int] = []
    for keyword in ("SELECT", "WITH"):
        idx = upper.find(keyword)
        if idx != -1:
            candidates.append(idx)

    if candidates:
        return text[min(candidates) :].strip()

    return text


def _validate_sql(
    sql: str,
    knowledge_base: dict[str, Any],
    dialect: str,
) -> str | None:
    """Validate *sql* against the knowledge-base schema.

    Returns:
        ``None`` when *sql* is valid; an error-message string otherwise.

    Checks performed:
    - Non-empty
    - Parseable by sqlglot for *dialect*
    - Top-level statement is ``SELECT``
    - All referenced table names exist in ``knowledge_base["schema"]["tables"]``
      (CTE aliases are excluded from this check)
    """
    if not sql.strip():
        return "Generated SQL is empty"

    # Parse -----------------------------------------------------------------
    try:
        statement = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError as exc:
        return f"SQL parse error: {exc}"

    if statement is None:
        return "Generated SQL could not be parsed"

    # Must be SELECT --------------------------------------------------------
    if not isinstance(statement, exp.Select):
        stmt_type = type(statement).__name__
        return f"Only SELECT statements are allowed; got {stmt_type}"

    # All referenced tables must be in the schema ---------------------------
    schema_tables = {
        t.get("name", "").lower()
        for t in knowledge_base.get("schema", {}).get("tables", [])
        if t.get("name")
    }
    if schema_tables:
        # Collect CTE aliases to avoid false positives
        cte_names = {cte.alias.lower() for cte in statement.find_all(exp.CTE) if cte.alias}
        ast_tables = {
            t.name.lower()
            for t in statement.find_all(exp.Table)
            if t.name and t.name.lower() not in cte_names
        }
        unknown = ast_tables - schema_tables
        if unknown:
            return f"Statement references tables not in the schema: {', '.join(sorted(unknown))}"

    # Phase 4B: column-level check (after table check so schema is known-good)
    col_error = _validate_columns(statement, knowledge_base, dialect)
    if col_error is not None:
        return col_error

    # LIMIT/OFFSET must be integer literals, not string literals
    limit_error = _validate_numeric_clauses(statement)
    if limit_error is not None:
        return limit_error

    return None


def _try_mechanical_repair(
    sql: str,
    knowledge_base: dict[str, Any],
    dialect: str,
) -> tuple[str, str | None]:
    """Attempt to repair *sql* without an LLM call.

    Steps tried (cheapest first):

    1. ``sqlglot.transpile()`` normalises syntax for the target dialect and
       may fix minor parse issues (e.g. quoted identifiers, trailing
       semicolons).
    2. Case-insensitive table-name correction against KB names — handles LLMs
       that produce ``Orders`` instead of ``orders``.

    Returns:
        ``(repaired_sql, None)`` on success (SQL is valid).
        ``(original_sql, error_message)`` when no mechanical fix worked.
    """
    # Step 0: fix string literals used where integers are required (LIMIT/OFFSET/FETCH)
    try:
        stmt = sqlglot.parse_one(sql, dialect=dialect)
        modified = False
        for node in stmt.find_all(exp.Literal):
            if node.is_string and isinstance(node.parent, (exp.Limit, exp.Offset, exp.Fetch)):
                node.replace(exp.Literal.number(node.this))
                modified = True
        if modified:
            candidate = stmt.sql(dialect=dialect)
            err = _validate_sql(candidate, knowledge_base, dialect)
            if err is None:
                return candidate, None
    except Exception:  # noqa: BLE001
        pass

    # Step 1: dialect transpile
    try:
        transpiled_list = sqlglot.transpile(sql, write=dialect)
        if transpiled_list:
            candidate = transpiled_list[0]
            err = _validate_sql(candidate, knowledge_base, dialect)
            if err is None:
                return candidate, None
    except Exception:  # noqa: BLE001
        pass

    # Step 2: case-insensitive table name correction
    schema_tables = {
        t.get("name", "").lower(): t.get("name", "")
        for t in knowledge_base.get("schema", {}).get("tables", [])
        if t.get("name")
    }
    if schema_tables:
        try:
            stmt = sqlglot.parse_one(sql, dialect=dialect)
            corrected = sql
            for tbl in stmt.find_all(exp.Table):
                if tbl.name and tbl.name.lower() in schema_tables:
                    canonical = schema_tables[tbl.name.lower()]
                    if tbl.name != canonical:
                        corrected = corrected.replace(tbl.name, canonical)
            if corrected != sql:
                err = _validate_sql(corrected, knowledge_base, dialect)
                if err is None:
                    return corrected, None
        except Exception:  # noqa: BLE001
            pass

    # Step 3: strip garbage after first statement (handles multi-statement output)
    try:
        statements = sqlglot.parse(sql, dialect=dialect)
        if len(statements) > 1 and statements[0] is not None:
            candidate = statements[0].sql(dialect=dialect)
            err = _validate_sql(candidate, knowledge_base, dialect)
            if err is None:
                return candidate, None
    except Exception:  # noqa: BLE001
        pass

    return sql, "no mechanical fix found"
