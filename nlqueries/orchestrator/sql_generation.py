"""
nlqueries.orchestrator.sql_generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM-driven SQL generation with sqlglot AST validation.

Public API
----------
``SQLGenerationResult``
    Dataclass carrying the generated SQL, validity flag, and metadata.

``generate_sql(question, knowledge_base, dialect)``
    Call the LLM to generate a SQL SELECT statement, validate it with
    sqlglot, and retry once on failure. Always returns a
    ``SQLGenerationResult`` even when both attempts fail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.expressions as exp

from nlqueries.llm import get_llm_client


@dataclass
class SQLGenerationResult:
    """Result of a SQL generation attempt.

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

    Strips markdown code fences (``\\`\\`\\`sql … \\`\\`\\``  and  ``\\`\\`\\` … \\`\\`\\``),
    then finds the first SQL keyword (``SELECT`` or ``WITH``) to skip any
    leading prose.
    """
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Find whichever SQL-starting keyword (SELECT or WITH) appears earliest
    upper = text.upper()
    candidates: list[int] = []
    for keyword in ("SELECT", "WITH"):
        idx = upper.find(keyword)
        if idx != -1:
            candidates.append(idx)

    if candidates:
        return text[min(candidates):].strip()

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

    return None
