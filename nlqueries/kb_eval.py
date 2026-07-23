# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.kb_eval
~~~~~~~~~~~~~~~~~
Community KB eval harness (SYL-3.2). Re-ask an agent's mined capsules (and,
optionally, a golden question set), generate SQL, and check that it **parses**
and **references only tables the agent knows** — a lightweight regression gate
usable from the CLI or CI. The richer QueryGraph-shape comparison (drift /
benign-diff classification) lives in the enterprise tier; this OSS version is a
deliberately simple "does the generated SQL still make structural sense" check.

Pure and deterministic: :func:`check_generated_sql` is the unit, and
:func:`run_eval` takes an injected ``generate_sql`` callable so it is testable
without a live agent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp


@dataclass(frozen=True)
class EvalQuestion:
    """One question to re-ask, and (for golden cases) the tables it should hit."""

    question: str
    source: str  # "capsule" | "golden"
    expect_tables: list[str] | None = None


@dataclass
class CaseOutcome:
    """The result of evaluating one question."""

    question: str
    source: str
    sql: str | None
    ok: bool
    reason: str


def known_tables(kb: dict[str, Any]) -> set[str]:
    """The lowercased set of table names the agent's KB knows about."""
    schema = kb.get("schema", {}) if isinstance(kb, dict) else {}
    tables = schema.get("tables", []) if isinstance(schema, dict) else []
    return {str(t["name"]).lower() for t in tables if isinstance(t, dict) and t.get("name")}


def build_cases(
    kb: dict[str, Any],
    agent_id: str,
    *,
    golden: Sequence[dict[str, Any]] | None = None,
    max_cases: int = 50,
) -> list[EvalQuestion]:
    """Build eval questions from the KB's capsules + optional golden questions."""
    cases: list[EvalQuestion] = []
    capsules = kb.get("query_capsules", []) if isinstance(kb, dict) else []
    for capsule in capsules:
        if not isinstance(capsule, dict):
            continue
        intent = str(capsule.get("intent") or "").strip()
        if intent:
            cases.append(EvalQuestion(question=intent, source="capsule"))

    for entry in golden or []:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("q") or "").strip()
        # Golden files can carry cases for many agents; keep the ones for this one
        # (or unscoped entries).
        if question and (not entry.get("agent") or entry.get("agent") == agent_id):
            expect = entry.get("expect_tables")
            cases.append(
                EvalQuestion(
                    question=question,
                    source="golden",
                    expect_tables=list(expect) if isinstance(expect, list) else None,
                )
            )

    return cases[: max(0, max_cases)]


def check_generated_sql(
    sql: str | None,
    known: set[str],
    expect_tables: list[str] | None = None,
    dialect: str = "postgres",
) -> tuple[bool, str]:
    """Check that *sql* parses and references only *known* tables.

    When *expect_tables* is given (golden cases), also require those tables to
    appear. Returns ``(ok, reason)``.
    """
    if not sql:
        return False, "no SQL generated"
    try:
        statement = sqlglot.parse_one(sql, dialect=dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        return False, f"SQL did not parse: {exc}"
    except Exception:  # noqa: BLE001 — any sqlglot failure is a parse failure here
        return False, "SQL did not parse"
    if statement is None:
        return False, "SQL did not parse"
    if not isinstance(statement, exp.Select):
        return False, f"not a SELECT (got {type(statement).__name__})"

    cte_names = {str(c.alias).lower() for c in statement.find_all(exp.CTE) if c.alias}
    tables = {str(t.name).lower() for t in statement.find_all(exp.Table) if t.name}
    unknown = tables - known - cte_names
    if unknown:
        return False, f"references unknown table(s): {sorted(unknown)}"

    if expect_tables:
        missing = {t.lower() for t in expect_tables} - tables
        if missing:
            return False, f"missing expected table(s): {sorted(missing)}"

    return True, "ok"


def run_eval(
    kb: dict[str, Any],
    cases: Sequence[EvalQuestion],
    generate_sql: Callable[[str], str | None],
    dialect: str = "postgres",
) -> list[CaseOutcome]:
    """Evaluate *cases*: generate SQL for each and check it.

    ``generate_sql`` maps a question to its generated SQL (or ``None``) — injected
    so this runs without a live agent in tests.
    """
    known = known_tables(kb)
    outcomes: list[CaseOutcome] = []
    for case in cases:
        try:
            sql = generate_sql(case.question)
        except Exception:  # noqa: BLE001 — a generation failure is a case failure
            sql = None
        ok, reason = check_generated_sql(sql, known, case.expect_tables, dialect)
        outcomes.append(
            CaseOutcome(question=case.question, source=case.source, sql=sql, ok=ok, reason=reason)
        )
    return outcomes
