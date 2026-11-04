"""The plan line, and the statement checked against it.

The model begins its reasoning with one line saying what one row of the answer
is and what is counted. Two disagreements between that line and the statement
are caught without a model call: rows told apart by a key the statement does
not group by, and a plain COUNT across a join the knowledge base's foreign keys
say repeats the counted rows. What these pin: the line is read however it is
decorated; each check fires on its shape and not on the shapes beside it; and
a repair replaces the statement only when it is valid, different and agrees
with its own plan line.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator import grain_check
from nlqueries.orchestrator.grain_check import (
    GrainPlan,
    check_grain,
    mismatches,
    parse_plan,
)
from nlqueries.orchestrator.orchestrator import Orchestrator
from nlqueries.orchestrator.provenance import Provenance, use_provenance
from nlqueries.orchestrator.sql_generation import SQLGenerationResult


def _col(name: str, *, pk: bool = False, ref: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "type": "INTEGER" if name.endswith("id") or name in ("qty",) else "TEXT",
        "description": "",
        "is_primary_key": pk,
        "is_foreign_key": ref is not None,
        "references": ref,
    }


def _kb(*, references: bool = True) -> dict[str, Any]:
    def ref(target: str) -> str | None:
        return target if references else None

    return {
        "schema": {
            "tables": [
                {"name": "customers", "columns": [_col("id", pk=True), _col("name")]},
                {
                    "name": "orders",
                    "columns": [
                        _col("id", pk=True),
                        _col("customer_id", ref=ref("customers.id")),
                        _col("status"),
                    ],
                },
                {
                    "name": "order_items",
                    "columns": [
                        _col("id", pk=True),
                        _col("order_id", ref=ref("public.orders.id")),
                        _col("qty"),
                    ],
                },
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }


_PER_CUSTOMER = (
    "Plan: rows = one customer per row, key customers.id; counted = orders, once each by "
    "orders.id; filters = none"
)


# ---------------------------------------------------------------------------
# Reading the plan line
# ---------------------------------------------------------------------------


def test_the_plan_line_is_read_from_the_reasoning() -> None:
    output = f"{_PER_CUSTOMER}\nEach customer's orders are counted.\n<sql>SELECT 1</sql>"
    plan = parse_plan(output)
    assert plan == GrainPlan(
        text=_PER_CUSTOMER.removeprefix("Plan: "),
        row_key=("customers", "id"),
        count_key=("orders", "id"),
    )


@pytest.mark.parametrize(
    "line",
    [
        "**Plan:** rows = one Customer per row, key `Customers`.`ID`; counted = none",
        '- plan: rows = one customer per row, key "customers"."id"; counted = none',
        "> Plan:   rows = one customer per row, KEY customers.id ; counted = none",
    ],
)
def test_the_plan_line_is_read_however_it_is_decorated(line: str) -> None:
    plan = parse_plan(f"Some text first.\n{line}\nMore.")
    assert plan is not None
    assert plan.row_key == ("customers", "id")
    assert plan.count_key is None


def test_a_single_row_and_nothing_counted_name_no_keys() -> None:
    plan = parse_plan("Plan: rows = a single row; counted = none; filters = status is open")
    assert plan is not None
    assert (plan.row_key, plan.count_key) == (None, None)


@pytest.mark.parametrize(
    "filters",
    ["key customers.id = 7", "visits made once each by visits.customer_id in a day"],
)
def test_a_key_named_in_another_field_is_not_taken_for_this_one(filters: str) -> None:
    # The filter's words are the filter's; neither the rows nor the count name a key.
    plan = parse_plan(f"Plan: rows = a single row; counted = none; filters = {filters}")
    assert plan is not None
    assert (plan.row_key, plan.count_key) == (None, None)


def test_no_plan_line_is_none() -> None:
    assert parse_plan("Customers are counted.\n<sql>SELECT 1</sql>") is None


# ---------------------------------------------------------------------------
# Rows told apart by the key
# ---------------------------------------------------------------------------

_ROWS = GrainPlan(text="", row_key=("customers", "id"), count_key=None)


@pytest.mark.parametrize(
    ("sql", "flagged"),
    [
        (
            "SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.name",
            True,
        ),
        ("SELECT name, COUNT(*) FROM customers GROUP BY name", True),
        (
            "SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id, c.name",
            False,
        ),
        # Grouped by the foreign key, which is the customer: nothing to say.
        ("SELECT o.customer_id, COUNT(*) FROM orders o GROUP BY o.customer_id", False),
        ("SELECT c.name FROM customers c", False),
        # An unqualified column across two tables is not resolved, not guessed.
        (
            "SELECT name, COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.id "
            "GROUP BY name",
            False,
        ),
    ],
)
def test_rows_grouped_without_their_key_are_flagged(sql: str, flagged: bool) -> None:
    found = mismatches(_ROWS, sql, "sqlite", _kb())
    assert bool(found) is flagged
    if flagged:
        assert found[0].startswith("The plan tells the rows apart by customers.id")
        assert "groups customers by name without it" in found[0]


# ---------------------------------------------------------------------------
# Counted once, through a join that repeats
# ---------------------------------------------------------------------------

_COUNT = GrainPlan(text="", row_key=None, count_key=("orders", "id"))


@pytest.mark.parametrize(
    ("sql", "flagged"),
    [
        ("SELECT COUNT(*) FROM orders o JOIN order_items i ON i.order_id = o.id", True),
        ("SELECT COUNT(o.id) FROM orders o JOIN order_items i ON o.id = i.order_id", True),
        (
            "SELECT COUNT(DISTINCT o.id) FROM orders o JOIN order_items i ON i.order_id = o.id",
            False,
        ),
        # Orders to customers is many to one: each order still appears once.
        ("SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id", False),
        ("SELECT COUNT(*) FROM orders o", False),
        # The count is inside a derived table, over no join.
        ("SELECT t.n FROM (SELECT COUNT(*) AS n FROM orders) t", False),
        # A count of its own in a subquery, beside an outer join with no count.
        (
            "SELECT o.id, (SELECT COUNT(*) FROM customers) FROM orders o "
            "JOIN order_items i ON i.order_id = o.id",
            False,
        ),
        # A union is not read: which branch a count stands for is not knowable.
        (
            "SELECT COUNT(*) FROM orders o JOIN order_items i ON i.order_id = o.id UNION SELECT 0",
            False,
        ),
    ],
)
def test_a_plain_count_across_a_repeating_join_is_flagged(sql: str, flagged: bool) -> None:
    found = mismatches(_COUNT, sql, "sqlite", _kb())
    assert bool(found) is flagged
    if flagged:
        assert "counts the rows of its join to order_items" in found[0]
        assert "COUNT(DISTINCT orders.id)" in found[0]


def test_without_foreign_keys_the_join_is_not_judged() -> None:
    sql = "SELECT COUNT(*) FROM orders o JOIN order_items i ON i.order_id = o.id"
    assert mismatches(_COUNT, sql, "sqlite", _kb(references=False)) == []


def test_unparseable_sql_has_no_mismatches() -> None:
    assert mismatches(_COUNT, "SELECT FROM WHERE", "sqlite", _kb()) == []


# ---------------------------------------------------------------------------
# The repair
# ---------------------------------------------------------------------------

_WRONG = (
    "SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name"
)
_RIGHT = (
    "SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.id "
    "GROUP BY c.id, c.name"
)
_QUESTION = "how many orders each customer placed"


class _LLM:
    supports_prompt_caching = False

    def __init__(self, reply: str | Exception, stream: str = "") -> None:
        self._reply = reply
        self._stream = stream
        self.calls: list[str] = []

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        self.calls.append(user)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    async def astream(self, system: Any, user: str) -> Any:
        yield self._stream


_RECORDED: dict[str, Provenance] = {}


def _check(
    llm: _LLM,
    *,
    sql: str = _WRONG,
    output: str = _PER_CUSTOMER,
    started: float | None = None,
    deadline: float | None = None,
) -> SQLGenerationResult:
    first = SQLGenerationResult(
        sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )
    _RECORDED["last"] = collected = Provenance()
    with use_provenance(collected):
        return asyncio.run(
            check_grain(
                first,
                output,
                question=_QUESTION,
                knowledge_base=_kb(),
                dialect="sqlite",
                llm=llm,  # type: ignore[arg-type]
                system="system",
                started=started,
                deadline=deadline,
            )
        )


def _record() -> dict[str, Any] | None:
    return _RECORDED["last"].to_dict()["grain_check"]


def test_a_repair_that_agrees_with_its_plan_replaces_the_statement() -> None:
    llm = _LLM(f"{_PER_CUSTOMER}\n<sql>{_RIGHT}</sql>")
    checked = _check(llm)
    assert checked.sql == _RIGHT
    assert checked.attempt_count == 2
    [prompt] = llm.calls
    assert _QUESTION in prompt and _WRONG in prompt and _PER_CUSTOMER in prompt
    assert "groups customers by name without it" in prompt
    record = _record()
    assert record is not None
    assert (record["attempted"], record["changed"], record["reason"]) == (True, True, None)
    assert record["plan"] == _PER_CUSTOMER.removeprefix("Plan: ")
    assert len(record["mismatches"]) == 1


def test_a_repair_is_checked_against_its_own_plan_line() -> None:
    # The model decided the rows are names, not customers, and said so.
    by_name = "Plan: rows = one name per row, key customers.name; counted = none; filters = none"
    other = "SELECT c.name, COUNT(*) AS n FROM customers c GROUP BY c.name"
    assert _check(_LLM(f"{by_name}\n<sql>{other}</sql>")).sql == other


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (f"<sql>{_WRONG}</sql>", "rejected: returned unchanged"),
        ("<sql>SELECT name FROM ghosts</sql>", "rejected: refused by validation"),
        (
            "<sql>SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o "
            "ON o.customer_id = c.id GROUP BY c.name</sql>",
            "rejected: still disagrees with its plan line",
        ),
        (RuntimeError("provider unavailable"), "failed: the model call raised"),
    ],
    ids=["unchanged", "invalid", "still-disagrees", "llm-failed"],
)
def test_otherwise_the_statement_stands(reply: Any, reason: str) -> None:
    llm = _LLM(reply)
    checked = _check(llm)
    assert checked.sql == _WRONG and checked.attempt_count == 1
    assert len(llm.calls) == 1
    record = _record()
    assert record is not None
    assert (record["attempted"], record["changed"], record["reason"]) == (True, False, reason)


def test_no_repair_starts_with_less_time_left_than_the_attempt_has_taken() -> None:
    llm = _LLM(f"<sql>{_RIGHT}</sql>")
    now = time.monotonic()
    assert _check(llm, started=now - 10, deadline=now + 3).sql == _WRONG
    assert llm.calls == []
    record = _record()
    assert record is not None and record["attempted"] is False
    assert record["reason"] == "skipped: too little time before the deadline"


@pytest.mark.parametrize(
    ("sql", "output"),
    [
        (_RIGHT, _PER_CUSTOMER),  # agrees
        (_WRONG, "No plan line here."),  # nothing to check against
    ],
    ids=["agrees", "no-plan-line"],
)
def test_nothing_is_asked_or_recorded_without_a_disagreement(sql: str, output: str) -> None:
    llm = _LLM(f"<sql>{_RIGHT}</sql>")
    assert _check(llm, sql=sql, output=output).sql == sql
    assert llm.calls == []
    assert _record() is None


def test_switched_off_the_statement_is_not_checked() -> None:
    llm = _LLM(f"<sql>{_RIGHT}</sql>")
    with patch.object(grain_check.config, "GRAIN_CHECK", False):
        assert _check(llm).sql == _WRONG
    assert llm.calls == []


# ---------------------------------------------------------------------------
# In the orchestrator
# ---------------------------------------------------------------------------


def _ask(llm: _LLM, generated: str) -> tuple[dict[str, Any], Any]:
    first = SQLGenerationResult(
        sql=generated, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        safe = re.sub(r"[^\w.-]", "_", "agent1")
        (kb_path / f"{safe}.yaml").write_text(yaml.dump(_kb()), encoding="utf-8")
        with (
            patch("nlqueries.orchestrator.orchestrator.config") as cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=first),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
            patch("nlqueries.orchestrator.orchestrator.get_tracer") as tracer,
        ):
            cfg.KB_PATH = kb_path
            cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0
            cfg.LITERAL_GROUNDING = False
            span = tracer.return_value.start_as_current_span.return_value.__enter__.return_value

            async def collect() -> list[str]:
                return [
                    tok
                    async for tok in Orchestrator().handle_question(
                        _QUESTION,
                        "agent1",
                        dialect="sqlite",
                        execution=ExecutionPolicy.generate_only(),
                    )
                ]

            tokens = asyncio.run(collect())
    return json.loads(tokens[-1]), span


def test_the_orchestrator_checks_the_statement_against_the_streamed_plan_line() -> None:
    llm = _LLM(
        f"<sql>{_RIGHT}</sql>",
        stream=f"{_PER_CUSTOMER}\nCounting each customer's orders.\n<sql>{_WRONG}</sql>",
    )
    frame, span = _ask(llm, _WRONG)
    assert frame["sql"] == _RIGHT
    assert frame["attempt_count"] == 2
    span.set_attribute.assert_any_call("grain_repair", True)


def test_the_orchestrator_leaves_a_statement_that_agrees() -> None:
    llm = _LLM("", stream=f"{_PER_CUSTOMER}\n<sql>{_RIGHT}</sql>")
    frame, span = _ask(llm, _RIGHT)
    assert frame["sql"] == _RIGHT
    assert llm.calls == []
    assert ("grain_repair", True) not in [c.args for c in span.set_attribute.call_args_list]


def test_the_connector_is_not_needed() -> None:
    # Generate-only: the check reads the statement, never the database.
    llm = _LLM(f"<sql>{_RIGHT}</sql>", stream=f"{_PER_CUSTOMER}\n<sql>{_WRONG}</sql>")
    with patch("nlqueries.connectors.loader.open_connector_for_agent") as opened:
        frame, _span = _ask(llm, _WRONG)
    opened.assert_not_called()
    assert frame["sql"] == _RIGHT


# ---------------------------------------------------------------------------
# The prompt asks for the line
# ---------------------------------------------------------------------------


def test_the_prompt_asks_for_the_plan_line_before_the_reasoning() -> None:
    from nlqueries.orchestrator.prompt_assembly import assemble_prompt

    static = assemble_prompt("How many orders?", _kb()).static_system
    form = (
        "Plan: rows = one <thing> per row, key <table.column>; counted = <thing>, once each "
        "by <table.column>; filters = <conditions>"
    )
    assert form in static
    assert static.index(form) < static.index("Then briefly explain your reasoning")
    assert "Write rows = a single row when the answer is one number or value" in static


def test_the_form_the_prompt_shows_is_one_the_parser_reads() -> None:
    filled = (
        "Plan: rows = one customer per row, key customers.id; counted = orders, once each "
        "by orders.id; filters = none"
    )
    plan = parse_plan(filled)
    assert plan is not None
    assert (plan.row_key, plan.count_key) == (("customers", "id"), ("orders", "id"))
