"""
tests.test_reading
~~~~~~~~~~~~~~~~~~
How the question was read, said beside the answer (owner feedback F20, phase
1): the text values, ranges, two-way matches and row limit the final statement
uses, each in a few words, from the statement alone.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.reading import MAX_ITEMS, read_as
from nlqueries.orchestrator.sql_generation import SQLGenerationResult
from nlqueries.query_settings import QuerySettings, use_query_settings

NOON = (
    "SELECT f.id FROM flights f WHERE f.origin = 'SFO' "
    "AND f.dep >= 1130 AND f.dep <= 1230 ORDER BY f.fare DESC LIMIT 10"
)
SAID = ["origin 'SFO'", "dep from 1130 to 1230", "top 10 by fare"]


def _said(sql: str, dialect: str = "postgres") -> list[str]:
    return [item.text for item in read_as(sql, dialect)]


# --- The rules ----------------------------------------------------------------------


def test_values_ranges_and_the_limit_are_said_in_order() -> None:
    items = read_as(NOON, "postgres")

    assert [i.text for i in items] == SAID
    assert [i.kind for i in items] == ["value", "range", "limit"]
    assert items[0].as_dict() == {"kind": "value", "text": "origin 'SFO'"}


@pytest.mark.parametrize(
    ("sql", "said"),
    [
        ("SELECT * FROM t WHERE status IN ('A', 'B')", ["status 'A' or 'B'"]),
        ("SELECT * FROM t WHERE 'SFO' = origin", ["origin 'SFO'"]),
        # A number is not a value the question named in words.
        ("SELECT * FROM t WHERE id = 3", []),
        ("SELECT * FROM t WHERE status IN (SELECT s FROM u)", []),
        ("SELECT * FROM t WHERE status IN ('A', 3)", []),
    ],
)
def test_a_column_compared_with_text_values_is_said(sql: str, said: list[str]) -> None:
    assert _said(sql) == said


@pytest.mark.parametrize(
    ("sql", "said"),
    [
        (
            "SELECT * FROM t WHERE d BETWEEN '2024-01-01' AND '2024-01-31'",
            ["d from '2024-01-01' to '2024-01-31'"],
        ),
        ("SELECT * FROM t WHERE price > 100 AND price < 200", ["price over 100 and under 200"]),
        (
            "SELECT * FROM t WHERE price >= 100 AND price < 200",
            ["price at least 100 and under 200"],
        ),
        # The literal on the left holds the column from the other side.
        ("SELECT * FROM t WHERE 100 <= price AND 200 >= price", ["price from 100 to 200"]),
        # One bound is the question's own words more often than not.
        ("SELECT * FROM t WHERE price > 100", []),
        ("SELECT * FROM t WHERE price > 100 AND cost < 5", []),
    ],
)
def test_a_column_held_between_two_bounds_is_said(sql: str, said: list[str]) -> None:
    assert _said(sql) == said


@pytest.mark.parametrize(
    ("sql", "said"),
    [
        (
            "SELECT * FROM f WHERE (a = 'BOS' AND b = 'DEN') OR (a = 'DEN' AND b = 'BOS')",
            ["either way between 'BOS' and 'DEN'"],
        ),
        # The same way twice, or another value, is not both ways.
        ("SELECT * FROM f WHERE (a = 'BOS' AND b = 'DEN') OR (a = 'BOS' AND b = 'DEN')", []),
        ("SELECT * FROM f WHERE (a = 'BOS' AND b = 'DEN') OR (a = 'DEN' AND b = 'LAX')", []),
        ("SELECT * FROM f WHERE (a = 'BOS' AND b = 'BOS') OR (a = 'BOS' AND b = 'BOS')", []),
        ("SELECT * FROM f WHERE (a = 'BOS' AND b = 'DEN') OR (a = 'DEN' AND c = 'BOS')", []),
        # A self-join: the same column name on two tables is two columns.
        (
            "SELECT * FROM f JOIN airport o ON f.o = o.id JOIN airport d ON f.d = d.id "
            "WHERE (o.code = 'BOS' AND d.code = 'DEN') OR (o.code = 'DEN' AND d.code = 'BOS')",
            ["either way between 'BOS' and 'DEN'"],
        ),
    ],
)
def test_two_columns_matched_either_way_round_are_said(sql: str, said: list[str]) -> None:
    assert _said(sql) == said


@pytest.mark.parametrize(
    ("sql", "said"),
    [
        # A bound on each of two tables' columns of one name is no range.
        (
            "SELECT * FROM o JOIN s ON o.id = s.id "
            "WHERE o.created_at >= '2024-01-01' AND s.created_at < '2024-02-01'",
            [],
        ),
        # Two ranges on columns of one name are each said with their table.
        (
            "SELECT * FROM o JOIN s ON o.id = s.id "
            "WHERE o.d >= 1 AND o.d <= 5 AND s.d >= 2 AND s.d <= 3",
            ["o.d from 1 to 5", "s.d from 2 to 3"],
        ),
    ],
)
def test_columns_of_one_name_on_two_tables_are_kept_apart(sql: str, said: list[str]) -> None:
    assert _said(sql) == said


@pytest.mark.parametrize(
    ("sql", "dialect", "said"),
    [
        ("SELECT * FROM t ORDER BY revenue DESC LIMIT 5", "postgres", ["top 5 by revenue"]),
        ("SELECT * FROM t ORDER BY revenue LIMIT 5", "postgres", ["bottom 5 by revenue"]),
        ("SELECT TOP 3 name FROM t ORDER BY revenue DESC", "mssql", ["top 3 by revenue"]),
        ("SELECT * FROM t LIMIT 5", "postgres", ["any 5 rows, in no set order"]),
        (
            "SELECT name, SUM(x) FROM t GROUP BY name ORDER BY SUM(x) DESC LIMIT 1",
            "duckdb",
            ["top 1 by SUM(x)"],
        ),
        # Ordering with no limit cuts nothing.
        ("SELECT * FROM t ORDER BY revenue DESC", "postgres", []),
        # Rows skipped first are said: these are the 11th to the 20th.
        (
            "SELECT * FROM t ORDER BY fare DESC LIMIT 10 OFFSET 10",
            "postgres",
            ["top 10 by fare after the first 10"],
        ),
        (
            "SELECT * FROM t LIMIT 10 OFFSET 20",
            "mysql",
            ["any 10 rows after the first 20, in no set order"],
        ),
        # An offset that is not a number cannot be said, so neither is the limit.
        ("SELECT * FROM t ORDER BY fare DESC LIMIT 10 OFFSET :skip", "postgres", []),
    ],
)
def test_a_row_limit_is_said_with_the_ordering_it_cuts(
    sql: str, dialect: str, said: list[str]
) -> None:
    assert _said(sql, dialect) == said


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE a = 'x' UNION SELECT a FROM u WHERE a = 'y'",
        "DELETE FROM t WHERE a = 'x'",
        "not sql at all (",
        # Only the outermost statement's own conditions.
        "SELECT * FROM (SELECT * FROM t WHERE a = 'x' LIMIT 3) AS s",
        "SELECT * FROM t WHERE a = 'x' OR b = 'y'",
    ],
)
def test_nothing_is_said_beyond_the_outermost_select_and_its_and_conditions(sql: str) -> None:
    assert _said(sql) == []


def test_at_most_a_few_items_are_said() -> None:
    sql = "SELECT * FROM t WHERE " + " AND ".join(f"c{i} = 'v{i}'" for i in range(9)) + " LIMIT 4"

    assert _said(sql) == [f"c{i} 'v{i}'" for i in range(MAX_ITEMS)]


# --- Through the orchestrator -------------------------------------------------------


class _QuietLLM:
    supports_prompt_caching = False

    async def astream(self, system: Any, user: str) -> Any:
        for t in ():
            yield t

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        return ""


def _answer(
    tmp_path: Path, settings: QuerySettings, *, sql: str = NOON, valid: bool = True
) -> dict[str, Any]:
    from nlqueries.orchestrator.orchestrator import Orchestrator

    first = SQLGenerationResult(
        sql=sql,
        is_valid=valid,
        validation_error=None if valid else "refused",
        dialect="postgres",
        attempt_count=1,
    )
    kb = {
        "schema": {"tables": [{"name": "flights", "columns": [{"name": "id", "type": "INT"}]}]},
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }
    (tmp_path / "agent1.yaml").write_text(yaml.dump(kb), encoding="utf-8")
    with (
        patch("nlqueries.orchestrator.orchestrator.config") as cfg,
        patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=_QuietLLM()),
        patch(
            "nlqueries.orchestrator.orchestrator.validate_and_repair",
            new=AsyncMock(return_value=first),
        ),
        patch("nlqueries.connectors.loader.lookup_source", side_effect=LookupError("none")),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        use_query_settings(settings),
    ):
        cfg.KB_PATH = tmp_path
        cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0

        async def collect() -> list[str]:
            return [
                tok
                async for tok in Orchestrator().handle_question(
                    "fares around noon from san francisco",
                    "agent1",
                    dialect="postgres",
                    execution=ExecutionPolicy.generate_only(),
                )
            ]

        tokens = asyncio.run(collect())
    frame: dict[str, Any] = json.loads(tokens[-1])
    return frame


def test_the_answer_says_how_the_question_was_read(tmp_path: Path) -> None:
    frame = _answer(tmp_path, QuerySettings(reading_line=True))

    assert [item["text"] for item in frame["reading"]] == SAID


def test_off_by_default_nothing_is_said(tmp_path: Path) -> None:
    assert "reading" not in _answer(tmp_path, QuerySettings())


def test_a_statement_that_is_not_valid_is_not_read(tmp_path: Path) -> None:
    frame = _answer(tmp_path, QuerySettings(reading_line=True), valid=False)

    assert "reading" not in frame


def test_nothing_to_say_leaves_the_frame_as_it_was(tmp_path: Path) -> None:
    frame = _answer(tmp_path, QuerySettings(reading_line=True), sql="SELECT id FROM flights")

    assert "reading" not in frame


def _cache_hit(settings: QuerySettings, sql: str = NOON) -> dict[str, Any]:
    from nlqueries.cache.semantic_cache import CacheEntry
    from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

    entry = CacheEntry(
        question="fares around noon",
        resolved_question="fares around noon",
        agent_type="sql",
        answer="Here are the fares.",
        sql=sql,
        created_at=datetime.now(UTC),
        hit_count=1,
    )

    async def run() -> list[str]:
        with (
            patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as cache,
            patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator"),
            patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            use_query_settings(settings),
        ):
            cache.return_value.get.return_value = entry
            return [
                t
                async for t in MultiAgentOrchestrator().handle_question(
                    "fares around noon", "agent1", dialect="postgres"
                )
            ]

    frame: dict[str, Any] = json.loads(asyncio.run(run())[-1])
    return frame


def test_a_cached_answer_says_what_the_live_one_said() -> None:
    hit = _cache_hit(QuerySettings(reading_line=True))

    assert hit["from_cache"] is True
    assert [item["text"] for item in hit["reading"]] == SAID
    assert "reading" not in _cache_hit(QuerySettings())
    # A statement the policy refuses is not run, and not read either.
    refused = "SELECT * FROM t WHERE a = 'x' AND pg_sleep(1) IS NULL"
    assert read_as(refused, "postgres"), "it would say something if it were read"
    assert "reading" not in _cache_hit(QuerySettings(reading_line=True), sql=refused)


# --- Entry points ------------------------------------------------------------------


def _result(reading: list[dict[str, str]]) -> Any:
    from nlqueries.orchestrator.sync_runner import AgentQueryResult

    return AgentQueryResult(
        question="q",
        resolved_question="q",
        agent_type="sql",
        answer="",
        sql=NOON,
        sql_result=None,
        sql_is_valid=True,
        execution_mode="generate_only",
        citations=[],
        merged_answer=None,
        latency_ms=1,
        session_id=None,
        reading=reading,
    )


READING = [
    {"kind": "value", "text": "origin 'SFO'"},
    {"kind": "range", "text": "dep from 1130 to 1230"},
]


def test_run_query_carries_the_reading() -> None:
    from nlqueries.orchestrator import sync_runner

    async def handle(*_a: Any, **_kw: Any) -> Any:
        # As the multi-agent orchestrator hands it on: with its agent type.
        frame = {"type": "sql", "agent_type": "sql", "sql": NOON, "is_valid": True}
        yield json.dumps({**frame, "reading": READING})

    with patch.object(sync_runner.MultiAgentOrchestrator, "handle_question", handle):
        result = asyncio.run(sync_runner.run_query("q", "agent1"))

    assert result.reading == READING


def test_the_cli_says_it_and_puts_it_in_json() -> None:
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    with (
        patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=_result(READING)),
        patch("nlqueries.cli.main._resolve_dialect", return_value="postgres"),
        patch("nlqueries.orchestrator.sync_runner.answer_types_for", return_value=["sql"]),
    ):
        text = CliRunner().invoke(cli, ["query", "agent1", "q", "--no-execute"])
        as_json = CliRunner().invoke(cli, ["query", "agent1", "q", "--no-execute", "--json"])

    assert text.exit_code == 0, text.output
    assert "Read as    : origin 'SFO'; dep from 1130 to 1230" in text.output
    assert json.loads(as_json.output)["reading"] == READING


def test_the_mcp_tool_says_it() -> None:
    from nlqueries.mcp_server import server
    from nlqueries.orchestrator import sync_runner

    async def run(*_a: Any, **_kw: Any) -> Any:
        return _result(READING)

    async def silent(*_a: Any, **_kw: Any) -> Any:
        return _result([])

    with (
        patch.object(sync_runner, "answer_types_for", return_value=["sql"]),
        patch("nlqueries.connectors.loader.dialect_for_agent", return_value="postgres"),
    ):
        with patch.object(sync_runner, "run_query", side_effect=run):
            said = asyncio.run(server.query("q", "agent1"))
        with patch.object(sync_runner, "run_query", side_effect=silent):
            quiet = asyncio.run(server.query("q", "agent1"))

    assert "**Read as**: origin 'SFO'; dep from 1130 to 1230" in said
    assert "Read as" not in quiet
