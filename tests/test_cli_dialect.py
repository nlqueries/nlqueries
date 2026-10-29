"""``--dialect`` on ``ask``, ``query`` and ``eval``: what it accepts, and its default.

It offered ``postgres``, ``snowflake`` and ``bigquery`` only, so a SQLite
connector could not be asked for SQLite SQL. It now
takes the grammar of every engine ``nlqueries connect`` registers, and without
the flag follows the connector.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import sqlglot
from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import _DB_SCHEMES, DIALECT_CHOICES, _resolve_dialect, cli
from nlqueries.sql_policy import _sqlglot_dialect, evaluate

AGENT = "sales-dev"

# ---------------------------------------------------------------------------
# The list itself
# ---------------------------------------------------------------------------


def test_the_choices_are_every_engine_connect_registers() -> None:
    """One list, kept beside the db-types it mirrors, so it cannot fall behind
    them the way the three copies of ``postgres`` / ``snowflake`` / ``bigquery``
    did. ``postgresql`` is a second spelling of ``postgres``, not an engine."""
    assert set(DIALECT_CHOICES) == set(_DB_SCHEMES) - {"postgresql"}


@pytest.mark.parametrize("choice", DIALECT_CHOICES)
def test_every_choice_is_a_grammar_sqlglot_has(choice: str) -> None:
    """Through ``sql_policy``'s aliases: ``mssql`` is ``tsql`` to sqlglot."""
    sqlglot.parse_one("SELECT 1", dialect=_sqlglot_dialect(choice))


def test_sql_policy_allows_a_plain_select_in_sqlite() -> None:
    assert evaluate("SELECT 1", "sqlite").allowed


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("connector", "explicit", "expected"),
    [
        ({"db_type": "sqlite", "database": "/data/dev.sqlite"}, None, "sqlite"),
        ({"db_type": "duckdb"}, None, "duckdb"),
        ({"db_type": "snowflake"}, None, "snowflake"),
        # `connect postgresql` saves that spelling.
        ({"db_type": "postgresql"}, None, "postgres"),
        # Handed down as sqlglot spells it, which has no `mssql`.
        ({"db_type": "mssql"}, None, "tsql"),
        # The generic connector's type names no grammar; its URL does.
        ({"db_type": "sqlalchemy", "url": "mysql+pymysql://u@db:3306/shop"}, None, "mysql"),
        ({"db_type": "sqlalchemy", "url": "not a url"}, None, "postgres"),
        # A backend sqlglot has no grammar for would fail at the first parse.
        ({"db_type": "sqlalchemy", "url": "firebird://u@db/shop"}, None, "postgres"),
        # Nothing to go on: the default the option had before.
        (None, None, "postgres"),
        ({}, None, "postgres"),
        # The flag always wins.
        ({"db_type": "sqlite"}, "postgres", "postgres"),
        ({"db_type": "postgres"}, "sqlite", "sqlite"),
        (None, "mssql", "tsql"),
    ],
)
def test_the_dialect_follows_the_flag_then_the_connector_then_postgres(
    connector: dict[str, Any] | None, explicit: str | None, expected: str
) -> None:
    connectors = {} if connector is None else {AGENT: connector}
    with patch.object(cli_main, "_load_connectors", return_value=connectors):
        assert _resolve_dialect(AGENT, explicit) == expected


def test_an_underscored_id_finds_the_connector_the_loader_opens() -> None:
    """The loader also matches an id whose punctuation became underscores (the
    MCP form). Looked up with a plain ``.get``, that id found nothing and fell
    back to Postgres while the query ran against SQLite."""
    connectors = {"sqlite:/data/dev.sqlite": {"db_type": "sqlite"}}
    with patch.object(cli_main, "_load_connectors", return_value=connectors):
        assert _resolve_dialect("sqlite__data_dev.sqlite", None) == "sqlite"


@pytest.mark.parametrize(
    ("connector", "explicit", "warned"),
    [
        # The flag wins, but checking as Postgres what runs on SQL Server is said.
        ({"db_type": "mssql"}, "postgres", True),
        # `mssql` is `tsql`: the same grammar, nothing to say.
        ({"db_type": "mssql"}, "mssql", False),
        # No connector: nothing to disagree with, and nothing will run.
        (None, "sqlite", False),
        # A backend sqlglot has no grammar for names no dialect to compare.
        ({"db_type": "sqlalchemy", "url": "firebird://u@db/shop"}, "mysql", False),
        # No flag: the connector decides, so there is no disagreement.
        ({"db_type": "mssql"}, None, False),
    ],
)
def test_a_flag_that_disagrees_with_the_database_is_warned_about(
    connector: dict[str, Any] | None, explicit: str | None, warned: bool
) -> None:
    connectors = {} if connector is None else {AGENT: connector}
    with (
        patch.object(cli_main, "_load_connectors", return_value=connectors),
        patch.object(cli_main, "err_console") as err,
    ):
        _resolve_dialect(AGENT, explicit)

    printed = " ".join(str(c.args[0]) for c in err.print.call_args_list)
    assert ("is not the dialect of" in printed) is warned


def test_an_unreadable_connectors_file_does_not_stop_an_explicit_dialect() -> None:
    """With --dialect given, the file is read only for the disagreement warning;
    that it does not parse is no reason to refuse to generate SQL."""
    import click

    def unreadable() -> dict[str, Any]:
        raise click.ClickException("connectors.yaml does not parse")

    with (
        patch.object(cli_main, "_load_connectors", side_effect=unreadable),
        patch.object(cli_main, "err_console") as err,
    ):
        assert _resolve_dialect(AGENT, "snowflake") == "snowflake"

    printed = " ".join(str(c.args[0]) for c in err.print.call_args_list)
    assert "is not the dialect of" not in printed


def test_without_a_flag_an_unreadable_connectors_file_is_still_an_error() -> None:
    """The control: with no flag the file decides the dialect, so it must be read."""
    import click

    def unreadable() -> dict[str, Any]:
        raise click.ClickException("connectors.yaml does not parse")

    with (
        patch.object(cli_main, "_load_connectors", side_effect=unreadable),
        pytest.raises(click.ClickException),
    ):
        _resolve_dialect(AGENT, None)


def test_an_entry_that_is_not_a_mapping_names_no_dialect() -> None:
    """`agent-a: postgresql://host/db` is a plausible hand-edit. The loader
    reports it where it opens the connector; choosing a dialect must not raise
    first, which ``ask`` did before it had any connector lookup at all."""
    with patch.object(cli_main, "_load_connectors", return_value={AGENT: "postgresql://h/db"}):
        assert _resolve_dialect(AGENT, None) == "postgres"


# ---------------------------------------------------------------------------
# The three commands, with generation mocked
# ---------------------------------------------------------------------------


class _Recorder:
    """Records the dialect each generation entry point is handed."""

    def __init__(self) -> None:
        self.dialects: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def run_query_sync(self, question: str, agent_id: str, **kwargs: Any) -> Any:
        self.dialects.append(kwargs["dialect"])
        self.kwargs.append(kwargs)
        return SimpleNamespace(
            question=question,
            resolved_question=None,
            agent_type="sql",
            answer="",
            sql="SELECT 1",
            sql_is_valid=True,
            sql_result=None,
            citations=[],
            merged_answer=None,
            latency_ms=1,
            session_id=None,
            provenance=None,
        )

    def orchestrator(self) -> Any:
        recorder = self

        class _Orchestrator:
            def handle_question(
                self, question: str, agent_id: str, *, dialect: str
            ) -> AsyncIterator[str]:
                recorder.dialects.append(dialect)

                async def _tokens() -> AsyncIterator[str]:
                    yield "SELECT 1"

                return _tokens()

        return _Orchestrator


def _invoke(
    command: str,
    args: list[str],
    connectors: dict[str, Any],
    tmp_path: Path,
    agent: str = AGENT,
    recorder: _Recorder | None = None,
) -> Any:
    """Run *command* with generation mocked; returns (result, dialects handed down)."""
    recorder = recorder or _Recorder()
    (tmp_path / f"{agent}.yaml").write_text("{}", encoding="utf-8")

    def _run_eval(kb: Any, cases: Any, generate: Any, *, dialect: str) -> list[Any]:
        recorder.dialects.append(dialect)
        generate("How many schools are there?")
        return []

    argv = {
        "ask": ["ask", agent, "How many schools are there?"],
        "query": ["query", agent, "How many schools are there?", "--no-execute", "--no-session"],
        "eval": ["eval", agent],
    }[command]
    with (
        patch.object(cli_main, "_load_connectors", return_value=connectors),
        patch.object(cli_main, "KB_PATH", tmp_path),
        patch("nlqueries.orchestrator.sync_runner.run_query_sync", recorder.run_query_sync),
        patch("nlqueries.orchestrator.Orchestrator", recorder.orchestrator()),
        patch("nlqueries.kb_eval.build_cases", return_value=[object()]),
        patch("nlqueries.kb_eval.run_eval", _run_eval),
    ):
        result = CliRunner().invoke(cli, argv + args)
    return result, recorder.dialects


@pytest.mark.parametrize("command", ["ask", "query", "eval"])
def test_dialect_sqlite_is_accepted_and_reaches_generation(command: str, tmp_path: Path) -> None:
    result, dialects = _invoke(command, ["--dialect", "sqlite"], {}, tmp_path)
    assert result.exit_code == 0, result.output
    assert dialects and set(dialects) == {"sqlite"}


@pytest.mark.parametrize("command", ["ask", "query", "eval"])
def test_an_unknown_dialect_is_still_refused(command: str, tmp_path: Path) -> None:
    result, dialects = _invoke(command, ["--dialect", "notasql"], {}, tmp_path)
    assert result.exit_code == 2
    assert "Invalid value for '--dialect'" in result.output
    assert dialects == []


@pytest.mark.parametrize("command", ["ask", "query", "eval"])
def test_a_sqlite_connector_needs_no_flag(command: str, tmp_path: Path) -> None:
    sqlite = {AGENT: {"db_type": "sqlite", "database": "/data/dev.sqlite"}}
    result, dialects = _invoke(command, [], sqlite, tmp_path)
    assert result.exit_code == 0, result.output
    assert dialects and set(dialects) == {"sqlite"}


@pytest.mark.parametrize("command", ["ask", "query", "eval"])
def test_an_underscored_id_gets_its_connector_s_dialect(command: str, tmp_path: Path) -> None:
    sqlite = {"sqlite:/data/dev.sqlite": {"db_type": "sqlite"}}
    with patch.object(cli_main, "_resolve_alias", side_effect=lambda value: value):
        result, dialects = _invoke(command, [], sqlite, tmp_path, agent="sqlite__data_dev.sqlite")
    assert result.exit_code == 0, result.output
    assert dialects and set(dialects) == {"sqlite"}


@pytest.mark.parametrize("command", ["ask", "query", "eval"])
def test_with_nothing_known_the_default_is_still_postgres(command: str, tmp_path: Path) -> None:
    result, dialects = _invoke(command, [], {}, tmp_path)
    assert result.exit_code == 0, result.output
    assert dialects and set(dialects) == {"postgres"}


def test_eval_generates_without_the_semantic_cache(tmp_path: Path) -> None:
    """eval measures generation: a cached answer would measure an earlier model
    or prompt, and its questions do not belong in the agent's cache. query keeps
    the cache."""
    evaluated = _Recorder()
    result, _ = _invoke("eval", [], {}, tmp_path, recorder=evaluated)
    assert result.exit_code == 0, result.output
    assert [k.get("use_cache") for k in evaluated.kwargs] == [False]

    queried = _Recorder()
    result, _ = _invoke("query", [], {}, tmp_path, recorder=queried)
    assert result.exit_code == 0, result.output
    assert [k.get("use_cache", True) for k in queried.kwargs] == [True]
