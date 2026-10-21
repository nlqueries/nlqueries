"""Driver and connectors-file text stays out of what a user is shown.

A driver's error carries the statement, its parameters and sometimes the
connection string on the lines after its first; PyYAML's parse error quotes the
line it stopped on. Either can hold a password or a host. The MCP ``query`` tool
shows the first line and a reference id, with the full text in the server log;
a connectors file that does not parse is reported by line number and problem.
"""

from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from nlqueries.cli.main import cli

SECRET = "hunter2"
HOST = "10.20.30.40"
DRIVER_ERROR = (
    "(psycopg.OperationalError) could not run the statement\n"
    f"DETAIL: dsn host={HOST} user=app password={SECRET}"
)
REFERENCE = re.compile(r"\(reference ([0-9a-f]{8})\)")

# The fourth line is not valid YAML, and it is the line holding the password.
MALFORMED = f"good:\n  db_type: postgres\n  host: h\nbad: password: {SECRET}\n"


def _result_with_error(error: str) -> MagicMock:
    sql_result = MagicMock()
    sql_result.error = error
    result = MagicMock()
    result.answer = ""
    result.agent_type = "sql"
    result.sql = "SELECT 1"
    result.sql_result = sql_result
    result.citations = []
    result.latency_ms = 1
    result.provenance = None
    return result


def _referenced_in_log(out: str, caplog: pytest.LogCaptureFixture) -> None:
    """*out* names a reference; the WARNING with it holds the first line only,
    and the full text is under the same reference at DEBUG."""
    match = REFERENCE.search(out)
    assert match, out
    for record in caplog.records:
        if record.levelno >= logging.WARNING:
            text = record.getMessage()
            assert "password=" not in text and SECRET not in text and HOST not in text, text
    with_ref = [r for r in caplog.records if match.group(1) in r.getMessage()]
    warning = [r for r in with_ref if r.levelno == logging.WARNING]
    assert len(warning) == 1 and "could not run the statement" in warning[0].getMessage()
    debug = [r.getMessage() for r in with_ref if r.levelno == logging.DEBUG]
    assert len(debug) == 1 and SECRET in debug[0] and HOST in debug[0]


# --- The MCP query tool --------------------------------------------------------


@pytest.mark.anyio
async def test_a_database_error_is_shown_without_what_the_driver_quoted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nlqueries.mcp_server.server import query

    result = _result_with_error(DRIVER_ERROR)
    with (
        patch(
            "nlqueries.orchestrator.sync_runner.run_query",
            new_callable=AsyncMock,
            return_value=result,
        ),
        caplog.at_level(logging.DEBUG, logger="nlqueries.mcp_server.server"),
    ):
        out = await query("How many orders?", "sales")

    assert "could not run the statement" in out
    assert "password=" not in out and SECRET not in out and HOST not in out
    _referenced_in_log(out, caplog)
    # Only the reply is cut down: the result keeps the raw text the correction
    # loop and the not-permitted check read.
    assert result.sql_result.error == DRIVER_ERROR


@pytest.mark.anyio
async def test_a_failed_query_is_shown_without_what_the_driver_quoted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nlqueries.mcp_server.server import query

    with (
        patch(
            "nlqueries.orchestrator.sync_runner.run_query",
            new_callable=AsyncMock,
            side_effect=RuntimeError(DRIVER_ERROR),
        ),
        caplog.at_level(logging.WARNING, logger="nlqueries.mcp_server.server"),
    ):
        out = await query("How many orders?", "sales")

    assert out.startswith("Query failed: RuntimeError: (psycopg.OperationalError)")
    assert "password=" not in out and SECRET not in out and HOST not in out
    match = REFERENCE.search(out)
    assert match, out
    # `_log.exception`: the full error is in the record's traceback.
    (record,) = [r for r in caplog.records if match.group(1) in r.getMessage()]
    assert record.exc_info is not None
    assert SECRET in str(record.exc_info[1]) and HOST in str(record.exc_info[1])


# --- A connectors file that does not parse -------------------------------------


@pytest.fixture
def malformed_file(tmp_path: Path) -> Path:
    f = tmp_path / "connectors.yaml"
    f.write_text(MALFORMED, encoding="utf-8")
    return f


def test_the_mcp_listing_names_the_line_not_its_text(malformed_file: Path) -> None:
    from nlqueries.mcp_server.server import list_connectors

    with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", malformed_file):
        out = list_connectors()

    assert out == "Failed to read connectors file: line 4: mapping values are not allowed here"


def test_the_loader_names_the_line_not_its_text(malformed_file: Path) -> None:
    from nlqueries.connectors.loader import ConnectorsFileError, _load_connectors

    with (
        patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", malformed_file),
        pytest.raises(ConnectorsFileError) as refused,
    ):
        _load_connectors()

    assert str(refused.value) == (
        "Failed to read connectors file: line 4: mapping values are not allowed here"
    )
    # Not chained to PyYAML's error either, which would print the line in any
    # traceback logged for this.
    assert SECRET not in "".join(traceback.format_exception(refused.value))


def test_the_cli_names_the_line_not_its_text(malformed_file: Path) -> None:
    with patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", malformed_file):
        result = CliRunner().invoke(cli, ["connectors"])

    assert result.exit_code == 1, result.output
    # A ClickException's exit, not the parse error escaping as a traceback.
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Failed to read connectors file: line 4: mapping values are not allowed here" in (
        result.output
    )
    assert SECRET not in result.output


def test_health_reports_the_file_and_runs_the_other_checks(malformed_file: Path) -> None:
    """A file that does not parse is one failed check, not the end of the run."""
    import json

    from nlqueries.cli.main import _CheckResult

    others = {
        name: MagicMock(return_value=_CheckResult(service, "ok", "up"))
        for name, service in [
            ("_check_qdrant", "Qdrant"),
            ("_check_llm", "LLM"),
            ("_check_embedding", "Embedding"),
            ("_check_config", "Config"),
        ]
    }
    with (
        patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", malformed_file),
        patch.multiple("nlqueries.cli.main", **others),
    ):
        result = CliRunner().invoke(cli, ["health", "--json"])

    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    by_service = {c["service"]: c for c in report["checks"]}
    assert set(by_service) == {"Qdrant", "Database", "LLM", "Embedding", "Config"}
    database = by_service["Database"]
    assert database["status"] == "fail"
    assert database["detail"].startswith(
        "Failed to read connectors file: line 4: mapping values are not allowed here"
    )
    assert report["healthy"] is False
    assert SECRET not in result.output


def test_the_mcp_listing_reads_the_file_through_the_loader(tmp_path: Path) -> None:
    """One reader: whatever the loader refuses, the listing reports the same way."""
    from nlqueries.connectors.loader import ConnectorsFileError
    from nlqueries.mcp_server.server import list_connectors

    # Valid on its own, so only the loader's refusal can produce the message.
    f = tmp_path / "connectors.yaml"
    f.write_text("good:\n  db_type: postgres\n", encoding="utf-8")
    with (
        patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f),
        patch(
            "nlqueries.connectors.loader._read_connectors_root",
            side_effect=ConnectorsFileError("line 9: a problem the loader found"),
        ),
    ):
        out = list_connectors()

    assert out == "Failed to read connectors file: line 9: a problem the loader found"


# --- A hybrid answer built from a failed query ---------------------------------


def _sql_frame_that_failed() -> list[str]:
    """The SQL sub-agent's tokens when the connector raised on execution."""
    import json

    payload = {
        "type": "sql",
        "sql": "SELECT region, revenue FROM sales",
        "is_valid": True,
        "execution_mode": "execute",
        "validation_error": None,
        "dialect": "postgres",
        "attempt_count": 1,
        "sql_table": {"error": DRIVER_ERROR},
    }
    return ["Answering.", json.dumps(payload)]


@pytest.mark.anyio
async def test_a_hybrid_answer_from_a_failed_query_is_shown_without_the_driver_text() -> None:
    """The document side finds nothing, so the SQL-only passthrough is the answer.

    That passthrough is the formatted table, and the table's failure note is the
    first thing in the MCP reply. Driven through the real orchestrator and
    merger; only the sub-agents, the classifier and the cache are stubbed.
    """
    import json

    from nlqueries.mcp_server.server import query
    from nlqueries.orchestrator import sync_runner
    from nlqueries.orchestrator.intent_classifier import (
        IntentClassificationResult,
        IntentType,
    )
    from nlqueries.orchestrator.multi_agent_orchestrator import drain_background_tasks

    def _agent(tokens: list[str]) -> MagicMock:
        async def _gen(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
            for t in tokens:
                yield t

        agent = MagicMock()
        agent.handle_question = _gen
        return agent

    doc_tokens = ["Nothing found.", json.dumps({"type": "citations", "citations": []})]
    cache = MagicMock()
    cache.get.return_value = None
    real_run_query = sync_runner.run_query

    async def _hybrid_run_query(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        return await real_run_query(*args, available_types=("sql", "document", "hybrid"), **kwargs)

    m = "nlqueries.orchestrator.multi_agent_orchestrator"
    with (
        patch(f"{m}.SemanticCache", return_value=cache),
        patch(
            f"{m}.aclassify_intent",
            return_value=IntentClassificationResult(
                intent=IntentType.hybrid, confidence=0.95, reasoning="Mock."
            ),
        ),
        patch(f"{m}.Orchestrator", return_value=_agent(_sql_frame_that_failed())),
        patch(f"{m}.DocumentOrchestrator", return_value=_agent(doc_tokens)),
        patch("nlqueries.orchestrator.sync_runner.run_query", _hybrid_run_query),
    ):
        out = await query("Revenue by region and the policy on it?", "sales")
        await drain_background_tasks()

    assert "hybrid agent" in out, out  # the route this test is about was taken
    assert "this query failed: (psycopg.OperationalError) could not run the statement" in out
    assert "password=" not in out and SECRET not in out and HOST not in out


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(["sql_query"], id="statement-table"),  # the note after the table
        pytest.param([], id="no-columns"),  # the "(no data; ...)" form
    ],
)
def test_the_combined_answer_prompt_carries_no_driver_text(columns: list[str]) -> None:
    from nlqueries.connectors.base import QueryResult
    from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult
    from nlqueries.orchestrator.result_merger import merge_results

    failed = QueryResult(
        columns=columns,
        rows=[["SELECT region FROM sales"]] if columns else [],
        row_count=1 if columns else 0,
        execution_time_ms=0.0,
        error=DRIVER_ERROR,
    )
    citation = Citation(
        chunk_id="c1",
        source_name="policy.pdf",
        page_number=1,
        chunk_index=0,
        excerpt="Revenue is reported quarterly.",
        relevance_score=0.9,
    )
    documents = DocumentRetrievalResult(chunks=[], citations=[citation], collection="doc_x")
    llm = MagicMock()
    llm.complete.return_value = "Combined."
    with patch("nlqueries.orchestrator.result_merger.get_llm_client", return_value=llm):
        merged = merge_results("Revenue?", sql_result=failed, document_result=documents)

    (_system, user), _ = llm.complete.call_args
    assert "this query failed: (psycopg.OperationalError) could not run the statement" in user
    assert "password=" not in user and SECRET not in user and HOST not in user
    # The result keeps the raw text; only what the model is shown is cut down.
    assert merged.sql_table is not None and merged.sql_table.error == DRIVER_ERROR


# --- The connectors file is read as UTF-8 --------------------------------------

UTF8_PASSWORD = "pässwörd-ß"


@pytest.fixture
def cp1252_locale():  # type: ignore[no-untyped-def]
    """`read_text()` and `write_text()` with no encoding behave as on a cp1252
    Windows locale.

    The container's own locale is UTF-8, where a missing encoding goes unnoticed.
    """
    real_read, real_write = Path.read_text, Path.write_text

    def read_text(self: Path, encoding: str | None = None, *args: object, **kwargs: object) -> str:
        return real_read(self, encoding or "cp1252", *args, **kwargs)  # type: ignore[arg-type]

    def write_text(
        self: Path, data: str, encoding: str | None = None, *args: object, **kwargs: object
    ) -> int:
        return real_write(self, data, encoding or "cp1252", *args, **kwargs)  # type: ignore[arg-type]

    with patch.object(Path, "read_text", read_text), patch.object(Path, "write_text", write_text):
        yield


def test_a_non_ascii_password_is_read_as_written(tmp_path: Path, cp1252_locale: None) -> None:
    from nlqueries.connectors.loader import _load_connectors

    f = tmp_path / "connectors.yaml"
    f.write_bytes(f"db:\n  db_type: postgres\n  password: {UTF8_PASSWORD}\n".encode())
    with patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", f):
        assert _load_connectors()["db"]["password"] == UTF8_PASSWORD


@pytest.fixture
def cp1252_file(tmp_path: Path) -> Path:
    f = tmp_path / "connectors.yaml"
    f.write_bytes("db:\n  db_type: postgres\n  password: hunter2-pässwörd\n".encode("cp1252"))
    return f


# 45 bytes precede the cp1252 "ä": "db:\n  db_type: postgres\n  password: hunter2-p".
NOT_UTF8 = (
    "Failed to read connectors file: it is not UTF-8 (the first invalid byte is at offset 45)"
)


def test_the_mcp_listing_says_a_cp1252_file_is_not_utf8(cp1252_file: Path) -> None:
    from nlqueries.mcp_server.server import list_connectors

    with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", cp1252_file):
        out = list_connectors()

    assert out == NOT_UTF8
    assert "0xe4" not in out and SECRET not in out


def test_the_cli_says_a_cp1252_file_is_not_utf8(cp1252_file: Path) -> None:
    with patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", cp1252_file):
        result = CliRunner().invoke(cli, ["connectors"])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert NOT_UTF8 in result.output
    assert "0xe4" not in result.output and SECRET not in result.output


def test_a_saved_connector_is_written_as_utf8(tmp_path: Path, cp1252_locale: None) -> None:
    """The writer states the encoding every reader now uses.

    `yaml.dump` escapes non-ASCII by default, so today's file is ASCII either
    way. This dumps with `allow_unicode=True`, as the knowledge-base writers in
    the same module do: written in the locale's cp1252, no reader could open it.
    """
    import yaml
    from nlqueries.cli import main as cli_main
    from nlqueries.connectors.loader import _load_connectors

    f = tmp_path / "connectors.yaml"
    real_dump = yaml.dump

    def dump_unicode(*args: object, **kwargs: object) -> str:
        return real_dump(*args, allow_unicode=True, **kwargs)  # type: ignore[no-any-return, call-overload]

    with (
        patch.object(cli_main, "CONNECTORS_FILE", f),
        patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", f),
        patch.object(cli_main.yaml, "dump", dump_unicode),
    ):
        cli_main._save_connector("db", {"db_type": "postgres", "password": UTF8_PASSWORD})
        assert _load_connectors()["db"]["password"] == UTF8_PASSWORD
