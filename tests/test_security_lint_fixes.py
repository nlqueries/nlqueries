"""Behaviour changed while enabling ruff's security rules.

Each test pins one change: a URL with another scheme is not opened, at each
place a URL is opened; a failure that used to pass silently is reported; a SQL
fragment that was interpolated unchecked is checked.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import MagicMock

import pytest
from nlqueries.http_urls import NotAnHttpUrl, http_url

_FILE_URL = "file:///etc/passwd"


@pytest.fixture
def no_urlopen(monkeypatch) -> list[str]:
    """Every non-web URL that reaches urlopen. Web ones fail as unreachable,
    so the other checks in a report carry on."""
    opened: list[str] = []

    def record(request, *args, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        if not url.lower().startswith(("http://", "https://")):
            opened.append(url)
        raise urllib.error.URLError("no network in this test")

    monkeypatch.setattr(urllib.request, "urlopen", record)
    return opened


# --- the scheme guard -------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://qdrant:6333", "https://example.com/x", "HTTP://LOUD"])
def test_web_urls_pass(url: str) -> None:
    assert http_url(url) == url


@pytest.mark.parametrize(
    "url", [_FILE_URL, "ftp://example.com/x", "data:text/plain,hi", "qdrant:6333", "/etc/passwd"]
)
def test_other_schemes_are_refused(url: str) -> None:
    with pytest.raises(NotAnHttpUrl, match="not an http or https URL"):
        http_url(url)


def test_the_cli_qdrant_check_does_not_open_a_file_url(no_urlopen) -> None:
    from nlqueries.cli.main import _check_qdrant

    result = _check_qdrant(_FILE_URL)

    assert result.status == "fail"
    assert "not an http or https URL" in result.detail
    assert no_urlopen == []


def test_kb_stats_does_not_open_a_file_url(no_urlopen, monkeypatch, tmp_path, caplog) -> None:
    from nlqueries import config
    from nlqueries.knowledge import kb_stats

    monkeypatch.setattr(config, "QDRANT_URL", _FILE_URL)
    kb = tmp_path / "kb.yaml"
    kb.write_text("tables: []\n", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="nlqueries.knowledge.kb_stats"):
        stats = kb_stats.compute_kb_stats("sales", kb)

    assert no_urlopen == []
    assert stats.cache_entries is None  # not counted, rather than counted as none
    # Shown as n/a; Qdrant not running is ordinary, so it is not a warning.
    (record,) = [r for r in caplog.records if "cache entries" in r.getMessage()]
    assert record.levelno == logging.DEBUG


def test_the_mcp_health_line_does_not_open_a_file_url(no_urlopen, monkeypatch) -> None:
    from nlqueries import config
    from nlqueries.mcp_server import server

    monkeypatch.setattr(config, "QDRANT_URL", _FILE_URL)

    report = server.health()

    assert no_urlopen == []
    (line,) = [line for line in report.splitlines() if "**Qdrant**" in line]
    assert "not an http or https URL" in line


def test_the_embedding_client_does_not_open_a_file_url(no_urlopen) -> None:
    from nlqueries.embeddings import embedder

    with pytest.raises(embedder.EmbeddingServiceUnavailable, match="not an http or https URL"):
        embedder._daemon_send(urllib.request.Request(_FILE_URL))
    assert no_urlopen == []


# --- failures that passed silently ------------------------------------------------

_LEGACY_URL = "postgresql://sales:s3cret@db.internal:notaport/sales"


def _make_url_quoting_its_input(monkeypatch) -> None:
    """SQLAlchemy 2.0's refusal, which quotes the string it was given."""
    from sqlalchemy import engine
    from sqlalchemy.exc import ArgumentError

    def make_url(name: Any) -> Any:
        raise ArgumentError(f"Could not parse SQLAlchemy URL from string '{name}'")

    monkeypatch.setattr(engine, "make_url", make_url)


@pytest.mark.parametrize("module", ["nlqueries.connectors.loader", "nlqueries.cli.main"])
def test_a_stored_password_that_cannot_go_into_the_url_is_reported(
    module: str, monkeypatch, caplog
) -> None:
    """The connection would go out without it and fail as a bad password."""
    import importlib

    target = importlib.import_module(module)
    _make_url_quoting_its_input(monkeypatch)
    monkeypatch.setattr(target, "_load_password", lambda connector_id, cfg: "n3w")
    cfg = {"url": _LEGACY_URL, "password_storage": "keychain"}

    with caplog.at_level(logging.WARNING):
        url = target._get_full_url("sales", cfg)

    assert url == _LEGACY_URL
    (message,) = [
        r.getMessage()
        for r in caplog.records
        if "Could not add the stored password for sales" in r.getMessage()
    ]
    assert "ArgumentError" in message
    assert "s3cret" not in caplog.text  # the URL the error quotes stays out of the log


def test_a_password_left_in_the_url_after_moving_to_the_keychain_is_reported(monkeypatch) -> None:
    from click.testing import CliRunner
    from nlqueries.cli import main as cli_main

    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(cli_main, "_resolve_alias", lambda connector_id: connector_id)
    _make_url_quoting_its_input(monkeypatch)
    monkeypatch.setattr(cli_main, "_require_connector", lambda connector_id: {"url": _LEGACY_URL})
    monkeypatch.setattr(cli_main, "_save_password", lambda connector_id, password: True)
    monkeypatch.setattr(cli_main, "_save_connector", lambda connector_id, cfg: saved.append(cfg))
    monkeypatch.setenv("NEW_DB_PASSWORD", "n3w")

    result = CliRunner().invoke(
        cli_main.cli, ["update-password", "sales", "--password-env", "NEW_DB_PASSWORD"]
    )

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "could not be removed from the URL" in output
    assert "(ArgumentError)" in output
    assert "s3cret" not in output  # the password it is warning about


def test_kb_stats_reports_a_live_schema_it_could_not_read(tmp_path, caplog) -> None:
    from nlqueries.knowledge import kb_stats

    kb = tmp_path / "kb.yaml"
    kb.write_text("tables: []\n", encoding="utf-8")
    connector = MagicMock()
    connector.extract_schema.side_effect = RuntimeError("connection refused")

    with caplog.at_level(logging.WARNING, logger="nlqueries.knowledge.kb_stats"):
        kb_stats.compute_kb_stats("sales", kb, connector)

    assert any("live schema" in r.getMessage() for r in caplog.records)


def test_kb_stats_reports_feedback_it_could_not_count(tmp_path, monkeypatch, caplog) -> None:
    from nlqueries.feedback import store
    from nlqueries.knowledge import kb_stats

    def broken(agent_id):
        raise OSError("feedback file unreadable")

    monkeypatch.setattr(store, "load_feedback", broken)
    kb = tmp_path / "kb.yaml"
    kb.write_text("tables: []\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="nlqueries.knowledge.kb_stats"):
        kb_stats.compute_kb_stats("sales", kb)

    assert any("Could not count the feedback" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("raised", "warned"),
    [
        (ModuleNotFoundError("No module named 'snowflake'", name="snowflake"), False),
        # google-cloud-bigquery missing beside another google.cloud package.
        (ImportError("cannot import name 'bigquery'", name="google.cloud"), False),
        # Broken here rather than missing there: a renamed name, a circular import.
        (ImportError("cannot import name 'X'", name="nlqueries.connectors.base"), True),
        (ModuleNotFoundError("No module named 'nlqueries.x'", name="nlqueries.x"), True),
        (ImportError("names no module"), True),
        (RuntimeError("bug"), True),
    ],
    ids=["driver", "driver-in-namespace", "renamed", "own-module", "unnamed", "other"],
)
def test_an_optional_connector_that_fails_for_another_reason_is_reported(
    monkeypatch, caplog, raised: Exception, warned: bool
) -> None:
    """A missing driver is ordinary; any other failure is not left to vanish."""
    import importlib

    import nlqueries.connectors as connectors

    registry: dict[str, Any] = {}
    monkeypatch.setattr(connectors, "CONNECTOR_REGISTRY", registry)
    real_import = importlib.import_module

    def import_module(name: str, *args: Any) -> Any:
        if name == "nlqueries.connectors.snowflake":
            raise raised
        return real_import(name, *args)

    monkeypatch.setattr(connectors.importlib, "import_module", import_module)
    with caplog.at_level(logging.WARNING, logger="nlqueries.connectors"):
        connectors._register_optional_connectors()

    assert "snowflake" not in registry
    reported = [
        r for r in caplog.records if "snowflake connector could not be loaded" in r.getMessage()
    ]
    assert bool(reported) is warned


# --- SQL built from strings, and the rest -----------------------------------------


def test_the_sql_server_history_limit_is_an_integer_or_nothing_runs(monkeypatch) -> None:
    from nlqueries.connectors.mssql import MSSQLConnector

    connector = MSSQLConnector()
    executed: list[str] = []
    conn = MagicMock()

    def execute(statement, params=None):
        executed.append(str(statement))
        result = MagicMock()
        result.mappings.return_value.all.return_value = []
        return result

    conn.execute.side_effect = execute
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    monkeypatch.setattr(connector, "_require_engine", lambda: engine)

    connector.extract_query_history(limit="5")  # type: ignore[arg-type]
    assert "TOP 5" in executed[-1]

    executed.clear()
    with pytest.raises(Exception):  # noqa: B017 - whatever the connector wraps it in
        connector.extract_query_history(limit="5; DROP TABLE orders")  # type: ignore[arg-type]
    assert not any("DROP" in sql for sql in executed)


@pytest.mark.parametrize(
    ("location", "qualifier"),
    [("europe-west1", "region-europe-west1"), ("US", "region-us"), ("us`; DROP x; --", None)],
)
def test_a_bigquery_location_is_used_only_in_the_shape_of_one(location: str, qualifier) -> None:
    from nlqueries.connectors import bigquery

    connector = bigquery.BigQueryConnector()
    connector._dataset_id = "shop"
    connector._client = MagicMock()
    connector._client.get_dataset.return_value.location = location

    resolved = connector._resolve_region_qualifier()

    assert resolved == (qualifier or bigquery._DEFAULT_REGION_QUALIFIER)


def test_the_embed_handler_without_a_model_raises_even_without_asserts() -> None:
    from nlqueries.embeddings.embed_server import _EmbedHandler

    class Bare(_EmbedHandler):
        encoder = None
        model = None

    with pytest.raises(RuntimeError, match="neither encoder nor model is set"):
        Bare._encode_unguarded(Bare.__new__(Bare), ["text"])


def test_tasklist_is_run_by_its_full_path(monkeypatch) -> None:
    import ntpath
    import subprocess

    from nlqueries.embeddings import embed_server

    ran: list[list[str]] = []

    def run(args, **kwargs):
        ran.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="python.exe 4242 Console", stderr="")

    monkeypatch.setattr(embed_server.sys, "platform", "win32")
    monkeypatch.setattr(embed_server.subprocess, "run", run)

    assert embed_server.is_pid_alive(4242)
    ((program, *_rest),) = ran
    # A Windows path wherever the test runs, so judged as one: on Linux the
    # module joins it with "/", which Windows accepts and posixpath calls relative.
    assert ntpath.isabs(program)
    assert ntpath.normcase(program).endswith(ntpath.join("system32", "tasklist.exe"))
