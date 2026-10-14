"""The embed server's log lines: plain text by default, JSON lines when asked.

Inside the nlqueries-enterprise API container this daemon runs beside the API,
whose support tooling reads JSON lines with the keys ``asctime``, ``name``,
``levelname``, ``message`` and ``exc_info``. The daemon's plain-text lines,
"Embedding request failed" tracebacks included, never reached it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from click.testing import CliRunner
from nlqueries.cli.main import cli
from nlqueries.embeddings import embed_server


@pytest.fixture(autouse=True)
def _root_logger_restored() -> Iterator[None]:
    """configure_logging replaces the root logger's handlers, pytest's own
    capture among them; put them back."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _log_a_failure() -> None:
    log = logging.getLogger("nlqueries.embeddings.embed_server")
    try:
        raise RuntimeError("the model is not loaded")
    except RuntimeError:
        log.exception("Embedding request failed")


def test_json_lines_carry_the_keys_the_api_lines_carry(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(embed_server.LOG_FORMAT_ENV, raising=False)
    embed_server.configure_logging("json")

    _log_a_failure()

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["name"] == "nlqueries.embeddings.embed_server"
    assert record["levelname"] == "ERROR"
    assert record["message"] == "Embedding request failed"
    assert "RuntimeError: the model is not loaded" in record["exc_info"]
    # The API's own format: 2026-10-14T12:37:19+0530.
    assert len(record["asctime"]) == 24 and record["asctime"][10] == "T"


def test_the_environment_asks_for_json_when_the_flag_does_not(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enterprise entrypoint sets the variable: an older core ignores it,
    where an unknown flag would stop the daemon starting."""
    monkeypatch.setenv(embed_server.LOG_FORMAT_ENV, "json")
    embed_server.configure_logging()

    _log_a_failure()

    assert json.loads(capsys.readouterr().err.strip().splitlines()[-1])["levelname"] == "ERROR"


def test_plain_text_is_the_default_and_unchanged(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(embed_server.LOG_FORMAT_ENV, raising=False)
    embed_server.configure_logging()

    _log_a_failure()

    err = capsys.readouterr().err
    # What logging.basicConfig wrote before, traceback and all.
    assert err.startswith("ERROR:nlqueries.embeddings.embed_server:Embedding request failed")
    assert "Traceback" in err


def test_an_unknown_format_is_text_and_says_so(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(embed_server.LOG_FORMAT_ENV, "xml")
    embed_server.configure_logging()

    err = capsys.readouterr().err
    assert err.startswith("WARNING:nlqueries.embeddings.embed_server:Unknown log format 'xml'")


def test_the_command_line_configures_logging_before_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(embed_server, "configure_logging", lambda fmt: calls.append(("log", fmt)))
    monkeypatch.setattr(
        embed_server, "serve", lambda port, backend: calls.append(("serve", (port, backend)))
    )

    embed_server.main(["--port", "9999", "--log-format", "json"])

    assert calls == [("log", "json"), ("serve", (9999, None))]


def _escape_a_handler(exc: BaseException) -> None:
    """Have *exc* escape a request handler on a real server."""
    server = embed_server._ThreadingEmbedServer(("127.0.0.1", 0), embed_server._EmbedHandler)
    try:
        try:
            raise exc
        except BaseException:
            server.handle_error(None, ("127.0.0.1", 50123))
    finally:
        server.server_close()


def test_a_handler_that_fails_is_an_error_in_the_chosen_format(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """socketserver prints an exception escaping a handler to stderr itself,
    past the logging configuration, which in JSON mode broke the
    one-object-per-line stream (from #261's review)."""
    monkeypatch.delenv(embed_server.LOG_FORMAT_ENV, raising=False)
    embed_server.configure_logging("json")

    _escape_a_handler(RuntimeError("the encode pool is gone"))

    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["levelname"] == "ERROR"
    assert "RuntimeError: the encode pool is gone" in record["exc_info"]


@pytest.mark.parametrize(
    "exc",
    [ConnectionResetError("reset"), BrokenPipeError("broken pipe"), TimeoutError("timed out")],
    ids=["reset", "broken-pipe", "timeout"],
)
def test_a_client_that_left_is_routine(
    exc: BaseException, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embedder gives up after its timeout and embeds in-process; the
    reply's write then fails. One INFO line, no traceback, so the stream that
    support reads for failures does not fill with it (from #262's review)."""
    monkeypatch.delenv(embed_server.LOG_FORMAT_ENV, raising=False)
    embed_server.configure_logging("json")

    _escape_a_handler(exc)

    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["levelname"] == "INFO"
    assert "exc_info" not in record
    assert record["message"].endswith(f"left before its reply was written ({type(exc).__name__})")


def test_the_foreground_command_logs_like_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nlqueries embed-server start --foreground` served without configuring
    logging, so nothing below WARNING reached the terminal (from #261's review)."""
    calls: list[str] = []
    monkeypatch.setattr(embed_server, "read_pid_file", lambda: (None, None))
    monkeypatch.setattr(embed_server, "configure_logging", lambda *a: calls.append("log"))
    monkeypatch.setattr(embed_server, "serve", lambda port: calls.append(f"serve:{port}"))

    result = CliRunner().invoke(cli, ["embed-server", "start", "--foreground", "--port", "9123"])

    assert result.exit_code == 0, result.output
    assert calls == ["log", "serve:9123"]
