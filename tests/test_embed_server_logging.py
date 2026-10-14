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
