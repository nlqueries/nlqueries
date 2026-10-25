"""LOG_LEVEL sets the log level of the CLI, the MCP server and the embedding daemon.

It was read into config and used by nothing, so setting it changed nothing.
Unset, each entry point keeps the behaviour it had: the MCP server and the
daemon log at INFO, and the CLI configures no logging at all.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from nlqueries import config
from nlqueries.cli import main as cli_main
from nlqueries.embeddings import embed_server


@pytest.fixture(autouse=True)
def _root_logger_restored() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _cli_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "nlqueries_cli", False)]


@pytest.mark.parametrize(
    ("value", "level", "unknown"),
    [
        ("", None, None),
        ("DEBUG", "DEBUG", None),
        ("WARNING", "WARNING", None),
        ("LOUD", None, "LOUD"),
    ],
)
def test_the_setting_names_a_level_or_is_reported(
    monkeypatch: pytest.MonkeyPatch, value: str, level: str | None, unknown: str | None
) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", value)
    assert config.log_level(None) == level
    assert config.log_level("INFO") == (level or "INFO")
    assert config.unknown_log_level() == unknown


def test_the_cli_configures_nothing_when_it_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", "")
    level = logging.getLogger().level

    cli_main._configure_logging()

    assert _cli_handlers() == []
    assert logging.getLogger().level == level


def test_the_cli_writes_records_at_the_level_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", "DEBUG")

    cli_main._configure_logging()
    logging.getLogger("nlqueries.orchestrator").debug("routing the question to sql")

    err = capsys.readouterr().err
    assert "DEBUG nlqueries.orchestrator: routing the question to sql" in err


def test_the_cli_drops_records_below_the_level(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", "ERROR")

    cli_main._configure_logging()
    logging.getLogger("nlqueries.orchestrator").warning("only a warning")

    assert "only a warning" not in capsys.readouterr().err


def test_the_cli_adds_its_handler_once_and_follows_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every command runs the group callback; in one process (tests, the
    library calling the CLI) that must not stack handlers, and a later run's
    stderr is the one written to."""
    monkeypatch.setattr(config, "LOG_LEVEL", "INFO")

    cli_main._configure_logging()
    capsys.readouterr()
    cli_main._configure_logging()
    logging.getLogger("nlqueries.cli").info("second run")

    assert len(_cli_handlers()) == 1
    assert capsys.readouterr().err.count("second run") == 1


def test_the_cli_warns_about_a_value_it_ignores(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", "LOUD")

    with caplog.at_level(logging.WARNING):
        cli_main._configure_logging()

    assert "LOG_LEVEL=LOUD is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL" in caplog.text
    assert _cli_handlers() == []


def test_every_command_runs_the_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner  # noqa: PLC0415

    calls: list[bool] = []
    monkeypatch.setattr(cli_main, "_configure_logging", lambda: calls.append(True))

    result = CliRunner().invoke(cli_main.cli, ["mcp-server", "--help"])

    assert result.exit_code == 0, result.output
    assert calls == [True]


@pytest.mark.parametrize(
    ("value", "expected"), [("", "INFO"), ("DEBUG", "DEBUG"), ("LOUD", "INFO")]
)
def test_the_mcp_server_hands_the_level_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str
) -> None:
    from nlqueries.mcp_server import server  # noqa: PLC0415

    monkeypatch.setattr(config, "LOG_LEVEL", value)

    assert server._build_server().settings.log_level == expected


def test_the_mcp_server_warns_about_a_value_it_ignores(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from nlqueries.mcp_server import server  # noqa: PLC0415

    monkeypatch.setattr(config, "LOG_LEVEL", "LOUD")
    monkeypatch.setattr(server.mcp, "run", lambda transport: None)

    with caplog.at_level(logging.WARNING):
        server.main(transport="stdio")

    assert "LOG_LEVEL=LOUD is not one of" in caplog.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", logging.INFO), ("WARNING", logging.WARNING), ("LOUD", logging.INFO)],
)
def test_the_daemon_logs_at_the_level(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", value)
    monkeypatch.delenv(embed_server.LOG_FORMAT_ENV, raising=False)

    embed_server.configure_logging()

    assert logging.getLogger().level == expected


def test_the_daemon_warns_about_a_value_it_ignores(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "LOG_LEVEL", "LOUD")
    monkeypatch.delenv(embed_server.LOG_FORMAT_ENV, raising=False)

    embed_server.configure_logging()

    assert "LOG_LEVEL=LOUD is not one of" in capsys.readouterr().err
