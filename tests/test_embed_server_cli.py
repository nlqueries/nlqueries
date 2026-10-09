"""``embed-server start``, ``stop`` and ``status`` against a PID file left behind.

A daemon that dies uncleanly -- a reboot, a killed terminal -- leaves its PID
file. ``status`` checked the PID; the other two did not. ``start`` took the
file's existence to mean a daemon was running and refused, and ``stop`` sent
SIGTERM to the dead PID, which on Windows raises ``OSError`` (winerror 87)
rather than ``ProcessLookupError`` and crashed with a traceback. All three now
read the file through ``embed_server.read_pid_file``.

No real daemon: the PID file is under ``tmp_path``, and ``is_pid_alive``,
``os.kill`` and ``subprocess.Popen`` are replaced.
"""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import cli
from nlqueries.embeddings import embed_server
from rich.console import Console


@pytest.fixture
def pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "embed-server.pid"
    monkeypatch.setattr(embed_server, "_PID_FILE", path)
    return path


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    """Both consoles, wide enough that no message wraps mid-assertion."""
    buffer = io.StringIO()
    monkeypatch.setattr(cli_main, "console", Console(file=buffer, width=300))
    monkeypatch.setattr(cli_main, "err_console", Console(file=buffer, width=300))
    yield buffer


def _run(*args: str) -> Any:
    return CliRunner().invoke(cli, ["embed-server", *args])


class _Daemon:
    """A process that is alive until it is sent SIGTERM."""

    def __init__(self, *, exits_on_sigterm: bool = True) -> None:
        self.alive = True
        self.exits_on_sigterm = exits_on_sigterm
        self.signals: list[tuple[int, int]] = []

    def is_pid_alive(self, pid: int) -> bool:
        return self.alive

    def kill(self, pid: int, sig: int) -> None:
        self.signals.append((pid, sig))
        if self.exits_on_sigterm:
            self.alive = False


# --- The reported case: a PID file whose process is gone -----------------------


def test_stop_reports_a_stale_pid_file_and_removes_it(pid_file: Path, out: io.StringIO) -> None:
    pid_file.write_text("105552")
    kill = MagicMock()
    with patch.object(embed_server, "is_pid_alive", return_value=False), patch("os.kill", kill):
        result = _run("stop")

    assert result.exit_code == 0, out.getvalue()
    assert "Removing stale PID file (process 105552 not found)" in out.getvalue()
    assert "Daemon is not running." in out.getvalue()
    assert not pid_file.exists()
    kill.assert_not_called()


def test_start_removes_a_stale_pid_file_and_starts(pid_file: Path, out: io.StringIO) -> None:
    pid_file.write_text("105552")
    popen = MagicMock()
    with (
        patch.object(embed_server, "is_pid_alive", return_value=False),
        patch("subprocess.Popen", popen),
    ):
        result = _run("start")

    assert result.exit_code == 0, out.getvalue()
    assert "Removing stale PID file (process 105552 not found)" in out.getvalue()
    assert "Embedding daemon started" in out.getvalue()
    assert not pid_file.exists()
    popen.assert_called_once()


def test_start_still_refuses_while_a_daemon_runs(pid_file: Path, out: io.StringIO) -> None:
    """The negative control: a live PID is still "already running"."""
    pid_file.write_text("4242")
    popen = MagicMock()
    with (
        patch.object(embed_server, "is_pid_alive", return_value=True),
        patch("subprocess.Popen", popen),
    ):
        result = _run("start")

    assert result.exit_code == 0
    assert "Daemon already running (PID 4242)" in out.getvalue()
    assert pid_file.read_text() == "4242"
    popen.assert_not_called()


def test_status_reports_a_stale_pid_file_and_removes_it(pid_file: Path, out: io.StringIO) -> None:
    pid_file.write_text("105552")
    with patch.object(embed_server, "is_pid_alive", return_value=False):
        result = _run("status")

    assert result.exit_code == 0
    assert "Removing stale PID file (process 105552 not found)" in out.getvalue()
    assert "not running" in out.getvalue()
    assert not pid_file.exists()


# --- Stopping a live daemon ------------------------------------------------------


def test_stop_waits_for_the_daemon_and_reports_it_stopped(pid_file: Path, out: io.StringIO) -> None:
    pid_file.write_text("4242")
    daemon = _Daemon()
    with (
        patch.object(embed_server, "is_pid_alive", daemon.is_pid_alive),
        patch("os.kill", daemon.kill),
    ):
        result = _run("stop")

    assert result.exit_code == 0, out.getvalue()
    assert "Daemon stopped (PID 4242)" in out.getvalue()
    assert daemon.signals == [(4242, signal.SIGTERM)]
    assert not pid_file.exists()


@pytest.mark.parametrize(
    "error",
    [OSError(87, "The parameter is incorrect"), PermissionError(5, "Access is denied")],
)
def test_a_daemon_that_cannot_be_signalled_is_one_line_not_a_traceback(
    pid_file: Path, out: io.StringIO, error: OSError
) -> None:
    """What Windows raises for a PID it will not terminate. ProcessLookupError
    was the only one caught, so these crashed the command."""
    pid_file.write_text("4242")
    with (
        patch.object(embed_server, "is_pid_alive", return_value=True),
        patch("os.kill", side_effect=error),
    ):
        result = _run("stop")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    text = out.getvalue() + result.output
    assert f"Could not stop the daemon (PID 4242): {error.strerror}" in text
    assert "Traceback" not in text
    # Still running, so still recorded: start and status read it.
    assert pid_file.read_text() == "4242"


def test_a_daemon_that_exits_before_it_is_signalled_is_not_running(
    pid_file: Path, out: io.StringIO
) -> None:
    """Alive when the file was read, gone by the time of the signal."""
    pid_file.write_text("4242")
    alive = iter([True, False])
    with (
        patch.object(embed_server, "is_pid_alive", side_effect=lambda pid: next(alive)),
        patch("os.kill", side_effect=OSError(87, "The parameter is incorrect")),
    ):
        result = _run("stop")

    assert result.exit_code == 0, out.getvalue()
    assert "Daemon is not running (process 4242 exited before it was stopped)" in out.getvalue()
    assert not pid_file.exists()


def test_a_daemon_that_outlives_the_wait_is_reported(
    pid_file: Path, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_main, "_EMBED_STOP_WAIT_SECONDS", 0.0)
    pid_file.write_text("4242")
    daemon = _Daemon(exits_on_sigterm=False)
    with (
        patch.object(embed_server, "is_pid_alive", daemon.is_pid_alive),
        patch("os.kill", daemon.kill),
    ):
        result = _run("stop")

    assert result.exit_code == 1
    assert "Daemon (PID 4242) did not exit within 0 s of being signalled" in out.getvalue()
    assert not pid_file.exists()


# --- A PID file that does not hold a PID ------------------------------------------


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("", "it is empty"),
        ("not-a-pid", "it holds 'not-a-pid', not a PID"),
        # Parses, but no daemon has it. tasklist lists PID 0 (System Idle
        # Process) and os.kill(0, 0) signals our own group, so the liveness
        # check -- patched to say "alive" here -- must not be asked.
        ("0", "it holds '0', not a PID"),
        ("-3", "it holds '-3', not a PID"),
    ],
)
@pytest.mark.parametrize("command", ["start", "stop", "status"])
def test_a_pid_file_without_a_pid_is_stale_to_every_command(
    pid_file: Path, out: io.StringIO, content: str, reason: str, command: str
) -> None:
    pid_file.write_text(content)
    kill = MagicMock()
    popen = MagicMock()
    with (
        patch.object(embed_server, "is_pid_alive", return_value=True),
        patch("os.kill", kill),
        patch("subprocess.Popen", popen),
    ):
        result = _run(command)

    assert result.exit_code == 0, out.getvalue()
    assert f"Removing stale PID file ({reason})" in out.getvalue()
    assert not pid_file.exists()
    kill.assert_not_called()
    assert popen.called == (command == "start")


@pytest.mark.parametrize("pid", [0, -1])
def test_no_pid_below_one_is_alive(pid: int) -> None:
    with (
        patch.object(subprocess, "run", side_effect=AssertionError("tasklist asked")),
        patch.object(os, "kill", side_effect=AssertionError("os.kill asked")),
    ):
        assert embed_server.is_pid_alive(pid) is False


# --- The daemon's own SIGTERM handler --------------------------------------------


def test_sigterm_stops_the_daemon(tmp_path: Path) -> None:
    """`server.shutdown()` waits for `serve_forever` to return, and the handler
    runs on the thread inside `serve_forever`: called there directly, it waited
    forever, so SIGTERM never stopped the daemon. In a child process, because
    a regression hangs rather than fails. Signal handlers run on the main thread
    on every platform, so raising SIGTERM in-process takes the same path."""
    child = textwrap.dedent(
        """
        import signal, sys, threading
        from pathlib import Path
        from nlqueries.embeddings import embed_server

        embed_server._PID_FILE = Path(sys.argv[1])
        embed_server._load_encoder = lambda backend: (lambda texts: [[0.0] * 384 for _ in texts])
        threading.Timer(1.0, lambda: signal.raise_signal(signal.SIGTERM)).start()
        embed_server.serve(port=0, backend="onnx")
        print("served and stopped")
        """
    )
    pid_path = tmp_path / "embed-server.pid"

    done = subprocess.run(
        [sys.executable, "-c", child, str(pid_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr
    assert "served and stopped" in done.stdout
    assert not pid_path.exists(), "serve() removes its own PID file on the way out"


# --- A tasklist that does not answer in time (from #235's review) ----------------


def test_a_slow_tasklist_is_taken_as_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown, so assumed running: guessing "gone" could remove a live
    daemon's PID file. TimeoutExpired is not an OSError, so it used to escape."""
    monkeypatch.setattr(sys, "platform", "win32")
    with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("tasklist", 5)):
        assert embed_server.is_pid_alive(4242) is True


def test_stop_survives_a_slow_tasklist(
    pid_file: Path, out: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real `is_pid_alive`, which `stop` calls on every poll."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli_main, "_EMBED_STOP_WAIT_SECONDS", 0.0)
    pid_file.write_text("4242")
    with (
        patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("tasklist", 5)),
        patch("os.kill"),
    ):
        result = _run("stop")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in out.getvalue() + result.output
    assert "Daemon (PID 4242) did not exit" in out.getvalue()
