"""Everything NLQueries keeps between runs lives under one configurable root.

``~/.nlqueries`` was written out in seven places across four modules, three of
them with no override at all: the embed server's pid file, the cache signing key
and the CLI's session transcripts. A deployment that cannot write to a home
directory could not be configured out of it.

That is not hypothetical. Running the containers with a read-only root killed
the embed server at startup with ``[Errno 30]`` — and with it every
natural-language query, since the API requires the daemon — because the pid file
had nowhere to go. The workaround was a tmpfs mounted at the home directory,
which worked only because it happened to cover the other two as well.

These tests read the paths through a reimported ``config``, because core freezes
its configuration in module-level ``os.getenv`` calls at import: setting the
variable without reloading proves nothing about a real process.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _config_with(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Reimport `nlqueries.config` with *env* applied."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import nlqueries.config as config_module

    return importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _restore_config():
    """Leave the real configuration in place for every other test in the run.

    Every module reloaded here has to be reloaded back, not just `config`:
    `embed_server` binds `_PID_FILE` at import, so a test that reloads it under
    a `tmp_path` leaves that module pointed at a directory pytest then deletes.
    Nothing fails today only because the default collection order happens to run
    the embed-server tests first.
    """
    yield
    import nlqueries.config as config_module

    importlib.reload(config_module)
    import nlqueries.embeddings.embed_server as embed_server_module

    importlib.reload(embed_server_module)


def test_state_dir_defaults_to_the_home_directory(monkeypatch, tmp_path: Path) -> None:
    """Canary. Every assertion below is about relocation, and all of them would
    hold for a build that had quietly changed where state lives by default —
    which would strand existing installations' data."""
    monkeypatch.delenv("NLQ_STATE_DIR", raising=False)
    config = _config_with(monkeypatch)

    state_dir = config.STATE_DIR
    assert state_dir == Path.home() / ".nlqueries"


@pytest.mark.parametrize(
    "attribute", ["KB_PATH", "CONNECTORS_FILE", "CAPSULES_DIR", "FEEDBACK_DIR"]
)
def test_the_existing_paths_follow_the_state_directory(
    monkeypatch, tmp_path: Path, attribute: str
) -> None:
    """One setting moves all of it, which is the point: an operator relocating
    state should not have to know how many paths there are."""
    monkeypatch.delenv(attribute, raising=False)
    config = _config_with(monkeypatch, NLQ_STATE_DIR=str(tmp_path / "state"))

    resolved = getattr(config, attribute)
    assert (tmp_path / "state") in resolved.parents


def test_a_per_path_override_still_wins(monkeypatch, tmp_path: Path) -> None:
    """The per-path variables predate this and are documented. An existing
    deployment that sets `KB_PATH` alone must be unaffected."""
    config = _config_with(
        monkeypatch,
        NLQ_STATE_DIR=str(tmp_path / "state"),
        KB_PATH=str(tmp_path / "elsewhere"),
    )

    kb_path, capsules_dir = config.KB_PATH, config.CAPSULES_DIR
    assert kb_path == tmp_path / "elsewhere"
    assert capsules_dir == tmp_path / "state" / "capsules"


def test_the_pid_file_follows_the_state_directory(monkeypatch, tmp_path: Path) -> None:
    """The path whose absence took the whole query surface down."""
    _config_with(monkeypatch, NLQ_STATE_DIR=str(tmp_path / "state"))
    import nlqueries.embeddings.embed_server as embed_server

    reloaded = importlib.reload(embed_server)
    pid_file = reloaded._PID_FILE
    assert pid_file == tmp_path / "state" / "embed-server.pid"


def test_the_signing_key_follows_the_state_directory(monkeypatch, tmp_path: Path) -> None:
    """Its own docstring already said "the state directory"; it just had no name
    in code, so it stayed pinned to a home directory."""
    _config_with(monkeypatch, NLQ_STATE_DIR=str(tmp_path / "state"))
    from nlqueries.cache.envelope import _key_path

    assert _key_path() == tmp_path / "state" / "cache_signing_key"


def test_no_home_directory_is_written_into_core(monkeypatch) -> None:
    """The property, rather than the six instances of it.

    `Path.home()` should appear exactly once in the package — as the default for
    `STATE_DIR`. Any other use is a path that a deployment cannot relocate, which
    is the whole fault being fixed, and it would otherwise be found the next time
    someone mounts a read-only root.
    """
    package = Path(__file__).resolve().parents[1] / "nlqueries"
    offenders = sorted(
        {
            str(path.relative_to(package))
            for path in package.rglob("*.py")
            for line in path.read_text(encoding="utf-8").splitlines()
            if "Path.home()" in line
        }
    )
    # By file, not by line: pinning the line number would make any edit above it
    # fail this for a reason that has nothing to do with the property.
    assert offenders == ["config.py"], offenders
