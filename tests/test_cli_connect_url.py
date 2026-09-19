"""CLI tests — `nlqueries connect sqlalchemy --url`.

The generic SQLAlchemy connector was registered and reachable in code, and both
`docs/connectors.md` and the website told people to reach it with
`connect sqlalchemy --url "..."`, but `connect` had no `--url` option and no
`sqlalchemy` scheme, so the documented command failed two different ways. These
cover the option itself and, more importantly, what happens to a password that
arrives inside the URL rather than in `--password`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import nlqueries.cli.main as cli_main
import pytest
from click.testing import CliRunner
from nlqueries.cli.main import cli

# Distinctive on purpose: a password that also appeared in the surrounding
# static text would let "the secret is absent" pass without the stripping that
# is the point of the test.
SECRET = "pw-9d41f7c2-not-in-any-fixture"


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the CLI's writer.

    `cli.main` binds `CONNECTORS_FILE` at import, so patching `config` alone
    would leave the real file as the write target -- the failure tests/conftest
    installs a session guard against.
    """
    path = tmp_path / "connectors.yaml"
    monkeypatch.setattr(cli_main, "CONNECTORS_FILE", path)
    return path


@pytest.fixture
def stub_handshake(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Let `connect` reach persistence without a database, recording credentials."""
    from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector

    seen: list[dict[str, Any]] = []

    def connect(self: Any, credentials: dict[str, Any]) -> None:
        seen.append(credentials)

    monkeypatch.setattr(SQLAlchemyConnector, "connect", connect)
    monkeypatch.setattr(SQLAlchemyConnector, "test_connection", lambda self: True)
    return seen


def test_connects_to_a_real_sqlite_url(tmp_path: Path, connectors_file: Path) -> None:
    """No stub: the whole path, against a database that actually exists."""
    db = tmp_path / "shop.db"
    sqlite3.connect(db).close()

    result = CliRunner().invoke(
        cli, ["connect", "sqlalchemy", "--url", f"sqlite:///{db.as_posix()}", "--alias", "gen"]
    )

    assert result.exit_code == 0, result.output
    written = connectors_file.read_text(encoding="utf-8")
    assert "db_type: sqlalchemy" in written
    assert "alias: gen" in written


def test_url_credential_reaches_the_connector(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    """SQLAlchemyConnector's only credential is `url`, and `connect` must pass it.

    Without this the command fails with "requires a 'url' credential" -- the
    credentials dict is assembled from the discrete options and carried no URL.
    """
    url = "mysql+pymysql://alice:secret@db.internal:3306/shop"
    result = CliRunner().invoke(cli, ["connect", "sqlalchemy", "--url", url])

    assert result.exit_code == 0, result.output
    assert stub_handshake[0]["url"] == url


def test_url_fields_are_read_back_out_of_the_url(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    """host/port/database/user come from the URL, not the option defaults.

    `--host` defaults to "localhost"; recording that for a connector pointing at
    db.internal would make the config file describe the wrong server.
    """
    result = CliRunner().invoke(
        cli,
        ["connect", "sqlalchemy", "--url", "mysql+pymysql://alice:secret@db.internal:3306/shop"],
    )

    assert result.exit_code == 0, result.output
    written = connectors_file.read_text(encoding="utf-8")
    assert "host: db.internal" in written
    assert "port: 3306" in written
    assert "database: shop" in written
    assert "user: alice" in written
    assert "localhost" not in written


def test_password_in_the_url_is_stripped_when_the_keychain_takes_it(
    monkeypatch: pytest.MonkeyPatch,
    connectors_file: Path,
    stub_handshake: list[dict[str, Any]],
) -> None:
    """A password inside the URL must be handled like one passed as --password."""
    captured: dict[str, str] = {}

    def save_password(connector_id: str, password: str) -> bool:
        captured[connector_id] = password
        return True  # keychain available

    monkeypatch.setattr(cli_main, "_save_password", save_password)

    result = CliRunner().invoke(
        cli,
        ["connect", "sqlalchemy", "--url", f"mysql+pymysql://alice:{SECRET}@db.internal:3306/shop"],
    )

    assert result.exit_code == 0, result.output
    written = connectors_file.read_text(encoding="utf-8")
    # Negative control: the URL really was written, so the assertion below is
    # about stripping rather than about an absent line.
    assert "url: mysql+pymysql://alice:" in written
    assert "db.internal:3306/shop" in written
    assert SECRET not in written
    assert SECRET not in result.output
    assert list(captured.values()) == [SECRET]


def test_url_password_never_printed(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    """The connect banner prints where it is going, never the URL itself."""
    result = CliRunner().invoke(
        cli,
        ["connect", "sqlalchemy", "--url", f"mysql+pymysql://alice:{SECRET}@db.internal:3306/shop"],
    )

    assert result.exit_code == 0, result.output
    assert SECRET not in result.output
    assert "db.internal" in result.output


def test_sqlalchemy_without_url_is_refused(connectors_file: Path) -> None:
    result = CliRunner().invoke(cli, ["connect", "sqlalchemy", "--database", "x"])

    assert result.exit_code != 0
    assert "sqlalchemy requires --url" in result.output
    assert not connectors_file.exists()


def test_url_on_another_db_type_is_refused(connectors_file: Path) -> None:
    """Two sources for one value, with nothing to say which wins."""
    result = CliRunner().invoke(cli, ["connect", "postgres", "--url", "postgresql://a:b@h/d"])

    assert result.exit_code != 0
    assert "only valid with the 'sqlalchemy' db-type" in result.output
    assert not connectors_file.exists()


def test_malformed_url_is_refused(connectors_file: Path) -> None:
    result = CliRunner().invoke(cli, ["connect", "sqlalchemy", "--url", "not a url"])

    assert result.exit_code != 0
    assert "not a valid SQLAlchemy URL" in result.output
    assert not connectors_file.exists()
