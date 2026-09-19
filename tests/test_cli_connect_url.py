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
from nlqueries import config
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
    # Both ends. `_save_connector` is a read-modify-write that WRITES through
    # `cli.main.CONNECTORS_FILE`, bound at import, and READS through
    # `load_connectors_for_update` -> `config.CONNECTORS_FILE`, resolved per
    # call. Patch one and registering a second connector reads a different file
    # from the one it writes, so the first entry vanishes -- which looks exactly
    # like the id collision this module tests for. tests/conftest.py warns about
    # the mirror image of this.
    monkeypatch.setattr(cli_main, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    return path


@pytest.fixture(autouse=True)
def keychain(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Stand in for the OS credential store, for every test in this module.

    `keyring` is a base dependency, not an optional extra, so on a machine with
    a working backend an unpatched test writes its fixture password into the
    real keychain under the service `nlqueries` and leaves it there.
    tests/test_connector_resolver_seam.py patches the same seam for the same
    reason. Autouse rather than per-test: the cost of forgetting it is a secret
    on the developer's machine, not a failing assertion.
    """
    saved: dict[str, str] = {}

    def save_password(connector_id: str, password: str) -> bool:
        saved[connector_id] = password
        return True  # as if the keychain accepted it

    monkeypatch.setattr(cli_main, "_save_password", save_password)
    return saved


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


def test_password_in_the_url_is_kept_out_of_the_written_config(
    connectors_file: Path,
    stub_handshake: list[dict[str, Any]],
    keychain: dict[str, str],
) -> None:
    """A password inside the URL is handled like one passed as --password.

    Masked, not removed. `URL.set(password=None)` is a no-op -- SQLAlchemy
    applies only non-None values -- so the writer's `str(...)` falls through to
    `URL.__repr__`, which is `render_as_string(hide_password=True)`. The secret
    is genuinely off disk; what stands in for it is `***`. That predates this
    change, and the literal is asserted so that a later change making the
    removal real fails here and gets looked at rather than passing silently.
    """
    result = CliRunner().invoke(
        cli,
        ["connect", "sqlalchemy", "--url", f"mysql+pymysql://alice:{SECRET}@db.internal:3306/shop"],
    )

    assert result.exit_code == 0, result.output
    written = connectors_file.read_text(encoding="utf-8")
    assert "url: mysql+pymysql://alice:***@db.internal:3306/shop" in written
    assert SECRET not in written
    assert SECRET not in result.output
    assert list(keychain.values()) == [SECRET]


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


def test_password_env_is_honoured_for_a_url(
    monkeypatch: pytest.MonkeyPatch,
    connectors_file: Path,
    stub_handshake: list[dict[str, Any]],
) -> None:
    """--password-env applies to --url, rather than being accepted and dropped.

    Refusing it instead would have left the URL as the only way to supply a
    secret for this type, and a URL is an argv value -- the shell-history
    exposure --password-env exists to avoid.
    """
    monkeypatch.setenv("DB_PW", SECRET)

    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "sqlalchemy",
            "--url",
            "postgresql://alice@db.internal/shop",
            "--password-env",
            "DB_PW",
        ],
    )

    assert result.exit_code == 0, result.output
    assert stub_handshake[0]["password"] == SECRET
    assert stub_handshake[0]["url"] == f"postgresql://alice:{SECRET}@db.internal/shop"


def test_password_option_is_honoured_for_a_url(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "sqlalchemy",
            "--url",
            "postgresql://alice@db.internal/shop",
            "--password",
            SECRET,
        ],
    )

    assert result.exit_code == 0, result.output
    assert stub_handshake[0]["password"] == SECRET


def test_password_in_both_the_url_and_an_option_is_refused(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    """Same principle as --url on another db-type: two sources, no precedence."""
    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "sqlalchemy",
            "--url",
            f"postgresql://alice:{SECRET}@db.internal/shop",
            "--password",
            "a-different-one",
        ],
    )

    assert result.exit_code != 0
    assert "already carries a password" in result.output
    assert not connectors_file.exists()


def test_unset_password_env_is_refused_for_a_url(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "sqlalchemy",
            "--url",
            "postgresql://alice@db.internal/shop",
            "--password-env",
            "NOT_SET_ANYWHERE",
        ],
    )

    assert result.exit_code != 0
    assert "is not set or is empty" in result.output
    assert not connectors_file.exists()


def test_a_url_with_no_password_does_not_prompt(
    tmp_path: Path, connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    """sqlite URLs have no password; the command must not stop for input.

    Empty stdin, so an interactive prompt would surface rather than hang.
    """
    result = CliRunner().invoke(cli, ["connect", "sqlalchemy", "--url", "sqlite:///x.db"], input="")

    assert result.exit_code == 0, result.output
    assert "Database password" not in result.output
    assert stub_handshake[0]["password"] is None


def test_two_hosts_with_the_same_database_get_distinct_ids(
    connectors_file: Path,
    stub_handshake: list[dict[str, Any]],
    keychain: dict[str, str],
) -> None:
    """Keying the id on the database alone silently destroys the first connector.

    `_save_connector` and `_save_password` both overwrite by id, so a colliding
    id takes the earlier entry's alias and keychain password with it and prints
    nothing to say so. Every other networked type composes
    `{db_type}:{host}:{database}` for this reason.
    """
    for host, secret, alias in (
        ("staging.internal", "staging-pw-a1b2", "stg"),
        ("prod.internal", "prod-pw-c3d4", "prod"),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "connect",
                "sqlalchemy",
                "--url",
                f"postgresql://alice:{secret}@{host}/shop",
                "--alias",
                alias,
            ],
        )
        assert result.exit_code == 0, result.output

    written = connectors_file.read_text(encoding="utf-8")
    assert "sqlalchemy:postgresql:staging.internal:shop" in written
    assert "sqlalchemy:postgresql:prod.internal:shop" in written
    # Both aliases survive, so neither entry replaced the other.
    assert "alias: stg" in written
    assert "alias: prod" in written
    assert sorted(keychain) == [
        "sqlalchemy:postgresql:prod.internal:shop",
        "sqlalchemy:postgresql:staging.internal:shop",
    ]


def test_id_falls_back_when_the_url_has_no_host_or_no_path(
    connectors_file: Path, stub_handshake: list[dict[str, Any]]
) -> None:
    """A SQLite URL has no host; a DSN-style URL carries no path."""
    result = CliRunner().invoke(cli, ["connect", "sqlalchemy", "--url", "sqlite:///data/app.db"])
    assert result.exit_code == 0, result.output
    assert "sqlalchemy:sqlite:data/app.db" in result.output

    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "sqlalchemy",
            "--url",
            "oracle+cx_oracle://alice:s@dbhost:1521/?service_name=ORCL",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "sqlalchemy:oracle+cx_oracle:dbhost" in result.output
