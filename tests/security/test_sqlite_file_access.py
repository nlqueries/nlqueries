"""
Can SQLite reach past its own database file? (W3-2)

SQLite has no `read_csv_auto`, and `load_extension` is already off in Python's
`sqlite3`, so the surface is narrower than DuckDB's. What it does have is
`ATTACH`, which takes a path and opens it — and measured on SQLite 3.50.4, a
connection opened `mode=ro` will still attach another database and read every
row in it. Read-only describes what may be *written*, not which files may be
*opened*.

The second half of this file is about the path itself. `mode=ro` arrives as a
URI parameter, so anything that builds that URI by concatenation lets the
`database` credential append parameters of its own.

Canaries are written by the test rather than borrowed from the host, so a
failure means "the engine reached something it should not have", not "the engine
found something interesting on this machine".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from nlqueries.connectors.sqlite import SQLiteConnector, _read_only_uri

from tests.conftest import granted

pytestmark = pytest.mark.security

_CANARY = "w32-canary-contents"


def _database_with(path: Path, script: str) -> Path:
    """Build a database with a plain writable connection.

    Setup is kept out of the object under test: the connector opens an existing
    database read-only, and a fixture that seeded through it would quietly come
    to depend on that not being true.
    """
    raw = sqlite3.connect(str(path))
    raw.executescript(script)
    raw.commit()
    raw.close()
    return path


@pytest.fixture()
def sqlite_lab(tmp_path: Path) -> tuple[SQLiteConnector, Path]:
    """A database, and another one next to it that queries must not reach."""
    _database_with(
        tmp_path / "lab.db",
        "CREATE TABLE orders (id INTEGER, total REAL);INSERT INTO orders VALUES (1, 9.99);",
    )
    secrets = _database_with(
        tmp_path / "secrets.db",
        f"CREATE TABLE creds (secret TEXT); INSERT INTO creds VALUES ('{_CANARY}');",
    )

    connector = granted(SQLiteConnector())
    connector.connect({"database": str(tmp_path / "lab.db")})
    return connector, secrets


def test_the_canary_database_is_actually_readable(sqlite_lab) -> None:
    """The instrument check: if the second database were unreadable for an
    unrelated reason, every assertion below would pass for the wrong reason."""
    _connector, secrets = sqlite_lab

    raw = sqlite3.connect(str(secrets))
    assert raw.execute("SELECT secret FROM creds").fetchall() == [(_CANARY,)]
    raw.close()


def test_attach_cannot_open_another_database(sqlite_lab) -> None:
    """`ATTACH` is a file open wearing a SQL statement's clothes.

    A read-only connection permits it: that was measured before this was fixed,
    and the attached database's rows came back.
    """
    connector, secrets = sqlite_lab

    result = connector.execute_query(f"ATTACH DATABASE '{secrets.as_posix()}' AS other")

    assert result.error is not None, "ATTACH opened a database outside this connection"


def test_a_write_is_refused(sqlite_lab) -> None:
    connector, _secrets = sqlite_lab

    result = connector.execute_query("INSERT INTO orders VALUES (2, 1.0)")

    assert result.error is not None, "a read-only database accepted an INSERT"


def test_dangerous_pragmas_are_refused(sqlite_lab) -> None:
    """Pragmas are allow-listed, because the dangerous ones are the ones nobody
    thinks to name: `writable_schema` makes the catalogue an ordinary table,
    `temp_store` decides whether scratch data reaches the disk, and
    `database_list` reports the path of every attached file."""
    connector, _secrets = sqlite_lab

    for pragma in ("database_list", "temp_store=FILE", "writable_schema=ON", "journal_mode"):
        result = connector.execute_query(f"PRAGMA {pragma}")
        assert result.error is not None, f"PRAGMA {pragma} was permitted"


def test_load_extension_is_refused(sqlite_lab) -> None:
    """Already off in Python's `sqlite3`, asserted so it stays off."""
    connector, _secrets = sqlite_lab

    result = connector.execute_query("SELECT load_extension('anything')")

    assert result.error is not None


def test_a_missing_database_is_refused_rather_than_created(tmp_path: Path) -> None:
    """SQLite would create an empty database and answer every question with
    "no tables", which is a misconfiguration reported as a success."""
    connector = granted(SQLiteConnector())
    missing = tmp_path / "nothing-here.db"

    with pytest.raises(FileNotFoundError):
        connector.connect({"database": str(missing)})

    assert not missing.exists(), "connect() created the database it was asked to refuse"


def test_a_hostile_path_cannot_append_uri_parameters(tmp_path: Path) -> None:
    """The injection this connector would have had if the URI were built by
    concatenation.

    Measured on SQLite 3.50.4: `f"file:{path}?mode=ro"` where *path* ends
    `?mode=rwc&` produces a URI with two `mode` parameters, and SQLite honours
    the first — so the credential decides the mode and the database opens
    writable. Percent-encoding keeps the whole value a filename.
    """
    hostile = Path(f"{(tmp_path / 'lab.db').as_posix()}?mode=rwc&")

    uri = _read_only_uri(hostile)

    assert "%3F" in uri, f"the '?' reached the URI unescaped: {uri}"
    assert uri.count("?") == 1, f"more than one URI query separator: {uri}"
    assert uri.endswith("?mode=ro")


def test_the_schema_still_reads(sqlite_lab) -> None:
    """The control for the pragma allow-list.

    `extract_schema` runs `PRAGMA table_info` and `PRAGMA foreign_key_list`
    itself, so a blanket pragma refusal would have left the product unable to
    describe the database it is pointed at — passing every test above by being
    useless.
    """
    connector, _secrets = sqlite_lab

    schema = connector.extract_schema()

    assert [t.name for t in schema.tables] == ["orders"]
    assert {c.name for c in schema.tables[0].columns} == {"id", "total"}


def test_ordinary_queries_still_work(sqlite_lab) -> None:
    """The other control. Whatever stops the payloads must not stop this."""
    connector, _secrets = sqlite_lab

    result = connector.execute_query("SELECT count(*) FROM orders")

    assert result.error is None
    assert result.rows == [[1]]
