"""
Can SQLite reach past its own database file? (W3-2)

SQLite provides no equivalent of `read_csv_auto`, and `load_extension` is
already disabled in Python's `sqlite3`, so the surface is narrower than
DuckDB's. It does provide `ATTACH`, which accepts a path and opens it. Measured
on SQLite 3.50.4: a connection opened `mode=ro` will still attach another
database and read its contents. Read-only restricts what may be written, not
which files may be opened.

The remaining tests cover the database path itself. `mode=ro` is supplied as a
URI parameter, so a URI built by string concatenation permits the `database`
credential to append parameters of its own.

Canary data is written by the test rather than taken from the host, so a
failure indicates that the engine reached data it should not have, rather than
that it found existing data on the machine.
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
    """Build a database using a plain writable connection.

    Setup is kept out of the object under test: the connector opens an existing
    database read-only, and a fixture that seeded through it would depend on
    that not being the case.
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
    """Verifies the test's own instrument: if the second database were
    unreadable for an unrelated reason, the assertions below would pass
    regardless of the connector's behaviour."""
    _connector, secrets = sqlite_lab

    raw = sqlite3.connect(str(secrets))
    assert raw.execute("SELECT secret FROM creds").fetchall() == [(_CANARY,)]
    raw.close()


def test_attach_cannot_open_another_database(sqlite_lab) -> None:
    """`ATTACH` opens a file specified by the statement.

    A read-only connection permits it. Measured before this was addressed: the
    attached database's rows were returned.
    """
    connector, secrets = sqlite_lab

    result = connector.execute_query(f"ATTACH DATABASE '{secrets.as_posix()}' AS other")

    assert result.error is not None, "ATTACH opened a database outside this connection"


def test_a_write_is_refused(sqlite_lab) -> None:
    connector, _secrets = sqlite_lab

    result = connector.execute_query("INSERT INTO orders VALUES (2, 1.0)")

    assert result.error is not None, "a read-only database accepted an INSERT"


def test_dangerous_pragmas_are_refused(sqlite_lab) -> None:
    """Pragmas are allow-listed. Those requiring restriction are not confined
    to an obvious set: `writable_schema` exposes the catalogue as a writable
    table, `temp_store` controls whether scratch data reaches disk, and
    `database_list` discloses the path of every attached file."""
    connector, _secrets = sqlite_lab

    for pragma in ("database_list", "temp_store=FILE", "writable_schema=ON", "journal_mode"):
        result = connector.execute_query(f"PRAGMA {pragma}")
        assert result.error is not None, f"PRAGMA {pragma} was permitted"


def test_load_extension_is_refused(sqlite_lab) -> None:
    """Already disabled in Python's `sqlite3`; asserted so it remains so."""
    connector, _secrets = sqlite_lab

    result = connector.execute_query("SELECT load_extension('anything')")

    assert result.error is not None


def test_a_missing_database_is_refused_rather_than_created(tmp_path: Path) -> None:
    """SQLite would otherwise create an empty database at this path, so a
    misconfiguration would present as success."""
    connector = granted(SQLiteConnector())
    missing = tmp_path / "nothing-here.db"

    with pytest.raises(FileNotFoundError):
        connector.connect({"database": str(missing)})

    assert not missing.exists(), "connect() created the database it was asked to refuse"


def test_a_hostile_path_cannot_append_uri_parameters(tmp_path: Path) -> None:
    """The injection that would arise if the URI were built by concatenation.

    Measured on SQLite 3.50.4: `f"file:{path}?mode=ro"`, where *path* ends
    `?mode=rwc&`, produces a URI with two `mode` parameters, and SQLite applies
    the first. The credential would therefore determine the access mode and the
    database would open writable. Percent-encoding keeps the value a filename.
    """
    hostile = Path(f"{(tmp_path / 'lab.db').as_posix()}?mode=rwc&")

    uri = _read_only_uri(hostile)

    assert "%3F" in uri, f"the '?' reached the URI unescaped: {uri}"
    assert uri.count("?") == 1, f"more than one URI query separator: {uri}"
    assert uri.endswith("?mode=ro")


def test_the_schema_still_reads(sqlite_lab) -> None:
    """The control for the pragma allow-list.

    `extract_schema` issues `PRAGMA table_info` and `PRAGMA foreign_key_list`
    itself, so a blanket refusal of pragmas would leave the connector unable to
    describe the database, while still satisfying every assertion above.
    """
    connector, _secrets = sqlite_lab

    schema = connector.extract_schema()

    assert [t.name for t in schema.tables] == ["orders"]
    assert {c.name for c in schema.tables[0].columns} == {"id", "total"}


def test_ordinary_queries_still_work(sqlite_lab) -> None:
    """The second control: the restrictions above must not block normal use."""
    connector, _secrets = sqlite_lab

    result = connector.execute_query("SELECT count(*) FROM orders")

    assert result.error is None
    assert result.rows == [[1]]
