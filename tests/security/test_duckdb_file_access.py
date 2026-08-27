"""
Can DuckDB read files that are not its database? (SEC-16)

The register carries this as *Unverified*: the audit reported reproducing it and
then destroyed the lab, so what survived was a claim. This settles it.

DuckDB is different from the other engines in a way that matters here. Its
file-reading facilities are ordinary-looking *table functions* — `read_csv_auto`,
`glob`, `read_parquet` — so they arrive inside a perfectly well-formed `SELECT`,
and the database file is not the boundary. Opening the database read-only does
nothing about them.

The canary is written by the test rather than borrowed from the host, so a pass
means "the engine read a file it should not have", not "the engine found
something interesting on this machine".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import granted

duckdb = pytest.importorskip("duckdb", reason="the duckdb package is not installed")

from nlqueries.connectors.duckdb import DuckDBConnector  # noqa: E402

pytestmark = pytest.mark.security

_CANARY = "sec16-canary-contents"


@pytest.fixture()
def duckdb_lab(tmp_path: Path) -> tuple[DuckDBConnector, Path]:
    """A DuckDB database, and a file next to it that queries must not reach."""
    canary = tmp_path / "not-the-database.csv"
    canary.write_text(f"secret\n{_CANARY}\n", encoding="utf-8")

    # Seeded with a plain writable connection: the connector opens an existing
    # database read-only and will not create one. Keeping setup out of the
    # object under test also means this fixture cannot quietly come to depend
    # on that object still being able to write.
    database = tmp_path / "lab.duckdb"
    seed = duckdb.connect(str(database))
    seed.execute("CREATE TABLE orders AS SELECT 1 AS id, 9.99 AS total")
    seed.close()

    connector = granted(DuckDBConnector())
    connector.connect({"database": str(database)})
    return connector, canary


def test_the_canary_is_actually_readable_on_disk(duckdb_lab) -> None:
    """The instrument check: if the file were unreadable for an unrelated
    reason, every assertion below would pass for the wrong reason."""
    _connector, canary = duckdb_lab

    assert _CANARY in canary.read_text(encoding="utf-8")


def test_read_csv_auto_cannot_reach_a_file_outside_the_database(duckdb_lab) -> None:
    """A `SELECT` whose table is a file path."""
    connector, canary = duckdb_lab

    result = connector.execute_query(f"SELECT * FROM read_csv_auto('{canary.as_posix()}')")

    leaked = result.error is None and any(
        _CANARY in str(cell) for row in result.rows for cell in row
    )
    assert not leaked, "read_csv_auto() returned the contents of a file outside the database"


def test_glob_cannot_enumerate_the_filesystem(duckdb_lab) -> None:
    """Enumeration is the cheaper half: it needs no read permission on contents,
    and it is how an attacker finds what is worth reading."""
    connector, canary = duckdb_lab

    result = connector.execute_query(f"SELECT * FROM glob('{canary.parent.as_posix()}/*')")

    enumerated = result.error is None and result.row_count > 0
    assert not enumerated, "glob() listed the directory the database happens to live in"


def test_ordinary_queries_against_the_database_still_work(duckdb_lab) -> None:
    """The control. Whatever stops the two above must not stop this."""
    connector, _canary = duckdb_lab

    result = connector.execute_query("SELECT count(*) FROM orders")

    assert result.error is None


# ---------------------------------------------------------------------------
# The rest of the ways out, measured on 1.5.5 before anything was changed:
# every one of these succeeded.
# ---------------------------------------------------------------------------


def test_attach_cannot_open_another_database(duckdb_lab, tmp_path: Path) -> None:
    """`ATTACH` takes a path, so it is a file read wearing different clothes."""
    connector, _canary = duckdb_lab
    other = tmp_path / "somebody-elses.duckdb"
    duckdb.connect(str(other)).close()

    result = connector.execute_query(f"ATTACH '{other.as_posix()}' AS other")

    assert result.error is not None, "ATTACH opened a database outside this connection"


def test_copy_to_cannot_write_a_file(duckdb_lab, tmp_path: Path) -> None:
    """The one that is not exfiltration but deposit.

    Every other payload here reads. `COPY TO` writes, which makes the database
    a way to put a file of the caller's choosing at a path of their choosing.
    """
    connector, _canary = duckdb_lab
    target = tmp_path / "written-by-a-query.csv"

    result = connector.execute_query(f"COPY (SELECT 1 AS x) TO '{target.as_posix()}'")

    assert result.error is not None, "COPY TO wrote a file"
    assert not target.exists(), "COPY TO left a file behind"


def test_extensions_cannot_be_installed(duckdb_lab) -> None:
    """`INSTALL httpfs` is how a filesystem sandbox becomes a network one."""
    connector, _canary = duckdb_lab

    result = connector.execute_query("INSTALL httpfs")

    assert result.error is not None, "an extension was installed at query time"


def test_the_database_is_opened_read_only(duckdb_lab) -> None:
    """Read-only does not stop the file functions -- that is why the sandbox
    config exists -- but it is still the thing that stops a write."""
    connector, _canary = duckdb_lab

    result = connector.execute_query("CREATE TABLE written AS SELECT 1 AS x")

    assert result.error is not None, "a read-only database accepted a CREATE TABLE"


def test_a_missing_database_is_refused_rather_than_created(tmp_path: Path) -> None:
    """DuckDB would create an empty database and answer every question with
    "no tables", which is a misconfiguration reported as a success."""
    connector = granted(DuckDBConnector())
    missing = tmp_path / "nothing-here.duckdb"

    with pytest.raises(FileNotFoundError):
        connector.connect({"database": str(missing)})

    assert not missing.exists(), "connect() created the database it was asked to refuse"


def test_no_credential_reopens_external_access(tmp_path: Path) -> None:
    """There is deliberately no way to switch this off.

    A setting that can be disabled by configuration is one that will be found
    disabled in the deployment that most needed it, and the credential blob is
    the least trustworthy place to take that instruction from.
    """
    database = tmp_path / "lab.duckdb"
    seed = duckdb.connect(str(database))
    seed.execute("CREATE TABLE orders AS SELECT 1 AS id")
    seed.close()

    canary = tmp_path / "canary.csv"
    canary.write_text(f"secret\n{_CANARY}\n", encoding="utf-8")

    connector = granted(DuckDBConnector())
    connector.connect(
        {
            "database": str(database),
            "enable_external_access": "true",
            "lock_configuration": "false",
            "config": {"enable_external_access": "true"},
            "read_only": False,
        }
    )

    result = connector.execute_query(f"SELECT * FROM read_csv_auto('{canary.as_posix()}')")

    assert result.error is not None, "a credential key reopened external access"
