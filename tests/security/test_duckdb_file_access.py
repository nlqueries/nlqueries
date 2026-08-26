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

import contextlib
from pathlib import Path

import pytest

from tests.conftest import granted

pytest.importorskip("duckdb", reason="the duckdb package is not installed")

from nlqueries.connectors.duckdb import DuckDBConnector  # noqa: E402

pytestmark = pytest.mark.security

_CANARY = "sec16-canary-contents"


@pytest.fixture()
def duckdb_lab(tmp_path: Path) -> tuple[DuckDBConnector, Path]:
    """A DuckDB database, and a file next to it that queries must not reach."""
    canary = tmp_path / "not-the-database.csv"
    canary.write_text(f"secret\n{_CANARY}\n", encoding="utf-8")

    connector = granted(DuckDBConnector())
    connector.connect({"database": str(tmp_path / "lab.duckdb")})
    with contextlib.suppress(Exception):  # setup only; the tests below say why
        connector.execute_query("CREATE TABLE orders AS SELECT 1 AS id, 9.99 AS total")
    return connector, canary


def test_the_canary_is_actually_readable_on_disk(duckdb_lab) -> None:
    """The instrument check: if the file were unreadable for an unrelated
    reason, every assertion below would pass for the wrong reason."""
    _connector, canary = duckdb_lab

    assert _CANARY in canary.read_text(encoding="utf-8")


@pytest.mark.xfail(
    strict=True,
    reason="SEC-16 — needs the SQL policy (W4) and external access disabled (W3-3)",
)
def test_read_csv_auto_cannot_reach_a_file_outside_the_database(duckdb_lab) -> None:
    """A `SELECT` whose table is a file path."""
    connector, canary = duckdb_lab

    result = connector.execute_query(f"SELECT * FROM read_csv_auto('{canary.as_posix()}')")

    leaked = result.error is None and any(
        _CANARY in str(cell) for row in result.rows for cell in row
    )
    assert not leaked, "read_csv_auto() returned the contents of a file outside the database"


@pytest.mark.xfail(
    strict=True,
    reason="SEC-16 — needs the SQL policy (W4) and external access disabled (W3-3)",
)
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
