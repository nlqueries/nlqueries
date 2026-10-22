"""`nlqueries health` reaches each registered database, through the real
connector and its permission check rather than a stand-in."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from nlqueries.cli import main as cli_main


@pytest.fixture
def shop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, total REAL)")
    monkeypatch.setattr(
        cli_main,
        "_load_connectors",
        lambda: {
            f"sqlite:{db}": {
                "db_type": "sqlite",
                "database": str(db),
                "url": f"sqlite:///{db}",
                "alias": "shop",
            }
        },
    )
    return db


def test_a_reachable_database_is_reported_healthy(shop: Path) -> None:
    """Every database was reported as "connection failed": the probe ran with no
    permission to execute, and the connector refused it."""
    results = {r.service: r for r in cli_main._check_connectors(None)}

    probe = results["Database (shop)"]
    assert probe.status == "ok", (probe.detail, probe.error)
    assert "connected" in probe.detail


def test_a_missing_database_is_still_reported_as_failing(
    shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: the probe still fails when the database does."""
    from nlqueries.connectors.sqlite import SQLiteConnector

    def refuse(self, credentials):
        raise OSError("unable to open database file")

    monkeypatch.setattr(SQLiteConnector, "connect", refuse)
    results = {r.service: r for r in cli_main._check_connectors(None)}

    assert results["Database (shop)"].status == "fail"
