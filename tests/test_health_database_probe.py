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
    # Closed, not just committed (which is all `with` does), so the file can go.
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, total REAL)")
    conn.commit()
    conn.close()
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


def test_a_missing_database_is_still_reported_as_failing(shop: Path) -> None:
    """The control, through the real connector: the probe still fails when the
    database is not there."""
    shop.unlink()

    results = {r.service: r for r in cli_main._check_connectors(None)}

    assert results["Database (shop)"].status == "fail"
