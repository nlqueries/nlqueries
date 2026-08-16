"""Tests for connector reuse (W-7).

Every query used to build a new SQLAlchemy engine — a fresh TCP connection, TLS
handshake and authentication against the customer's database — and then never
dispose it. At any real concurrency that is visible connection churn on the
customer's side, and it defeats SQLAlchemy's pooling entirely: a pool discarded
after one query has pooled nothing.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from nlqueries import config
from nlqueries.connectors import loader


@pytest.fixture(autouse=True)
def _clean_cache():
    loader.invalidate_connector_cache()
    yield
    loader.invalidate_connector_cache()


@pytest.fixture
def connectors_file(tmp_path, monkeypatch):
    path = tmp_path / "connectors.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "postgres:localhost:db": {
                    "db_type": "postgres",
                    "url": "postgresql://user:secret@localhost:5432/db",
                }
            }
        )
    )
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_TTL_SECONDS", 900.0)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_MAX_ENTRIES", 32)
    return path


@pytest.fixture
def built(monkeypatch):
    """Count how many connectors get built, and hand back disposable doubles."""
    made: list[MagicMock] = []

    class _Connector:
        def __init__(self) -> None:
            self.closed = False
            made.append(self)  # type: ignore[arg-type]

        def connect(self, _cfg: dict[str, Any]) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Connector)
    return made


def test_a_second_query_reuses_the_first_connector(connectors_file, built) -> None:
    first = loader.open_connector_for_agent("postgres:localhost:db")
    second = loader.open_connector_for_agent("postgres:localhost:db")

    assert first is second
    assert len(built) == 1, f"built {len(built)} connectors for two queries"


def test_a_changed_credential_rebuilds_without_a_restart(connectors_file, built) -> None:
    """The fingerprint covers the resolved password, so rotation is self-healing
    even before anything calls invalidate explicitly."""
    loader.open_connector_for_agent("postgres:localhost:db")

    connectors_file.write_text(
        yaml.safe_dump(
            {
                "postgres:localhost:db": {
                    "db_type": "postgres",
                    "url": "postgresql://user:rotated@localhost:5432/db",
                }
            }
        )
    )
    loader.open_connector_for_agent("postgres:localhost:db")

    assert len(built) == 2, "kept using a connector built with the old credential"
    assert built[0].closed is True, "the superseded connector was not disposed"


def test_an_expired_entry_is_rebuilt_and_disposed(connectors_file, built, monkeypatch) -> None:
    loader.open_connector_for_agent("postgres:localhost:db")
    monkeypatch.setattr(config, "CONNECTOR_CACHE_TTL_SECONDS", 0.0)

    loader.open_connector_for_agent("postgres:localhost:db")

    assert len(built) == 2
    assert built[0].closed is True


def test_eviction_disposes_rather_than_waiting_for_the_collector(
    connectors_file, built, monkeypatch
) -> None:
    """An engine released only by garbage collection is not on a schedule any
    DBA would recognise as one."""
    monkeypatch.setattr(config, "CONNECTOR_CACHE_MAX_ENTRIES", 1)

    entries = {
        f"postgres:host{i}:db": {
            "db_type": "postgres",
            "url": f"postgresql://user:secret@host{i}:5432/db",
        }
        for i in range(2)
    }
    connectors_file.write_text(yaml.safe_dump(entries))

    loader.open_connector_for_agent("postgres:host0:db")
    loader.open_connector_for_agent("postgres:host1:db")

    assert built[0].closed is True, "the evicted connector's engine was left open"
    assert built[1].closed is False


def test_invalidation_closes_and_forgets(connectors_file, built) -> None:
    loader.open_connector_for_agent("postgres:localhost:db")
    loader.invalidate_connector_cache("postgres:localhost:db")

    assert built[0].closed is True
    loader.open_connector_for_agent("postgres:localhost:db")
    assert len(built) == 2


def test_invalidating_everything_closes_everything(connectors_file, built) -> None:
    entries = {
        f"postgres:host{i}:db": {
            "db_type": "postgres",
            "url": f"postgresql://user:secret@host{i}:5432/db",
        }
        for i in range(3)
    }
    connectors_file.write_text(yaml.safe_dump(entries))
    for i in range(3):
        loader.open_connector_for_agent(f"postgres:host{i}:db")

    loader.invalidate_connector_cache()

    assert all(connector.closed for connector in built)


def test_twenty_threads_build_at_most_a_handful(connectors_file, built) -> None:
    """Callers arrive from asyncio.to_thread, so the race is real. A lock held
    across the whole build would serialise every first connection on a cold
    cache, so a few duplicates under a simultaneous cold start are accepted —
    what must not happen is one per query, forever."""
    results: list[Any] = []

    def _open() -> None:
        results.append(loader.open_connector_for_agent("postgres:localhost:db"))

    threads = [threading.Thread(target=_open) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 20 and all(r is not None for r in results)
    assert len(built) < 20, f"built {len(built)} connectors for 20 concurrent opens"
    # Everything that lost the race must have been disposed, not leaked.
    survivors = [c for c in built if not c.closed]
    assert len(survivors) == 1, f"{len(survivors)} connectors left open"


def test_disabling_the_cache_restores_the_old_behaviour(
    connectors_file, built, monkeypatch
) -> None:
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)

    first = loader.open_connector_for_agent("postgres:localhost:db")
    second = loader.open_connector_for_agent("postgres:localhost:db")

    assert first is not second
    assert len(built) == 2


def test_the_fingerprint_does_not_store_a_readable_password(connectors_file) -> None:
    cfg = yaml.safe_load(connectors_file.read_text())["postgres:localhost:db"]
    fingerprint = loader._fingerprint("postgres:localhost:db", cfg)

    assert "secret" not in fingerprint
    assert len(fingerprint) == 64
