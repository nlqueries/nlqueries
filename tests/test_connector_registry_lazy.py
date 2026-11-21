"""Optional connectors are loaded on first use, not when the registry is imported.

A driver's native library loaded before a process forks is shared with the
child, and the Snowflake driver's cannot survive that: in a Celery prefork
worker, whose parent imports this package before forking, a child's Snowflake
login brought the parent down with no error (2026-11-20). These keep the
drivers out of any process that only imports the registry.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

import nlqueries.connectors as connectors
import pytest
from nlqueries.connectors import _ConnectorRegistry
from nlqueries.connectors.base import DatabaseConnector


def _in_a_fresh_interpreter(code: str) -> str:
    """Run *code* where nothing has been imported yet, and return its output."""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    return done.stdout.strip()


def test_importing_the_registry_loads_no_optional_driver() -> None:
    out = _in_a_fresh_interpreter(
        "import sys, nlqueries.connectors\n"
        "print(sorted(m for m in ('snowflake.connector', 'google.cloud.bigquery') "
        "if m in sys.modules))"
    )
    assert out == "[]"


def test_importing_the_registry_loads_no_optional_connector_module() -> None:
    out = _in_a_fresh_interpreter(
        "import sys, nlqueries.connectors\n"
        "print(sorted(m for m in sys.modules if m in ("
        "'nlqueries.connectors.snowflake', 'nlqueries.connectors.bigquery', "
        "'nlqueries.connectors.redshift', 'nlqueries.connectors.mssql', "
        "'nlqueries.connectors.duckdb')))"
    )
    assert out == "[]"


def test_a_lookup_loads_the_connector_it_asks_for_and_no_other() -> None:
    pytest.importorskip("snowflake.connector")
    out = _in_a_fresh_interpreter(
        "import sys\n"
        "from nlqueries.connectors import CONNECTOR_REGISTRY as r\n"
        "cls = r.get('snowflake')\n"
        "print(cls.__name__, 'snowflake.connector' in sys.modules, "
        "'nlqueries.connectors.duckdb' in sys.modules)"
    )
    assert out == "SnowflakeConnector True False"


class _Fake(DatabaseConnector):  # type: ignore[misc]
    pass


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> _ConnectorRegistry:
    reg = _ConnectorRegistry({"postgres": _Fake})
    reg.defer("fake", __name__, "_Fake")
    loads: list[str] = []
    real = connectors.importlib.import_module

    def counting(name: str, *args: Any) -> Any:
        loads.append(name)
        return real(name, *args)

    monkeypatch.setattr(connectors.importlib, "import_module", counting)
    reg.loads = loads  # type: ignore[attr-defined]
    return reg


def test_a_pending_connector_is_listed_without_being_loaded(registry: _ConnectorRegistry) -> None:
    assert "fake" in registry
    assert sorted(registry) == ["fake", "postgres"]
    assert len(registry) == 2
    assert registry.loads == []  # type: ignore[attr-defined]


def test_it_is_loaded_once_on_first_use(registry: _ConnectorRegistry) -> None:
    assert registry.get("fake") is _Fake
    assert registry["fake"] is _Fake
    assert registry.loads == [__name__]  # type: ignore[attr-defined]


def test_items_and_values_load_every_pending_connector(registry: _ConnectorRegistry) -> None:
    assert dict(registry.items()) == {"postgres": _Fake, "fake": _Fake}
    assert registry.loads == [__name__]  # type: ignore[attr-defined]


def test_an_unknown_name_is_none_by_get_and_a_key_error_by_index(
    registry: _ConnectorRegistry,
) -> None:
    assert registry.get("nope") is None
    assert "nope" not in registry
    with pytest.raises(KeyError):
        registry["nope"]


def test_assigning_and_deleting_behave_as_a_dict(registry: _ConnectorRegistry) -> None:
    """As tests register fakes: an assignment replaces a pending connector."""
    registry["fake"] = DatabaseConnector  # type: ignore[assignment]
    assert registry.get("fake") is DatabaseConnector
    assert registry.loads == []  # type: ignore[attr-defined]
    del registry["fake"]
    assert "fake" not in registry
    with pytest.raises(KeyError):
        del registry["fake"]


def test_a_pending_connector_can_be_deleted_unloaded(registry: _ConnectorRegistry) -> None:
    del registry["fake"]
    assert "fake" not in registry
    assert registry.loads == []  # type: ignore[attr-defined]


def test_a_connector_whose_driver_is_missing_is_not_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = _ConnectorRegistry({})
    monkeypatch.setattr(connectors, "CONNECTOR_REGISTRY", reg)
    monkeypatch.setattr(
        connectors, "_driver_installed", lambda package: package != "snowflake.connector"
    )

    connectors._register_optional_connectors()

    assert "snowflake" not in reg
    assert "bigquery" in reg  # its driver "is" installed here
    assert "duckdb" in reg  # imports its driver only when it connects: always offered


def test_the_driver_check_imports_nothing() -> None:
    out = _in_a_fresh_interpreter(
        "import sys\n"
        "from nlqueries.connectors import _driver_installed\n"
        "_driver_installed('snowflake.connector')\n"
        "print('snowflake.connector' in sys.modules)"
    )
    assert out == "False"


def test_a_missing_driver_found_at_load_is_quiet_and_no_longer_offered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    reg = _ConnectorRegistry({})
    reg.defer("gone", "no_such_driver_package_anywhere", "X")
    with caplog.at_level(logging.WARNING, logger="nlqueries.connectors"):
        assert reg.get("gone") is None
    assert "gone" not in reg
    assert not [r for r in caplog.records if "could not be loaded" in r.getMessage()]
