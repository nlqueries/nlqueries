"""``open_connector_for_agent`` must say why it returned ``None``.

It has five ways of failing — no connector id for the agent, no connectors file,
the connector absent from that file, an unregistered ``db_type``, and a refused
connection — and every one of them returned a bare ``None``. The caller cannot
tell them apart, and each has a different fix.

Downstream that silence becomes an answer with no rows and no error: the SQL
path reads a ``None`` connector as "nothing to execute", so a misconfigured
agent is indistinguishable from a query that legitimately matched nothing. That
is the shape of the defect this whole area has been fixing, with the reason not
merely dropped at a boundary but never written down at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from nlqueries import config
from nlqueries.connectors import loader
from nlqueries.execution import ExecutionPolicy

_CONNECTOR_ID = "postgres:localhost:db"


@pytest.fixture(autouse=True)
def _clean_cache():
    loader.invalidate_connector_cache()
    yield
    loader.invalidate_connector_cache()


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "connectors.yaml"
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)
    return path


def _write(path: Path, **entry: Any) -> None:
    base = {"db_type": "postgres", "url": "postgresql://user:hunter2@localhost:5432/db"}
    base.update(entry)
    path.write_text(yaml.safe_dump({_CONNECTOR_ID: base}))


def _open(agent: str = _CONNECTOR_ID):
    return loader.open_connector_for_agent(agent, ExecutionPolicy.execute_read_only())


def test_a_successful_open_says_nothing(connectors_file, caplog, monkeypatch) -> None:
    """Canary. Every assertion below looks for a warning, and all of them would
    be satisfied by a function that warns on every call — which would bury the
    real ones and train people to filter the lot out."""

    class _Connector:
        def connect(self, credentials: dict[str, Any]) -> None:
            return None

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Connector)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is not None
    assert caplog.records == []


def test_no_connectors_file_says_so(connectors_file, caplog) -> None:
    """The file is absent entirely — nothing has ever been projected."""
    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "is missing or empty" in caplog.text


def test_an_agent_with_no_entry_is_told_what_is_there(connectors_file, caplog) -> None:
    """The file exists and nothing in it matches — a different fault from the
    file being missing, with a different fix. Listing what *is* there is what
    makes it actionable: the usual cause is an id that does not match, and the
    right one is in that list."""
    connectors_file.write_text(yaml.safe_dump({"some-other-agent": {"db_type": "postgres"}}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "no entry" in caplog.text
    assert "some-other-agent" in caplog.text, "the message must name what is available"


def test_an_empty_entry_says_so(connectors_file, caplog) -> None:
    """`agent-a:` with no value under it parses to None. Distinct from a missing
    entry, and it was previously indistinguishable from one."""
    connectors_file.write_text(yaml.safe_dump({_CONNECTOR_ID: None}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "is empty" in caplog.text


def test_an_unregistered_db_type_names_what_is_registered(connectors_file, caplog) -> None:
    """Almost always a missing driver extra, and the fix is unguessable without
    knowing what *is* available."""
    _write(connectors_file, db_type="teradata")

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "not registered" in caplog.text
    assert "teradata" in caplog.text
    assert "postgres" in caplog.text, "the message must list what is registered"


def test_a_refused_connection_reports_the_drivers_own_words(
    connectors_file, caplog, monkeypatch
) -> None:
    """When a connector does refuse at `connect` time, its own words survive.

    The exception's message is the diagnosis and any summary written here would
    be a worse one — "server does not support SSL, but SSL was required" is the
    whole answer to the defect this area has been fixing. Note that the real
    instance of it was raised on the execute path rather than this one, since
    `create_engine` opens no socket; this holds the reporting, not the claim
    that refusals usually land here.
    """

    class _Refusing:
        def connect(self, credentials: dict[str, Any]) -> None:
            raise OSError("server does not support SSL, but SSL was required")

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Refusing)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "No connector could be opened" in caplog.text
    assert "server does not support SSL" in caplog.text, "the driver's message must survive"


def test_a_configuration_fault_is_reported_here_too(connectors_file, caplog) -> None:
    """What actually reaches this branch, most of the time.

    `connect` on the SQLAlchemy-backed connectors only builds an engine —
    `create_engine` opens no socket — so a server refusing us surfaces later in
    `execute_query`, not here. This block covers `credentials_for` as well, and
    an entry with no URL raises `KeyError: 'url'` inside it. The message must not
    send the reader looking at the network for a fault in the file.
    """
    connectors_file.write_text(yaml.safe_dump({_CONNECTOR_ID: {"db_type": "postgres"}}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "No connector could be opened" in caplog.text
    assert "connection attempt failed" not in caplog.text, (
        "this branch is mostly configuration; naming the network misdirects"
    )
    assert "KeyError" in caplog.text, "the traceback carries the actual cause"


def test_the_password_is_not_logged(connectors_file, caplog, monkeypatch) -> None:
    """These lines carry a connector id, a db_type and a driver message. The
    entry they are derived from holds a password in its URL, so this holds the
    line that none of that reaches the log."""

    class _Refusing:
        def connect(self, credentials: dict[str, Any]) -> None:
            raise OSError('connection to server at "localhost", port 5432 failed')

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Refusing)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "hunter2" not in caplog.text
