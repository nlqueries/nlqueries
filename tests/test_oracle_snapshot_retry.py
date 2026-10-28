"""An Oracle query refused with ORA-01466 is retried once, after a delay.

Core's read-only transaction on Oracle cannot read a table truncated or created
just before it. A fresh transaction straight away is refused too (measured 0/8),
so the retry waits ``ORACLE_SNAPSHOT_RETRY_SECONDS``, 3 by default (owner,
2026-10-28). These tests drive ``_execute_query`` with a first attempt that fails
the way Oracle does. The real thing is in tests/integration/test_oracle_integration.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nlqueries import config
from nlqueries.connectors import sqlalchemy_connector as module
from nlqueries.connectors.base import QueryResult
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector
from sqlalchemy import create_engine
from sqlalchemy.exc import DatabaseError

from tests.conftest import granted

_REFUSAL = DatabaseError(
    "SELECT COUNT(*) FROM t",
    None,
    Exception("ORA-01466: unable to read data - table definition has changed"),
)


@pytest.fixture
def connector(tmp_path: Path) -> SQLAlchemyConnector:
    # As tests/test_sqlalchemy_connector.py does: connect() refuses sqlite URLs.
    c = granted(SQLAlchemyConnector())
    c._engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")  # noqa: SLF001
    return c


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []
    monkeypatch.setattr(module.time, "sleep", waits.append)
    return waits


def _attempts(
    monkeypatch: pytest.MonkeyPatch, c: SQLAlchemyConnector, dialect: str, *outcomes: Any
) -> list[int]:
    """Make each attempt raise or return the next of *outcomes*; count them."""
    calls: list[int] = []
    remaining = list(outcomes)

    def _run_once(*_: Any) -> QueryResult:
        calls.append(1)
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(c, "_run_once", _run_once)
    monkeypatch.setattr(c, "_dialect_name", lambda: dialect)
    return calls


_OK = QueryResult(columns=["n"], rows=[[0]], row_count=1, execution_time_ms=1.0, error=None)


def test_a_refused_query_is_retried_once_after_the_delay(monkeypatch, connector, slept) -> None:
    calls = _attempts(monkeypatch, connector, "oracle", _REFUSAL, _OK)
    result = connector._execute_query("SELECT COUNT(*) FROM t")
    assert result.error is None and result.rows == [[0]]
    assert (len(calls), slept) == (2, [config.ORACLE_SNAPSHOT_RETRY_SECONDS])


def test_a_second_refusal_is_reported(monkeypatch, connector, slept) -> None:
    calls = _attempts(monkeypatch, connector, "oracle", _REFUSAL, _REFUSAL)
    result = connector._execute_query("SELECT COUNT(*) FROM t")
    assert result.error is not None and "ORA-01466" in result.error
    assert len(calls) == 2 and len(slept) == 1


def test_another_oracle_error_is_not_retried(monkeypatch, connector, slept) -> None:
    other = DatabaseError("SELECT 1", None, Exception("ORA-00942: table or view does not exist"))
    calls = _attempts(monkeypatch, connector, "oracle", other)
    assert "ORA-00942" in (connector._execute_query("SELECT 1").error or "")
    assert (len(calls), slept) == (1, [])


def test_only_oracle_is_retried(monkeypatch, connector, slept) -> None:
    calls = _attempts(monkeypatch, connector, "postgresql", _REFUSAL)
    assert connector._execute_query("SELECT 1").error is not None
    assert (len(calls), slept) == (1, [])


def test_a_delay_of_zero_turns_the_retry_off(monkeypatch, connector, slept) -> None:
    monkeypatch.setattr(config, "ORACLE_SNAPSHOT_RETRY_SECONDS", 0.0)
    calls = _attempts(monkeypatch, connector, "oracle", _REFUSAL)
    assert connector._execute_query("SELECT 1").error is not None
    assert (len(calls), slept) == (1, [])


def test_the_configured_delay_is_the_one_waited(monkeypatch, connector, slept) -> None:
    monkeypatch.setattr(config, "ORACLE_SNAPSHOT_RETRY_SECONDS", 7.5)
    _attempts(monkeypatch, connector, "oracle", _REFUSAL, _OK)
    connector._execute_query("SELECT 1")
    assert slept == [7.5]


def test_an_ordinary_query_still_runs(connector) -> None:
    """The split into _run_once leaves the plain path as it was."""
    result = connector._execute_query("SELECT 1 AS n")
    assert result.error is None and result.rows == [[1]]


# --- The setting -------------------------------------------------------------------


def _setting(monkeypatch: pytest.MonkeyPatch, written: str | None) -> float:
    if written is None:
        monkeypatch.delenv("ORACLE_SNAPSHOT_RETRY_SECONDS", raising=False)
    else:
        monkeypatch.setenv("ORACLE_SNAPSHOT_RETRY_SECONDS", written)
    return config._oracle_snapshot_retry_seconds()


def test_the_default_is_three_seconds(monkeypatch) -> None:
    assert _setting(monkeypatch, None) == 3.0


@pytest.mark.parametrize(("written", "seconds"), [("0", 0.0), ("2.5", 2.5), ("10", 10.0)])
def test_a_deployment_can_set_its_own(monkeypatch, written, seconds) -> None:
    assert _setting(monkeypatch, written) == seconds


@pytest.mark.parametrize("written", ["-1", "inf", "nan", "1e400", "three", ""])
def test_a_value_that_is_not_a_delay_is_ignored_with_a_warning(
    monkeypatch, caplog, written
) -> None:
    assert _setting(monkeypatch, written) == 3.0
    assert "ORACLE_SNAPSHOT_RETRY_SECONDS" in caplog.text
