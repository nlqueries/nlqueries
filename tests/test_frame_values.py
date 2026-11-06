"""Every value a database returns reaches the caller in the answer's frame.

The frame is JSON, and the encoder knew two driver types, `Decimal` and dates.
Any other value raised TypeError on the way out, which failed the whole answer
over one column: a PostgreSQL table with a `uuid` key could not be asked about
at all. These pin the encoding of each type, and that an answer whose rows hold
one comes back with its rows.
"""

from __future__ import annotations

import datetime
import decimal
import ipaddress
import json
import uuid
from typing import Any

import pytest
from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.orchestrator import _json_default

from tests.test_result_second_look import _LLM, _run

_ID = uuid.UUID("6f1c2b9e-3d4a-4c5b-9e8f-0a1b2c3d4e5f")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (decimal.Decimal("12.50"), 12.5),
        (datetime.date(2026, 11, 6), "2026-11-06"),
        (datetime.datetime(2026, 11, 6, 9, 30), "2026-11-06T09:30:00"),
        (datetime.time(9, 30, 15), "09:30:15"),
        (_ID, "6f1c2b9e-3d4a-4c5b-9e8f-0a1b2c3d4e5f"),
        (datetime.timedelta(days=1, hours=2), "1 day, 2:00:00"),
        (ipaddress.ip_address("10.0.0.1"), "10.0.0.1"),
        (memoryview(b"\x01\xab"), "\\x01ab"),
        (b"\x01\xab", "\\x01ab"),
        (bytearray(b"\x01\xab"), "\\x01ab"),
    ],
    ids=[
        "numeric",
        "date",
        "timestamp",
        "time",
        "uuid",
        "interval",
        "inet",
        "bytea",
        "bytes",
        "bytearray",
    ],
)
def test_each_driver_type_is_encoded(value: Any, expected: Any) -> None:
    assert json.loads(json.dumps({"v": value}, default=_json_default)) == {"v": expected}


def test_an_answer_whose_rows_hold_a_uuid_comes_back_with_them() -> None:
    rows = QueryResult(
        columns=["id", "name"],
        rows=[[_ID, "Ann"]],
        row_count=1,
        execution_time_ms=1.0,
        error=None,
    )
    frame, connector = _run(_LLM(), [rows], first_sql="SELECT id, name FROM customers")

    assert frame["sql"] == "SELECT id, name FROM customers"
    assert frame["sql_table"]["rows"] == [[str(_ID), "Ann"]]
    assert frame["sql_table"]["error"] is None
    assert connector.execute_query.call_count == 1
