"""
Every payload, through the real connector, against a real database.

These assert the **absence of an effect**: the marker table is read back, the
sequence is compared before and after. A denial that returns a tidy error while
the row still lands is not a fix, and only the second half of that sentence is
visible from an error string.

Where a payload still succeeds, the test is `xfail(strict=True)` naming the
register row. That does two things a skip would not: it keeps the failure
recorded rather than hidden, and it turns green the moment the finding is fixed
— at which point strict xfail fails the build and makes somebody delete the
marker and update the register. An open finding cannot quietly become a closed
one, and a closed one cannot quietly reopen.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from nlqueries.connectors.postgres import PostgresConnector

from tests.conftest import granted
from tests.security.payloads import POSTGRES, SAFE_POSTGRES, Payload

pytestmark = pytest.mark.security


def _run(credentials: dict, sql: str):
    connector = granted(PostgresConnector())
    connector.connect(credentials)
    try:
        return connector.execute_query(sql)
    finally:
        connector.close()


#: Payloads the layers shipped so far do not stop. Each names the wave that will.
#: Remove an entry when its wave lands; strict xfail will insist.
STILL_OPEN = {
    "advisory_lock": "SEC-02 — needs the SQL policy (W4); a read-only transaction permits locks",
    "sleep": "SEC-02 — needs the SQL policy (W4); bounded only by statement_timeout",
    "server_file_read": "SEC-02 — needs the SQL policy (W4) or a role without pg_read_server_files",
    # `row_lock` was listed here and the lab refused it on the first run:
    # PostgreSQL declines `SELECT ... FOR UPDATE` in a read-only transaction,
    # which was assumed rather than measured. Strict xfail turned that into a
    # build failure instead of a note nobody would have written down.
}


@pytest.mark.parametrize("payload", POSTGRES, ids=lambda p: p.id)
def test_payload_has_no_effect(payload: Payload, privileged_credentials, marker, request):
    """Run it as the *privileged* login, so this measures the application alone.

    The database owner is what a deployment has before anyone reads the
    hardening guide. Anything stopped here is stopped by NLQueries; anything
    that gets through depends on the operator having done the other half.
    """
    if payload.id in STILL_OPEN:
        request.node.add_marker(pytest.mark.xfail(reason=STILL_OPEN[payload.id], strict=True))

    before = marker.sequence_value()
    result = _run(privileged_credentials, payload.sql)

    assert result.error is not None, f"{payload.id} executed: {payload.effect}"
    assert marker.rows() == [], f"{payload.id} left a marker: {payload.effect}"
    assert marker.sequence_value() == before, f"{payload.id} advanced the sequence"


@pytest.mark.parametrize("sql", SAFE_POSTGRES, ids=lambda s: s[:38])
def test_ordinary_analytics_still_run(sql: str, privileged_credentials):
    """The control every one of these layers has to pass.

    A policy that refuses the payloads and also refuses a GROUP BY has not made
    the product safer, it has made it useless — and that is the failure mode
    somebody fixes by turning the control off.
    """
    result = _run(privileged_credentials, sql)

    assert result.error is None, f"a safe query was refused: {result.error}"


def test_the_lab_can_actually_record_an_effect(lab_admin, marker):
    """A negative test suite has to prove its own instrument works.

    If `lab.mark()` silently did nothing, every assertion above would pass for
    the wrong reason and this file would be theatre.
    """
    assert marker.rows() == []

    with lab_admin.connect() as conn:
        conn.execute(sa.text("SELECT lab.mark('instrument-check')"))
        conn.execute(sa.text("COMMIT"))

    assert marker.rows() == ["instrument-check"]


def test_least_privilege_login_cannot_reach_the_marker(least_privilege_credentials):
    """The operator's half, from `docs/database-hardening.md`.

    The guide's role has no EXECUTE on lab.mark and no SELECT on lab.marker, so
    the payload fails on privileges even where the application would have let it
    through. This is what "layered" means concretely.
    """
    result = _run(least_privilege_credentials, "SELECT lab.mark('via-readonly')")

    assert result.error is not None
    assert "permission denied" in result.error.lower()
