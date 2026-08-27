"""
`--no-execute` and generation-only evaluation must not touch a database.

Both were reported as executing anyway. The reproduction here is deterministic
and needs no model: the SQL-generation step is stubbed to return a fixed
statement, and the connector loader is replaced by a spy that records every
attempt to open a connection. What is being measured is not whether a query
succeeded — it is whether a connector was constructed at all.

That is the right question, because the fix is a capability the caller mints and
the connector demands. A test asserting "no rows came back" would pass against a
connector that connected, ran the statement, and had its result discarded.

Both were `xfail(strict=True)` until the execution capability landed, at which
point they XPASSed and the build failed until the markers came out — which is
what the strictness is for. They are ordinary tests now, and they stay: a
regression here would put a language model's output back on a database that a
generation-only run was never meant to touch.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner
from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.sync_runner import AgentQueryResult

pytestmark = pytest.mark.security

_SQL = "SELECT count(*) FROM lab.orders"


def _result(
    sql: str | None = _SQL,
    sql_result: QueryResult | None = None,
    *,
    sql_is_valid: bool = True,
) -> AgentQueryResult:
    """What the orchestrator hands the CLI."""
    return AgentQueryResult(
        question="how many orders?",
        resolved_question="how many orders?",
        agent_type="sql",
        answer="",
        sql=sql,
        sql_result=sql_result,
        citations=[],
        merged_answer=None,
        latency_ms=1,
        session_id=None,
        sql_is_valid=sql_is_valid,
        execution_mode="execute_read_only",
    )


def test_no_execute_reaches_the_orchestrator() -> None:
    """SEC-07.

    The exposure is not that the CLI runs the query — with `--no-execute` its own
    branch is skipped. It is that `run_query_sync` executes *before* the flag is
    ever consulted, so the intent never leaves the command function.

    So the thing to assert is that the intent is communicated at all. Anything
    else tests a stub: a spy the test itself calls proves nothing, and a
    connector count taken around the CLI branch measures the half that already
    works.

    `run_query_sync(question, agent_id, **kwargs)` takes no execution parameter
    today, so nothing in what it receives can distinguish a generate-only run
    from an ordinary one. After W2 that becomes an explicit capability.
    """
    from nlqueries.cli.main import cli

    received: dict[str, object] = {}

    def _capture(question: str, agent_id: str, **kwargs: object) -> AgentQueryResult:
        received.update(kwargs)
        return _result(sql_result=QueryResult(["count"], [[2]], 1, 1.0, None))

    with (
        patch("nlqueries.orchestrator.sync_runner.run_query_sync", side_effect=_capture),
        patch("nlqueries.cli.main._resolve_alias", side_effect=lambda a: a),
    ):
        outcome = CliRunner().invoke(cli, ["query", "demo_agent", "how many?", "--no-execute"])

    assert outcome.exit_code == 0
    assert any("execut" in key or "polic" in key for key in received), (
        "--no-execute never reaches the orchestrator: it was called with "
        f"{sorted(received) or 'no keyword arguments'}, so nothing downstream "
        "can tell a generate-only run from an ordinary one"
    )


def test_the_cli_will_not_run_sql_the_validator_rejected() -> None:
    """SEC-08 — the July review's finding 3, still live on this path.

    `orchestrator.py` gates its own execution on `is_valid`. `AgentQueryResult`
    carries no validity field, so by the time the CLI sees the result the only
    signal left is `sql_result is None` — which is exactly the state a rejected
    statement produces. The orchestrator's fix made this *more* reachable.
    """
    from nlqueries.cli.main import cli

    executed: list[str] = []

    class _Connector:
        def connect(self, credentials: dict) -> None:
            pass

        def execute_query(self, sql: str, timeout_seconds: float | None = None) -> QueryResult:
            executed.append(sql)
            return QueryResult(["x"], [[1]], 1, 1.0, None)

    rejected = "SELECT lab.mark('validator-said-no')"

    with (
        patch(
            "nlqueries.orchestrator.sync_runner.run_query_sync",
            return_value=_result(sql=rejected, sql_result=None, sql_is_valid=False),
        ),
        patch("nlqueries.cli.main._resolve_alias", side_effect=lambda a: a),
        patch(
            "nlqueries.cli.main._require_connector",
            return_value={"db_type": "postgres", "url": "postgresql://u:p@127.0.0.1:1/db"},
        ),
        patch.dict("nlqueries.cli.main.CONNECTOR_REGISTRY", {"postgres": _Connector}, clear=False),
        patch("nlqueries.cli.main._load_password", return_value=""),
    ):
        CliRunner().invoke(cli, ["query", "demo_agent", "how many?"])

    assert executed == [], (
        "the CLI executed SQL the validator rejected: validity is not carried on "
        "AgentQueryResult, so `sql_result is None` reads as 'not run yet' rather "
        "than 'refused'"
    )


class TestTheConnectorRefusesOnItsOwn:
    """Enforcement independent of the orchestration layer.

    Orchestration refuses to reach a connector without permission. This class
    verifies the check beneath it: a connector with no granted policy will not
    execute a statement, regardless of the caller.
    """

    def _connector(self):
        from nlqueries.connectors.sqlite import SQLiteConnector

        connector = SQLiteConnector()
        connector.connect({"database": ":memory:"})
        return connector

    def test_an_ungranted_connector_refuses(self) -> None:
        from nlqueries.execution import ExecutionNotPermitted

        connector = self._connector()

        with pytest.raises(ExecutionNotPermitted) as excinfo:
            connector.execute_query("SELECT 1")

        # The message has to name what to do, because whoever reads it is
        # holding a connector and wondering why it will not work.
        assert "generate_only" in str(excinfo.value)
        assert "execute_read_only" in str(excinfo.value)

    def test_a_granted_connector_runs(self) -> None:
        from nlqueries.execution import ExecutionPolicy

        connector = self._connector()
        connector.bind_execution_policy(ExecutionPolicy.execute_read_only())

        assert connector.execute_query("SELECT 1").error is None

    def test_granting_generate_only_is_not_granting(self) -> None:
        """`bind_execution_policy` is not a synonym for permission."""
        from nlqueries.execution import ExecutionNotPermitted, ExecutionPolicy

        connector = self._connector()
        connector.bind_execution_policy(ExecutionPolicy.generate_only())

        with pytest.raises(ExecutionNotPermitted):
            connector.execute_query("SELECT 1")

    def test_permission_does_not_leak_between_requests(self) -> None:
        """The reason permission lives on a per-request wrapper.

        Connectors are pooled and shared across in-flight requests. If the
        policy lived on the shared object, one request granted execution would
        grant it to every other request holding the same connector — and nothing
        about the shared connector would look different.
        """
        from nlqueries.connectors.base import PermittedConnector
        from nlqueries.execution import ExecutionNotPermitted, ExecutionPolicy

        shared = self._connector()
        allowed = PermittedConnector(shared, ExecutionPolicy.execute_read_only())
        denied = PermittedConnector(shared, ExecutionPolicy.generate_only())

        assert allowed.execute_query("SELECT 1").error is None
        with pytest.raises(ExecutionNotPermitted):
            denied.execute_query("SELECT 1")
        # And the grant did not rub off on the pooled object itself.
        with pytest.raises(ExecutionNotPermitted):
            shared.execute_query("SELECT 1")
