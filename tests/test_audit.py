"""
Audit events: authorisation decisions rendered with their JSON, and one record
per statement sent through ``PermittedConnector``.

The execution record is asserted to carry a digest of the statement and never
its text, so a distinctive literal is used throughout and searched for in every
place the record is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import stat
import sys
import threading
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from nlqueries import audit, config
from nlqueries.auth.authorizer import ConfigAllowlistAuthorizer, Grant
from nlqueries.auth.enforcement import guard
from nlqueries.auth.principal import (
    Action,
    AuditEvent,
    Principal,
    Source,
    local_principal,
    record,
)
from nlqueries.connectors import loader
from nlqueries.connectors.base import PermittedConnector
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionNotPermitted, ExecutionPolicy
from nlqueries.sql_policy import POLICY_VERSION

#: Appears in every statement below, and must appear in no record.
_SECRET = "needle-7f3a"
_SQL = f"SELECT '{_SECRET}' AS marker"

#: Whether the permission bits are enforced. On Windows ``chmod`` sets only the
#: read-only flag and the mode reads 0o666, so the mode checks run elsewhere;
#: the line and append checks run everywhere.
_POSIX = sys.platform != "win32"

ALICE = Principal(subject="alice@example.com", source=Source.OIDC)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No file sink and no process context unless a test sets one."""
    monkeypatch.setattr(config, "AUDIT_LOG", "")
    monkeypatch.setattr(audit, "_process_context", None)
    monkeypatch.setattr(audit, "_reported", set())
    monkeypatch.setattr(audit, "_failures_reported", set())


def _permitted(agent_id: str | None = "sales", policy: ExecutionPolicy | None = None):
    inner = SQLiteConnector()
    inner.connect({"database": ":memory:"})
    return PermittedConnector(
        inner, policy or ExecutionPolicy.execute_read_only(), agent_id=agent_id
    )


def _events(caplog, event: str) -> list[dict[str, Any]]:
    return [r.audit for r in caplog.records if getattr(r, "audit", {}).get("event") == event]


def _json_of(message: str) -> dict[str, Any]:
    """The JSON a rendered audit message carries after its words."""
    return json.loads(message[message.index("{") :])


class TestAuthorizationRendering:
    def test_the_message_carries_the_events_fields(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            record(
                AuditEvent(
                    principal="alice@example.com",
                    source=Source.OIDC,
                    action=Action.QUERY_EXECUTE,
                    agent_id="payroll",
                    decision=AuditEvent.DENY,
                    reason="no grant covers this action on this agent",
                )
            )

        (logged,) = caplog.records
        message = logged.getMessage()
        assert message.startswith("authorization denied {")
        rendered = _json_of(message)
        assert rendered["event"] == "authorization"
        assert rendered["schema_version"] == audit.SCHEMA_VERSION
        assert rendered["principal"] == "alice@example.com"
        assert rendered["source"] == "oidc"
        assert rendered["action"] == "query:execute"
        assert rendered["agent_id"] == "payroll"
        assert rendered["decision"] == "deny"
        assert rendered["reason"] == "no grant covers this action on this agent"
        assert rendered["at"]
        # The same record as `extra`, which existing handlers read.
        assert logged.audit == rendered
        assert logged.levelno == logging.WARNING

    def test_the_json_has_sorted_keys(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            record(
                AuditEvent(
                    principal="alice",
                    source=Source.LOCAL,
                    action=Action.HEALTH_READ,
                    decision=AuditEvent.ALLOW,
                )
            )

        message = caplog.records[0].getMessage()
        body = message[message.index("{") :]
        assert body == json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"))
        assert caplog.records[0].levelno == logging.INFO


class TestExecutionEvents:
    def test_one_event_with_a_digest_and_no_sql_text(self, caplog) -> None:
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            result = connector.execute_query(_SQL)

        assert result.rows == [[_SECRET]]
        (event,) = _events(caplog, "query_execution")
        assert event["sql_sha256"] == hashlib.sha256(_SQL.encode()).hexdigest()
        assert event["sql_bytes"] == len(_SQL.encode())
        assert event["agent_id"] == "sales"
        assert event["connector_type"] == "sqlite"
        assert event["dialect"] == "sqlite"
        assert event["policy_version"] == POLICY_VERSION
        assert event["policy_decision"] == "allow"
        assert event["outcome"] == "ok"
        assert event["row_count"] == 1
        assert event["truncated"] is False
        assert event["error_class"] is None
        assert event["duration_ms"] >= 0
        assert event["surface"] == "library"
        assert len(event["request_id"]) == 32
        # Nowhere: not the record, not the message.
        for logged in caplog.records:
            assert _SECRET not in logged.getMessage()
            assert _SECRET not in json.dumps(getattr(logged, "audit", {}))

    def test_a_database_error_is_recorded_as_one(self, caplog) -> None:
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            result = connector.execute_query(f"SELECT * FROM missing WHERE x = '{_SECRET}'")

        assert result.error
        (event,) = _events(caplog, "query_execution")
        assert event["outcome"] == "error"
        assert event["row_count"] == 0
        assert _SECRET not in caplog.text

    def test_a_refusal_is_recorded_with_its_class_and_still_raised(self, caplog) -> None:
        connector = _permitted(policy=ExecutionPolicy.generate_only())

        with (
            caplog.at_level(logging.INFO, logger="nlqueries.audit"),
            pytest.raises(ExecutionNotPermitted),
        ):
            connector.execute_query(_SQL)

        (event,) = _events(caplog, "query_execution")
        assert event["outcome"] == "exception"
        assert event["error_class"] == "ExecutionNotPermitted"
        assert event["row_count"] is None

    def test_a_statement_the_policy_refuses_is_recorded_as_denied(self, caplog) -> None:
        """Recorded, not enforced: the policy is applied before execution by
        the orchestrator. The record says what it would have decided."""
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            connector.execute_query("PRAGMA user_version")

        (event,) = _events(caplog, "query_execution")
        assert event["policy_decision"] == "deny"

    def test_the_bound_principal_and_surface_are_recorded(self, caplog) -> None:
        connector = _permitted()

        with (
            caplog.at_level(logging.INFO, logger="nlqueries.audit"),
            audit.bound(audit.Surface.MCP, ALICE) as context,
        ):
            connector.execute_query(_SQL)
            connector.execute_query(_SQL)

        events = _events(caplog, "query_execution")
        assert len(events) == 2
        for event in events:
            assert event["surface"] == "mcp"
            assert event["principal"] == "alice@example.com"
            assert event["source"] == "oidc"
            assert event["request_id"] == context.request_id

    def test_a_thread_without_the_context_gets_the_process_one(self, caplog) -> None:
        audit.bind_process(audit.Surface.CLI, local_principal("alice"))
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            worker = threading.Thread(target=connector.execute_query, args=(_SQL,))
            worker.start()
            worker.join()

        (event,) = _events(caplog, "query_execution")
        assert event["surface"] == "cli"
        assert event["principal"] == "alice"

    def test_an_agent_opened_through_the_loader_is_named(
        self, caplog, tmp_path, monkeypatch
    ) -> None:
        database = tmp_path / "app.db"
        sqlite3.connect(database).close()
        connectors = tmp_path / "connectors.yaml"
        connectors.write_text(
            yaml.safe_dump({"sqlite:app": {"db_type": "sqlite", "url": f"sqlite:///{database}"}})
        )
        monkeypatch.setattr(config, "CONNECTORS_FILE", connectors)
        monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)
        connector = loader.open_connector_for_agent(
            "sqlite:app", ExecutionPolicy.execute_read_only()
        )
        assert connector is not None

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            connector.execute_query(_SQL)

        (event,) = _events(caplog, "query_execution")
        assert event["agent_id"] == "sqlite:app"
        assert event["connector_type"] == "sqlite"

    def test_a_sqlalchemy_connector_reports_its_backend_dialect(self, tmp_path) -> None:
        """Its type is `sqlalchemy` whatever it reaches; the backend names the grammar."""
        database = tmp_path / "app.db"
        sqlite3.connect(database).close()
        connector = SQLAlchemyConnector()
        assert audit.connector_identity(connector) == ("sqlalchemy", None)

        connector.connect({"url": f"sqlite:///{database}"})

        assert connector.dialect_name == "sqlite"
        assert audit.connector_identity(connector) == ("sqlalchemy", "sqlite")

    def test_a_failure_to_record_does_not_fail_the_query(self, caplog, monkeypatch) -> None:
        def broken(*_args: Any) -> dict[str, Any]:
            raise RuntimeError("recording broke")

        monkeypatch.setattr(audit, "_execution_fields", broken)
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            result = connector.execute_query(_SQL)

        assert result.rows == [[_SECRET]]
        assert any("could not be recorded" in r.getMessage() for r in caplog.records)

    def test_a_repeated_failure_to_record_warns_once(self, caplog, monkeypatch) -> None:
        """A fault every query hits is one traceback at WARNING, then DEBUG."""

        def broken(*_args: Any) -> dict[str, Any]:
            raise RuntimeError("recording broke")

        monkeypatch.setattr(audit, "_execution_fields", broken)
        connector = _permitted()

        with caplog.at_level(logging.DEBUG, logger="nlqueries.audit"):
            connector.execute_query(_SQL)
            connector.execute_query(_SQL)

        said = [r for r in caplog.records if "could not be recorded" in r.getMessage()]
        assert [r.levelno for r in said] == [logging.WARNING, logging.DEBUG]
        assert all(r.exc_info for r in said)

    def test_a_different_cause_gets_its_own_warning(self, caplog, monkeypatch) -> None:
        def broken(*_args: Any) -> dict[str, Any]:
            raise RuntimeError("recording broke")

        def broken_elsewhere(*_args: Any) -> dict[str, Any]:
            raise KeyError("a field")

        connector = _permitted()
        with caplog.at_level(logging.DEBUG, logger="nlqueries.audit"):
            monkeypatch.setattr(audit, "_execution_fields", broken)
            connector.execute_query(_SQL)
            monkeypatch.setattr(audit, "_execution_fields", broken_elsewhere)
            connector.execute_query(_SQL)

        said = [r for r in caplog.records if "could not be recorded" in r.getMessage()]
        assert [r.levelno for r in said] == [logging.WARNING, logging.WARNING]

    def test_nothing_is_evaluated_when_nobody_is_listening(self, monkeypatch) -> None:
        calls: list[str] = []
        import nlqueries.sql_policy as sql_policy

        real = sql_policy.evaluate
        monkeypatch.setattr(
            sql_policy, "evaluate", lambda sql, dialect: calls.append(sql) or real(sql, dialect)
        )
        monkeypatch.setattr(logging.getLogger("nlqueries.audit"), "level", logging.WARNING)

        _permitted().execute_query(_SQL)

        assert calls == []


class TestTheFileSink:
    def test_the_file_is_0600_and_holds_one_line_per_event(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))
        connector = _permitted()

        connector.execute_query(_SQL)
        record(
            AuditEvent(
                principal="alice",
                source=Source.LOCAL,
                action=Action.QUERY_EXECUTE,
                agent_id="sales",
                decision=AuditEvent.ALLOW,
            )
        )
        connector.execute_query(_SQL)

        if _POSIX:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        lines = path.read_text().splitlines()
        assert [json.loads(line)["event"] for line in lines] == [
            "query_execution",
            "authorization",
            "query_execution",
        ]
        assert _SECRET not in path.read_text()

    def test_each_line_ends_with_exactly_a_line_feed(self, tmp_path, monkeypatch) -> None:
        """Read as bytes: a text-mode descriptor on Windows would write CRLF."""
        path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))
        connector = _permitted()

        connector.execute_query(_SQL)
        connector.execute_query(_SQL)

        data = path.read_bytes()
        assert b"\r" not in data
        assert data.endswith(b"}\n")
        assert data.count(b"\n") == 2
        assert [json.loads(line)["event"] for line in data.split(b"\n")[:-1]] == [
            "query_execution",
            "query_execution",
        ]

    def test_an_existing_file_is_appended_to_and_narrowed(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text('{"earlier":true}\n')
        path.chmod(0o644)
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))

        _permitted().execute_query(_SQL)

        if _POSIX:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        lines = path.read_text().splitlines()
        assert lines[0] == '{"earlier":true}'
        assert json.loads(lines[1])["event"] == "query_execution"

    def test_a_file_that_cannot_be_written_is_reported_once(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        # A directory: opening it for writing fails.
        monkeypatch.setattr(config, "AUDIT_LOG", str(tmp_path))
        connector = _permitted()

        with caplog.at_level(logging.WARNING, logger="nlqueries.audit.sink"):
            first = connector.execute_query(_SQL)
            second = connector.execute_query(_SQL)

        assert first.rows == second.rows == [[_SECRET]]
        warned = [r for r in caplog.records if "could not be written" in r.getMessage()]
        assert len(warned) == 1


@pytest.mark.skipif(not _POSIX, reason="POSIX symlinks, owners and permission bits")
class TestTheFileSinkRefusesFilesItCannotVouchFor:
    def _refused(self, caplog) -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == "nlqueries.audit.sink"]

    def test_a_symlink_at_the_path_is_not_followed(self, tmp_path, monkeypatch, caplog) -> None:
        target = tmp_path / "elsewhere.txt"
        target.write_text("keep\n")
        target.chmod(0o600)
        link = tmp_path / "audit.jsonl"
        link.symlink_to(target)
        monkeypatch.setattr(config, "AUDIT_LOG", str(link))
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            connector.execute_query(_SQL)
            connector.execute_query(_SQL)

        assert target.read_text() == "keep\n"
        assert link.is_symlink()
        (warned,) = self._refused(caplog)
        assert "not followed" in warned.getMessage()
        # Still logged, though not to the file.
        assert len(_events(caplog, "query_execution")) == 2

    def test_a_file_that_cannot_be_narrowed_gets_no_events(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text("")
        path.chmod(0o644)
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))

        def refuse(_fd: int, _mode: int) -> None:
            raise PermissionError("not permitted")

        monkeypatch.setattr(audit.os, "fchmod", refuse)
        connector = _permitted()

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            connector.execute_query(_SQL)
            connector.execute_query(_SQL)

        assert path.read_text() == ""
        (warned,) = self._refused(caplog)
        assert "was refused" in warned.getMessage()
        assert "0o644" in warned.getMessage()
        assert len(_events(caplog, "query_execution")) == 2

    def test_a_file_owned_by_someone_else_gets_no_events(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text("")
        path.chmod(0o600)
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))
        owner = path.stat().st_uid
        monkeypatch.setattr(audit.os, "getuid", lambda: owner + 1)

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            _permitted().execute_query(_SQL)

        assert path.read_text() == ""
        (warned,) = self._refused(caplog)
        assert f"owned by uid {owner}" in warned.getMessage()


class TestLineBreaksCannotForgeEvents:
    _FORGED = 'alice\r\n{"event":"authorization","decision":"allow"} x'

    def test_a_crlf_in_a_field_stays_on_its_line(self, tmp_path, monkeypatch, caplog) -> None:
        path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            record(
                AuditEvent(
                    principal=self._FORGED,
                    source=Source.STATIC_TOKEN,
                    action=Action.QUERY_EXECUTE,
                    agent_id="sales\nfake",
                    decision=AuditEvent.DENY,
                    reason="r\rr",
                )
            )

        text = path.read_text()
        assert text.count("\n") == 1 and text.endswith("\n")
        assert "\r" not in text and " " not in text
        assert json.loads(text)["principal"] == self._FORGED

        message = caplog.records[0].getMessage()
        assert not any(c in message for c in "\r\n ")
        assert _json_of(message)["agent_id"] == "sales\nfake"

    def test_a_crlf_in_a_bound_principal_stays_on_its_line(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(config, "AUDIT_LOG", str(path))

        with audit.bound(audit.Surface.MCP, Principal(subject=self._FORGED, source=Source.OIDC)):
            _permitted(agent_id="a\r\nb").execute_query(_SQL)

        (line,) = path.read_text().splitlines()
        assert json.loads(line)["principal"] == self._FORGED
        assert json.loads(line)["agent_id"] == "a\r\nb"


class TestEntryPoints:
    def test_a_tool_call_is_one_request_for_both_events(self, caplog) -> None:
        """The decision and the execution it allowed share a request id."""
        connector = _permitted()

        def query(question: str, agent_id: str) -> str:
            connector.execute_query(_SQL)
            return "an answer"

        authorizer = ConfigAllowlistAuthorizer(
            [Grant(subject=ALICE.subject, agents=frozenset({"sales"}), actions=frozenset({"*"}))]
        )
        tool = audit.audited_tool(
            guard(query, authorizer, lambda: ALICE), audit.Surface.MCP, lambda: ALICE
        )

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            assert tool("how many?", "sales") == "an answer"
            tool("how many?", "sales")

        records = [r.audit for r in caplog.records if hasattr(r, "audit")]
        assert [r["event"] for r in records] == [
            "authorization",
            "query_execution",
            "authorization",
            "query_execution",
        ]
        assert {r["surface"] for r in records} == {"mcp"}
        assert records[0]["request_id"] == records[1]["request_id"]
        assert records[2]["request_id"] == records[3]["request_id"]
        assert records[0]["request_id"] != records[2]["request_id"]
        assert records[1]["principal"] == ALICE.subject

    @pytest.mark.anyio
    async def test_an_async_tool_keeps_its_signature_and_context(self) -> None:
        import inspect

        async def query(question: str, agent_id: str, explain: bool = False) -> str:
            context = audit.current()
            return f"{context.surface}:{context.principal}"

        tool = audit.audited_tool(query, audit.Surface.MCP, lambda: ALICE)

        assert inspect.signature(tool) == inspect.signature(query)
        assert tool.__name__ == "query"
        assert await tool("q", "sales") == "mcp:alice@example.com"
        assert audit.current().surface is audit.Surface.LIBRARY

    def test_the_built_server_records_its_surface(self, caplog) -> None:
        from mcp.server.fastmcp import FastMCP
        from nlqueries.mcp_server import server as mcp_server
        from starlette.testclient import TestClient

        server: FastMCP = mcp_server._build_server()
        server.settings.stateless_http = True
        server.settings.json_response = True

        with (
            caplog.at_level(logging.INFO, logger="nlqueries.audit"),
            TestClient(server.streamable_http_app(), base_url="http://127.0.0.1:8000") as client,
        ):
            client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "list_agents", "arguments": {}},
                },
                headers={"Accept": "application/json, text/event-stream"},
            )

        (decision,) = _events(caplog, "authorization")
        assert decision["surface"] == "mcp"
        assert len(decision["request_id"]) == 32

    def test_the_cli_binds_its_surface_for_the_process(self, tmp_path, monkeypatch) -> None:
        from nlqueries.cli.main import cli

        monkeypatch.setattr(config, "CONNECTORS_FILE", tmp_path / "connectors.yaml")
        monkeypatch.setattr("getpass.getuser", lambda: "operator")

        CliRunner().invoke(cli, ["connectors"])

        context = audit.current()
        assert context.surface is audit.Surface.CLI
        assert context.principal == "operator"
        assert context.source == "local"

    def test_a_user_that_cannot_be_named_does_not_stop_the_cli(self, tmp_path, monkeypatch) -> None:
        """`getuser` raises ModuleNotFoundError on 3.11 and 3.12 when there is
        no `pwd` module and no user variable to fall back on."""
        from nlqueries.cli.main import cli

        def no_user() -> str:
            raise ModuleNotFoundError("No module named 'pwd'")

        monkeypatch.setattr(config, "CONNECTORS_FILE", tmp_path / "connectors.yaml")
        monkeypatch.setattr("getpass.getuser", no_user)

        result = CliRunner().invoke(cli, ["connectors"])

        assert result.exit_code == 0, result.output
        context = audit.current()
        assert context.surface is audit.Surface.CLI
        assert context.principal is None
        assert context.source is None
