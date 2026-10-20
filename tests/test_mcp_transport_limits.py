"""
The MCP server's binding, transport checks and input limits.

Five things, each about what reaches a tool and as whom:

- a call that arrived over HTTP is never the process owner, even on a server
  built for stdio;
- every spelling of "all interfaces" is refused like `0.0.0.0`;
- a listener bound beyond loopback answers only to the names it is published
  under, and the SSE transport caps request bodies like streamable HTTP does;
- over-long questions and SQL are refused before a tool does anything;
- `health` names agents only to a caller who may list them, and "not found"
  names none.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries.auth.principal import Principal, Source
from nlqueries.mcp_server import server as mcp_server
from starlette.testclient import TestClient

ENV_VARS = (
    "NLQ_MCP_RESOURCE_URL",
    "NLQ_MCP_OIDC_DISCOVERY_URL",
    "NLQ_MCP_OIDC_CLIENT_ID",
    "NLQ_MCP_STATIC_TOKEN",
    "NLQ_MCP_STATIC_TOKEN_FILE",
    "NLQ_MCP_STATIC_SUBJECT",
    "NLQ_ALLOW_UNAUTHENTICATED_MCP",
    "NLQ_ALLOW_INSECURE_BIND",
    "NLQ_MCP_GRANTS_FILE",
    "NLQ_MCP_MAX_QUESTION_CHARS",
    "NLQ_MCP_MAX_SQL_CHARS",
)

GOOD_TOKEN = "a" * 32


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _grants(tmp_path: Path, actions: list[str], subject: str = "alice") -> Path:
    path = tmp_path / "grants.yaml"
    path.write_text(
        f"grants:\n  - subject: {subject}\n    agents: ['*']\n    actions: {actions!r}\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1. An HTTP caller is not the process owner
# ---------------------------------------------------------------------------


@contextmanager
def _request_context(request: object | None) -> Iterator[None]:
    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext

    token = request_ctx.set(
        RequestContext(
            request_id=1, meta=None, session=MagicMock(), lifespan_context=None, request=request
        )
    )
    try:
        yield
    finally:
        request_ctx.reset(token)


class TestAnHttpCallerIsNotTheProcessOwner:
    """A server built for stdio resolves a caller with no token to the local
    process owner. Served over HTTP -- the module-level `mcp` mounted in an app
    -- that named every remote caller as the account the server runs as, and
    the local profile granted them everything."""

    def test_a_request_over_http_with_no_token_is_anonymous(self) -> None:
        resolve = mcp_server._principal_for_call(networked=False)

        with _request_context(request=MagicMock(name="starlette-request")):
            principal = resolve()

        assert principal.source is Source.ANONYMOUS
        assert not principal.is_local

    def test_a_request_over_stdio_is_still_the_process_owner(self) -> None:
        """stdio requests carry no HTTP request; Claude Desktop must keep working."""
        resolve = mcp_server._principal_for_call(networked=False)

        with _request_context(request=None):
            principal = resolve()

        assert principal.is_local

    def test_a_stdio_server_served_over_http_refuses_the_caller(self, caplog) -> None:
        """End to end: the module's stdio server mounted as an HTTP app. Before,
        the call ran as the process owner; now the caller is anonymous and the
        local profile refuses them."""
        server = mcp_server._build_server()
        server.settings.stateless_http = True
        server.settings.json_response = True

        with (
            caplog.at_level(logging.INFO, logger="nlqueries.audit"),
            TestClient(server.streamable_http_app(), base_url="http://127.0.0.1:8000") as client,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "list_agents", "arguments": {}},
                },
                headers={"Accept": "application/json, text/event-stream"},
            )

        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["isError"] is True
        assert "Not authorized" in result["content"][0]["text"]
        audits = [r.audit for r in caplog.records if hasattr(r, "audit")]
        assert [(a["principal"], a["decision"]) for a in audits] == [("anonymous", "deny")]


# ---------------------------------------------------------------------------
# 2. Every spelling of a wildcard is a wildcard
# ---------------------------------------------------------------------------

WILDCARD_SPELLINGS = [
    "0",
    "0.0",
    "0.0.0",
    "00",
    "0x0",
    "::0",
    "0:0:0:0:0:0:0:0",
    "::ffff:0.0.0.0",
    "[::]",
    " 0.0.0.0 ",
]


class TestEveryWildcardSpellingIsRefused:
    @pytest.mark.parametrize("host", WILDCARD_SPELLINGS)
    def test_it_is_refused_without_the_switch(self, host: str) -> None:
        with pytest.raises(SystemExit) as excinfo:
            mcp_server._refuse_unauthenticated_exposure(host)

        assert mcp_server._INSECURE_BIND_ENV in str(excinfo.value)

    @pytest.mark.parametrize("host", WILDCARD_SPELLINGS)
    def test_it_is_refused_with_authentication_on(self, host: str) -> None:
        with pytest.raises(SystemExit, match="authentication is on"):
            mcp_server._refuse_unauthenticated_exposure(host, authenticated=True)

    @pytest.mark.parametrize("host", WILDCARD_SPELLINGS)
    def test_the_switch_still_permits_it(self, host: str, monkeypatch) -> None:
        monkeypatch.setenv(mcp_server._INSECURE_BIND_ENV, "1")

        mcp_server._refuse_unauthenticated_exposure(host)

    @pytest.mark.parametrize(
        "host", ["127.1", "::ffff:127.0.0.1", "10.0.0.1", "0.0.0.1", "mcp.example.com"]
    )
    def test_a_specific_address_is_not_a_wildcard(self, host: str) -> None:
        mcp_server._refuse_unauthenticated_exposure(host)

    def test_a_mapped_address_is_read_as_the_ipv4_address(self) -> None:
        """Whether `ipaddress` calls `::ffff:0.0.0.0` unspecified differs between
        Python patch releases, so the mapping is done here rather than left to
        it."""
        import ipaddress

        assert mcp_server._address("::ffff:0.0.0.0") == ipaddress.IPv4Address("0.0.0.0")
        assert mcp_server._address("::ffff:127.0.0.1") == ipaddress.IPv4Address("127.0.0.1")

    def test_main_refuses_a_shorthand_wildcard(self, monkeypatch) -> None:
        """Through the entry point, which is where the CLI's `--host` arrives."""
        monkeypatch.setattr(mcp_server, "_require_authentication", lambda transport: True)
        monkeypatch.setattr(mcp_server, "_build_server", lambda **kwargs: pytest.fail("built"))

        with pytest.raises(SystemExit):
            mcp_server.main(transport="sse", host="0")


# ---------------------------------------------------------------------------
# 3. Host and Origin checks, and the SSE body cap
# ---------------------------------------------------------------------------


def _networked(monkeypatch, tmp_path: Path, *, resource: str | None) -> Any:
    """A networked server on 0.0.0.0. With a resource URL it authenticates with
    a static token; without one it runs with authentication switched off."""
    if resource is None:
        monkeypatch.setenv("NLQ_ALLOW_UNAUTHENTICATED_MCP", "1")
        return mcp_server._build_server(host="0.0.0.0", port=8080, networked=True)
    monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("NLQ_MCP_STATIC_SUBJECT", "alice")
    monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", resource)
    monkeypatch.setenv("NLQ_MCP_GRANTS_FILE", str(_grants(tmp_path, ["*"])))
    return mcp_server._build_server(host="0.0.0.0", port=8080, authenticated=True, networked=True)


def _post_message(server: Any, host: str, **headers: str) -> int:
    """POST to the SSE message endpoint for a session that does not exist.

    The Host and Origin checks run before the session lookup, so a refused
    header gives 421 or 403 and an accepted one falls through to 400 or 404.
    """
    client = TestClient(server.sse_app())
    response = client.post(
        "/messages/?session_id=" + "0" * 32,
        content=b"{}",
        headers={"Host": host, "Content-Type": "application/json", **headers},
    )
    return response.status_code


class TestHostChecksOnANonLoopbackBind:
    def test_a_foreign_host_gets_421(self, monkeypatch, tmp_path) -> None:
        server = _networked(monkeypatch, tmp_path, resource=None)

        assert _post_message(server, "evil.example") == 421

    @pytest.mark.parametrize("host", ["localhost:8080", "127.0.0.1:8080", "127.0.0.1:9999"])
    def test_the_loopback_names_are_accepted(self, host: str, monkeypatch, tmp_path) -> None:
        server = _networked(monkeypatch, tmp_path, resource=None)

        assert _post_message(server, host) not in (421, 403)

    def test_the_resource_host_is_accepted_and_others_are_not(self, monkeypatch, tmp_path) -> None:
        server = _networked(monkeypatch, tmp_path, resource="https://mcp.example.com/mcp")
        bearer = {"Authorization": f"Bearer {GOOD_TOKEN}"}

        assert _post_message(server, "mcp.example.com", **bearer) not in (421, 403)
        assert _post_message(server, "other.example.com", **bearer) == 421

    def test_only_the_resource_origin_is_accepted(self, monkeypatch, tmp_path) -> None:
        server = _networked(monkeypatch, tmp_path, resource="https://mcp.example.com")
        bearer = {"Authorization": f"Bearer {GOOD_TOKEN}"}

        good = _post_message(server, "mcp.example.com", Origin="https://mcp.example.com", **bearer)
        bad = _post_message(server, "mcp.example.com", Origin="https://evil.example", **bearer)

        assert good not in (421, 403)
        assert bad == 403

    @pytest.mark.parametrize("host", ["localhost:8080", "127.0.0.1:8080"])
    def test_the_compose_deployment_keeps_working(self, host: str, monkeypatch, tmp_path) -> None:
        """docker-compose.yml publishes on 127.0.0.1 and sets the resource URL to
        http://localhost:8080; clients use either name."""
        server = _networked(monkeypatch, tmp_path, resource="http://localhost:8080")

        status = _post_message(server, host, Authorization=f"Bearer {GOOD_TOKEN}")

        assert status not in (421, 403)

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "127.0.0.1",
            "[::1]",
            "[::1]:8080",
            "mcp.example.com",
            "mcp.example.com:443",
        ],
    )
    def test_every_form_a_client_or_proxy_sends_is_accepted(
        self, host: str, monkeypatch, tmp_path
    ) -> None:
        """The SDK matches `name:*` only when a port follows, so a bare name has
        to be listed itself. A proxy on a scheme's default port sends no port;
        some clients send the default port explicitly."""
        server = _networked(monkeypatch, tmp_path, resource="https://mcp.example.com")

        status = _post_message(server, host, Authorization=f"Bearer {GOOD_TOKEN}")

        assert status not in (421, 403)

    @pytest.mark.parametrize("host", ["evil.example", "mcp.example.com:8443", "evil.example:443"])
    def test_other_hosts_and_ports_are_refused(self, host: str, monkeypatch, tmp_path) -> None:
        """A resource URL on the default port allows that port and no other."""
        server = _networked(monkeypatch, tmp_path, resource="https://mcp.example.com")

        status = _post_message(server, host, Authorization=f"Bearer {GOOD_TOKEN}")

        assert status == 421

    @pytest.mark.parametrize(
        "host", ["mcp.example.com", "mcp.example.com:8443", "mcp.example.com:9000"]
    )
    def test_a_non_default_port_allows_the_host_on_any_port(
        self, host: str, monkeypatch, tmp_path
    ) -> None:
        server = _networked(monkeypatch, tmp_path, resource="https://mcp.example.com:8443")

        status = _post_message(server, host, Authorization=f"Bearer {GOOD_TOKEN}")

        assert status not in (421, 403)

    def test_a_resource_url_with_a_bad_port_adds_nothing(self, monkeypatch) -> None:
        """`urlsplit` raises on reading such a port; the server still starts,
        answering to the loopback names only."""
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com:99999")

        security = mcp_server._transport_security("0.0.0.0")

        assert security is not None
        assert not any("example" in h for h in security.allowed_hosts)
        assert security.allowed_origins == []

    def test_streamable_http_checks_the_host_too(self, monkeypatch, tmp_path) -> None:
        server = _networked(monkeypatch, tmp_path, resource=None)

        with TestClient(server.streamable_http_app()) as client:
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                headers={"Host": "evil.example", "Accept": "application/json, text/event-stream"},
            )

        assert response.status_code == 421

    def test_a_loopback_bind_keeps_the_sdk_defaults(self) -> None:
        """The SDK already protects a loopback bind, and allows `[::1]` there."""
        server = mcp_server._build_server(host="127.0.0.1", port=8080, networked=True)

        security = server.settings.transport_security
        assert security is not None
        assert "[::1]:*" in security.allowed_hosts


def _post_sse_body(monkeypatch, tmp_path: Path, size: int) -> int:
    """POST *size* bytes to the SSE message endpoint, with only the server's cap
    in the way.

    From mcp 1.30 the SDK caps SSE bodies itself, at the same 4 MiB, and would
    answer 413 whether or not the server adds its own cap. Its limit is raised
    here, where the setting exists, so a 413 at 4 MiB can only come from the
    server's middleware. On 1.29 there is no such setting and no SDK cap.
    """
    server = _networked(monkeypatch, tmp_path, resource=None)
    if "max_request_body_size" in type(server.settings).model_fields:
        server.settings.max_request_body_size = 64 * 1024 * 1024
    client = TestClient(server.sse_app())

    response = client.post(
        "/messages/?session_id=" + "0" * 32,
        content=b"x" * size,
        headers={"Host": "localhost:8080", "Content-Type": "application/json"},
    )
    return response.status_code


class TestTheSseBodyCap:
    def test_an_oversized_body_gets_413(self, monkeypatch, tmp_path) -> None:
        from mcp.server.streamable_http_manager import DEFAULT_MAX_REQUEST_BODY_SIZE

        assert _post_sse_body(monkeypatch, tmp_path, DEFAULT_MAX_REQUEST_BODY_SIZE + 1) == 413

    def test_a_body_at_the_cap_gets_through(self, monkeypatch, tmp_path) -> None:
        """It reaches the transport, which finds no such session. The cap is
        4 MiB exactly, not something smaller."""
        from mcp.server.streamable_http_manager import DEFAULT_MAX_REQUEST_BODY_SIZE

        assert _post_sse_body(monkeypatch, tmp_path, DEFAULT_MAX_REQUEST_BODY_SIZE) != 413

    def test_the_cap_is_the_streamable_http_one(self) -> None:
        from mcp.server.streamable_http_manager import DEFAULT_MAX_REQUEST_BODY_SIZE

        assert DEFAULT_MAX_REQUEST_BODY_SIZE == 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# 4. Input limits
# ---------------------------------------------------------------------------


def _tool(name: str, server: Any = None) -> Any:
    server = server or mcp_server._build_server()
    return server._tool_manager.get_tool(name).fn


#: Every bounded argument, with arguments that make a valid call otherwise.
BOUNDED_CALLS = [
    ("query", "question", {"agent_id": "sales"}),
    (
        "submit_feedback",
        "question",
        {"agent_id": "sales", "generated_sql": "SELECT 1", "rating": "up"},
    ),
    ("submit_feedback", "generated_sql", {"question": "q", "agent_id": "sales", "rating": "up"}),
    (
        "submit_feedback",
        "corrected_sql",
        {"question": "q", "agent_id": "sales", "generated_sql": "SELECT 1", "rating": "down"},
    ),
]


@contextmanager
def _spied_work() -> Iterator[dict[str, MagicMock]]:
    """Patch what the bounded tools do, to see whether they did it."""
    with ExitStack() as stack:
        run = stack.enter_context(
            patch("nlqueries.orchestrator.sync_runner.run_query", new_callable=AsyncMock)
        )
        run.return_value = MagicMock(
            answer="ok", sql=None, sql_result=None, citations=[], provenance=None
        )
        record = stack.enter_context(patch("nlqueries.feedback.store.record_feedback"))
        yield {"query": run, "submit_feedback": record}


async def _call(fn: Any, **kwargs: Any) -> Any:
    import inspect

    result = fn(**kwargs)
    return await result if inspect.isawaitable(result) else result


class TestInputLimits:
    @pytest.mark.anyio
    @pytest.mark.parametrize(("tool", "field", "others"), BOUNDED_CALLS)
    async def test_an_over_limit_argument_is_refused_and_nothing_runs(
        self, tool: str, field: str, others: dict[str, Any]
    ) -> None:
        limit = 4_000 if field == "question" else 8_000

        with _spied_work() as work:
            out = await _call(_tool(tool), **others, **{field: "x" * (limit + 1)})

        assert out.startswith("Refused:")
        assert field in out
        work[tool].assert_not_called()

    @pytest.mark.anyio
    @pytest.mark.parametrize(("tool", "field", "others"), BOUNDED_CALLS)
    async def test_an_argument_at_the_limit_is_accepted(
        self, tool: str, field: str, others: dict[str, Any]
    ) -> None:
        limit = 4_000 if field == "question" else 8_000

        with _spied_work() as work:
            out = await _call(_tool(tool), **others, **{field: "x" * limit})

        assert not out.startswith("Refused:")
        work[tool].assert_called_once()

    @pytest.mark.anyio
    async def test_the_limits_are_configurable(self, monkeypatch) -> None:
        monkeypatch.setenv("NLQ_MCP_MAX_QUESTION_CHARS", "10")
        monkeypatch.setenv("NLQ_MCP_MAX_SQL_CHARS", "20")
        server = mcp_server._build_server()

        with _spied_work():
            question = await _call(_tool("query", server), question="x" * 11, agent_id="sales")
            sql = _tool("submit_feedback", server)(
                question="q", agent_id="sales", generated_sql="x" * 21, rating="up"
            )

        assert question.startswith("Refused:")
        assert sql.startswith("Refused:")

    @pytest.mark.parametrize("value", ["0", "-5", "lots"])
    def test_a_bad_setting_falls_back_to_the_default(self, value: str, monkeypatch, caplog) -> None:
        monkeypatch.setenv("NLQ_MCP_MAX_QUESTION_CHARS", value)

        with caplog.at_level(logging.WARNING):
            limits = mcp_server._input_limits()

        assert limits["question"] == 4_000
        assert "NLQ_MCP_MAX_QUESTION_CHARS" in caplog.text

    def test_every_tool_with_a_bounded_argument_is_covered(self) -> None:
        """A tool added later with a `question` or SQL argument is bounded by
        registration, not by remembering; this lists which tools that is today."""
        bounded = {
            tool.name
            for tool in mcp_server.mcp._tool_manager.list_tools()
            if set((tool.parameters or {}).get("properties", {})) & set(mcp_server._input_limits())
        }

        assert bounded == {tool for tool, _, _ in BOUNDED_CALLS}

    def test_the_published_parameters_are_unchanged(self) -> None:
        published = {
            t.name: set((t.parameters or {}).get("properties", {}))
            for t in mcp_server.mcp._tool_manager.list_tools()
        }

        assert published["submit_feedback"] == {
            "question",
            "agent_id",
            "generated_sql",
            "rating",
            "corrected_sql",
        }
        assert published["health"] == {"probe_llm"}


# ---------------------------------------------------------------------------
# 5. What `health` and "not found" say about agents
# ---------------------------------------------------------------------------


@contextmanager
def _healthy(agents: list[str]) -> Iterator[None]:
    with (
        patch("urllib.request.urlopen", return_value=MagicMock()),
        patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=None),
        patch.object(mcp_server, "list_agents", return_value=agents),
    ):
        yield


def _served_health(tmp_path: Path, actions: list[str]) -> Any:
    from nlqueries.auth.authorizer import ConfigAllowlistAuthorizer, load_grants

    authorizer = ConfigAllowlistAuthorizer(load_grants(_grants(tmp_path, actions)))
    return mcp_server._health_tool(
        authorizer, lambda: Principal(subject="alice", source=Source.STATIC_TOKEN)
    )


class TestHealthNamesAgentsOnlyToThoseWhoMayListThem:
    def test_without_agents_list_it_gives_the_count(self, tmp_path) -> None:
        health = _served_health(tmp_path, ["health:read"])

        with _healthy(["sales", "support"]):
            out = health()

        assert "2 agent(s)" in out
        assert "sales" not in out
        assert "support" not in out

    def test_with_agents_list_it_names_them(self, tmp_path) -> None:
        health = _served_health(tmp_path, ["health:read", "agents:list"])

        with _healthy(["sales", "support"]):
            out = health()

        assert "sales, support" in out

    def test_the_stdio_owner_sees_them(self) -> None:
        with _healthy(["sales"]):
            out = mcp_server.mcp._tool_manager.get_tool("health").fn()

        assert "sales" in out

    def test_the_registered_tool_is_the_gated_one(self, monkeypatch, tmp_path) -> None:
        """Through `_build_server`, as a networked caller with only `health:read`."""
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com")
        monkeypatch.setenv(
            "NLQ_MCP_GRANTS_FILE", str(_grants(tmp_path, ["health:read"], subject="anonymous"))
        )
        server = mcp_server._build_server(authenticated=True, networked=True)

        with _healthy(["sales"]):
            out = server._tool_manager.get_tool("health").fn()

        assert "1 agent(s)" in out
        assert "sales" not in out


class TestNotFoundNamesNoOtherAgent:
    def test_get_agent_schema(self, tmp_path: Path) -> None:
        (tmp_path / "secret_agent.yaml").write_text("schema: {}\n", encoding="utf-8")

        with patch.object(mcp_server.config, "KB_PATH", tmp_path):
            out = mcp_server.get_agent_schema("ghost")

        assert "ghost" in out
        assert "not found" in out
        assert "secret_agent" not in out

    @pytest.mark.anyio
    async def test_query(self, tmp_path: Path) -> None:
        (tmp_path / "secret_agent.yaml").write_text("schema: {}\n", encoding="utf-8")

        with (
            patch(
                "nlqueries.orchestrator.sync_runner.run_query",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError("kb not found"),
            ),
            patch.object(mcp_server.config, "KB_PATH", tmp_path),
        ):
            out = await mcp_server.query("How many?", "ghost")

        assert "ghost" in out
        assert "not found" in out
        assert "secret_agent" not in out
