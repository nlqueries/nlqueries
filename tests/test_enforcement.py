"""
That the decision is actually in front of the tools (SEC-05, step 3).

An authorizer nothing consults is a document. These tests cover the wiring: that
every tool is wrapped, that a refusal stops the call rather than merely logging
it, that the agent the call is about is the agent that gets authorised, and that
wrapping did not quietly change what the tools claim to accept.
"""

from __future__ import annotations

import logging

import pytest
from nlqueries.auth.authorizer import ConfigAllowlistAuthorizer, DenyAll, Grant, LocalAuthorizer
from nlqueries.auth.enforcement import NotAuthorized, guard
from nlqueries.auth.principal import Action, AuditEvent, Principal, Source, local_principal

ALICE = Principal(subject="alice", source=Source.OIDC)

calls: list[tuple] = []


def _allowlist(agents: list[str], actions: list[str]) -> ConfigAllowlistAuthorizer:
    return ConfigAllowlistAuthorizer(
        [Grant(subject="alice", agents=frozenset(agents), actions=frozenset(actions))]
    )


def invalidate_cache(agent_id: str) -> str:
    """Stands in for the real tool: same name, so the same action applies."""
    calls.append(("invalidate_cache", agent_id))
    return f"cleared {agent_id}"


async def query(question: str, agent_id: str, dialect: str = "postgres") -> str:
    calls.append(("query", agent_id))
    return "an answer"


def list_agents() -> list[str]:
    calls.append(("list_agents", None))
    return ["sales"]


@pytest.fixture(autouse=True)
def _clear_calls():
    calls.clear()
    yield
    calls.clear()


class TestRefusalStopsTheCall:
    def test_a_denied_call_does_not_reach_the_tool(self) -> None:
        """The point. Logging a denial and running the tool anyway is the shape
        of failure this whole stream keeps finding."""
        guarded = guard(invalidate_cache, DenyAll(), lambda: ALICE)

        with pytest.raises(NotAuthorized):
            guarded("sales")

        assert calls == []

    def test_an_allowed_call_reaches_the_tool(self) -> None:
        guarded = guard(invalidate_cache, _allowlist(["sales"], ["*"]), lambda: ALICE)

        assert guarded("sales") == "cleared sales"
        assert calls == [("invalidate_cache", "sales")]

    def test_the_refusal_does_not_say_why(self) -> None:
        """A caller learning which agents exist from the wording of a refusal is
        the disclosure this is meant to prevent."""
        guarded = guard(invalidate_cache, _allowlist(["sales"], ["*"]), lambda: ALICE)

        with pytest.raises(NotAuthorized) as raised:
            guarded("payroll")

        assert "no grant" not in str(raised.value)


class TestTheAgentBeingAuthorised:
    def test_a_positional_argument_is_understood(self) -> None:
        guarded = guard(invalidate_cache, _allowlist(["sales"], ["*"]), lambda: ALICE)

        with pytest.raises(NotAuthorized):
            guarded("payroll")

    def test_a_keyword_argument_is_understood(self) -> None:
        """Same call, spelled the other way. Reading kwargs alone would authorise
        the positional form against no agent at all."""
        guarded = guard(invalidate_cache, _allowlist(["sales"], ["*"]), lambda: ALICE)

        with pytest.raises(NotAuthorized):
            guarded(agent_id="payroll")

    def test_an_agentless_tool_is_authorised_without_one(self) -> None:
        guarded = guard(list_agents, _allowlist(["sales"], ["agents:list"]), lambda: ALICE)

        assert guarded() == ["sales"]


class TestAsyncTools:
    @pytest.mark.anyio
    async def test_an_async_tool_is_guarded(self) -> None:
        """`query` is the async one, and the one that runs SQL."""
        guarded = guard(query, DenyAll(), lambda: ALICE)

        with pytest.raises(NotAuthorized):
            await guarded("how many orders?", "sales")

        assert calls == []

    @pytest.mark.anyio
    async def test_an_allowed_async_tool_runs(self) -> None:
        guarded = guard(query, _allowlist(["sales"], ["query:execute"]), lambda: ALICE)

        assert await guarded("how many orders?", "sales") == "an answer"


class TestTheAuditTrail:
    def test_a_denial_is_recorded_with_the_agent(self, caplog) -> None:
        guarded = guard(invalidate_cache, DenyAll(), lambda: ALICE)

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"), pytest.raises(NotAuthorized):
            guarded("payroll")

        assert caplog.records[0].audit["decision"] == "deny"
        assert caplog.records[0].audit["agent_id"] == "payroll"
        assert caplog.records[0].audit["action"] == str(Action.CACHE_INVALIDATE)

    def test_an_allow_is_recorded_too(self, caplog) -> None:
        guarded = guard(invalidate_cache, _allowlist(["sales"], ["*"]), lambda: ALICE)

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            guarded("sales")

        assert caplog.records[0].audit["decision"] == AuditEvent.ALLOW

    def test_a_local_call_is_recorded(self, caplog) -> None:
        """The local profile goes through the same path, so it leaves the same
        evidence."""
        guarded = guard(invalidate_cache, LocalAuthorizer(), lambda: local_principal("alice"))

        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            guarded("sales")

        assert caplog.records[0].audit["source"] == "local"


class TestTheToolsAreStillThemselves:
    def test_wrapping_preserves_the_published_parameters(self) -> None:
        """FastMCP derives each tool's schema by inspecting the function. A
        wrapper taking *args would publish nine tools that accept anything, and
        nothing would fail: they would just describe themselves wrongly.
        """
        import warnings

        warnings.filterwarnings("ignore")
        from nlqueries.mcp_server.server import mcp

        published = {
            t.name: set((t.parameters or {}).get("properties", {}))
            for t in mcp._tool_manager.list_tools()
        }

        assert published["query"] >= {"question", "agent_id"}
        assert published["invalidate_cache"] == {"agent_id"}
        assert published["list_connectors"] == set()

    def test_every_registered_tool_went_through_the_guard(self) -> None:
        """A tool registered without it is a tool with no authorisation.

        Counting the tools would pass with the guard removed, so this looks for
        the __wrapped__ attribute functools.wraps leaves behind — the evidence
        that each registered callable is a wrapper around the original.
        """
        import warnings

        warnings.filterwarnings("ignore")
        from nlqueries.mcp_server.server import _ALL_TOOLS, mcp

        registered = mcp._tool_manager.list_tools()

        assert len(registered) == len(_ALL_TOOLS)
        for tool in registered:
            assert hasattr(tool.fn, "__wrapped__"), f"{tool.name} is not guarded"
