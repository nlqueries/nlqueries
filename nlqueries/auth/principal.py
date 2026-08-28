"""
nlqueries.auth.principal
~~~~~~~~~~~~~~~~~~~~~~~~
Who is calling, what they are asking to do, and which agent they are asking it
of.

The MCP transport reaches nine tools without credentials (SEC-05). One of them
runs SQL against a configured database; the SQL policy constrains what may be
run, never who may run it. Closing that needs three things before any of it can
be enforced: an identity, a vocabulary of actions to authorise, and a record of
what was decided.

This module is those three. It decides nothing on its own — the verifier that
produces a :class:`Principal` and the authorizer that consults one arrive with
the transport wiring.

Two properties are load-bearing and are asserted in the tests rather than left
to habit. Actions are per-agent, because agents are the unit customers separate
data along, and an authorizer that could only answer "may this subject query"
would be useless to anyone running one agent over sales and another over
payroll. And every tool maps to an action: a tool added later without one would
otherwise be a tool nobody wrote a rule for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger("nlqueries.audit")


class Action(StrEnum):
    """What a caller is asking to do.

    Named for the effect rather than the tool, so a tool renamed or split does
    not silently become a different permission.
    """

    #: Enumerate the agents that exist.
    AGENTS_LIST = "agents:list"
    #: Read an agent's table and column names.
    SCHEMA_READ = "schema:read"
    #: Generate SQL and run it against the agent's database.
    QUERY_EXECUTE = "query:execute"
    #: Record feedback against an agent.
    FEEDBACK_SUBMIT = "feedback:submit"
    #: Read the questions and statements an agent has served.
    HISTORY_READ = "history:read"
    #: Read connector names, hosts and databases.
    CONNECTORS_LIST = "connectors:list"
    #: Discard an agent's cached answers.
    CACHE_INVALIDATE = "cache:invalidate"
    #: Read cache hit rates and volumes.
    CACHE_STATS_READ = "cache:stats"
    #: Liveness.
    HEALTH_READ = "health:read"


#: The action each tool performs. Every entry in ``_ALL_TOOLS`` must appear
#: here; ``tests/test_principal.py`` fails the build otherwise, so a tool added
#: without a rule cannot reach the transport unnoticed.
TOOL_ACTIONS: dict[str, Action] = {
    "list_agents": Action.AGENTS_LIST,
    "get_agent_schema": Action.SCHEMA_READ,
    "query": Action.QUERY_EXECUTE,
    "submit_feedback": Action.FEEDBACK_SUBMIT,
    "get_query_history": Action.HISTORY_READ,
    "list_connectors": Action.CONNECTORS_LIST,
    "invalidate_cache": Action.CACHE_INVALIDATE,
    "get_cache_stats": Action.CACHE_STATS_READ,
    "health": Action.HEALTH_READ,
}

#: Actions that name no agent. Everything else is authorised against one, and an
#: authorizer asked about one of these receives ``agent_id=None``.
AGENTLESS_ACTIONS = frozenset({Action.AGENTS_LIST, Action.CONNECTORS_LIST, Action.HEALTH_READ})


class Source(StrEnum):
    """How a principal was established."""

    #: An OIDC ID token, verified against the provider's JWKS.
    OIDC = "oidc"
    #: A pre-shared token held in the deployment's secret store. For a
    #: single-operator install that has no identity provider.
    STATIC_TOKEN = "static-token"
    #: The process owner, over stdio. There is no token: the caller launched
    #: this process and already has whatever it has.
    LOCAL = "local"


@dataclass(frozen=True)
class Principal:
    """An authenticated caller.

    Carries no permissions of its own. What a principal may do is the
    authorizer's answer, not a property of the identity — otherwise the two
    would drift and the audit record would describe the wrong one.
    """

    subject: str
    source: Source
    scopes: frozenset[str] = field(default_factory=frozenset)
    #: Claims from the token, for an authorizer that maps groups to agents.
    #: Never logged: it can carry names, email addresses and group membership.
    claims: dict[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.subject or not self.subject.strip():
            raise ValueError("A principal must have a subject.")

    @property
    def is_local(self) -> bool:
        return self.source is Source.LOCAL


def local_principal(username: str) -> Principal:
    """The principal for a stdio session: whoever owns the process.

    The local profile satisfies the same interface as a remote one so that there
    is a single authorisation path rather than an authenticated one and an
    unauthenticated one that drift apart.
    """
    return Principal(subject=username, source=Source.LOCAL)


@dataclass(frozen=True)
class AuditEvent:
    """One authorisation decision.

    Both outcomes are recorded. A denial nobody wrote down is a control nobody
    can evidence, and an allow nobody wrote down leaves an incident with no way
    to establish what was reached.
    """

    principal: str
    source: Source
    action: Action
    decision: str
    agent_id: str | None = None
    reason: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    ALLOW = "allow"
    DENY = "deny"

    def as_dict(self) -> dict[str, object]:
        """The event as a flat record.

        Deliberately excludes the principal's claims and any token: this is
        written to logs that travel into support bundles.
        """
        return {
            "at": self.at.isoformat(),
            "principal": self.principal,
            "source": str(self.source),
            "action": str(self.action),
            "agent_id": self.agent_id,
            "decision": self.decision,
            "reason": self.reason,
        }


def record(event: AuditEvent) -> None:
    """Emit *event* to the audit logger.

    Denials are warnings and allows are informational, so a deployment can route
    the two differently without parsing the message.
    """
    payload = event.as_dict()
    if event.decision == AuditEvent.DENY:
        logger.warning("authorization denied", extra={"audit": payload})
    else:
        logger.info("authorization allowed", extra={"audit": payload})
