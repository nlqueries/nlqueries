"""
nlqueries.auth.authorizer
~~~~~~~~~~~~~~~~~~~~~~~~~
Deciding whether a principal may perform an action on an agent.

Step 3 of SEC-05, decisions A and D. Authentication established who is calling;
this is what they may do. The two are kept apart deliberately: a permission held
on the identity would drift from the decision that is actually enforced, and the
audit record would then describe the wrong one.

The default is :class:`DenyAll`. A deployment that configures authentication and
forgets the grants gets a server where nothing works, which is recoverable, and
not one where everything does, which is the finding.

Enterprise supplies its own implementation over its tenant tables. Core supplies
the interface, a file-backed allowlist, and the local profile — it does not
reach into enterprise's RBAC, because core's OSS profile has none of those
tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from nlqueries.auth.principal import Action, Principal

logger = logging.getLogger(__name__)

#: Grants every agent, or every action. Spelled out rather than implied by an
#: empty list, which reads as "none" at least as naturally as "all".
WILDCARD = "*"


@dataclass(frozen=True)
class Decision:
    """Whether the call proceeds, and why.

    The reason is carried even when allowed: an audit record that says only
    "allowed" cannot answer which grant permitted it.
    """

    allowed: bool
    reason: str = ""


@runtime_checkable
class AgentAuthorizer(Protocol):
    """What core asks before letting a call through.

    Deliberately small. Enterprise implements it over its tenant tables and RLS
    session; core implements it over a file.
    """

    def authorize(self, principal: Principal, action: Action, agent_id: str | None) -> Decision: ...


class DenyAll:
    """Refuses everything. The default when nothing is configured."""

    def authorize(self, principal: Principal, action: Action, agent_id: str | None) -> Decision:
        return Decision(False, "no authorizer is configured")


class LocalAuthorizer:
    """Allows everything, for a principal that owns the process.

    stdio has no token because the caller started this process and already has
    whatever it has — the knowledge base, the connector file, the credentials.
    Denying it anything would be theatre.

    It goes through the same interface as every other decision so that there is
    one enforcement path, and so an audit record exists for local calls too.
    """

    def authorize(self, principal: Principal, action: Action, agent_id: str | None) -> Decision:
        if not principal.is_local:
            return Decision(False, "not a local principal")
        return Decision(True, "local process owner")


@dataclass(frozen=True)
class Grant:
    """One line of the allowlist."""

    subject: str
    agents: frozenset[str]
    actions: frozenset[str]

    def covers(self, action: Action, agent_id: str | None) -> bool:
        if WILDCARD not in self.actions and str(action) not in self.actions:
            return False
        if agent_id is None:
            # An action that names no agent needs no agent grant.
            return True
        return WILDCARD in self.agents or agent_id in self.agents


class ConfigAllowlistAuthorizer:
    """Grants read from a file.

    The file is the whole of the policy: a subject with no grant is denied, and
    there is no deny list, because a policy with both grants and denials needs a
    precedence rule and precedence rules are where authorisation bugs live.
    """

    def __init__(self, grants: list[Grant]) -> None:
        self._by_subject: dict[str, list[Grant]] = {}
        for grant in grants:
            self._by_subject.setdefault(grant.subject, []).append(grant)

    def authorize(self, principal: Principal, action: Action, agent_id: str | None) -> Decision:
        applicable = self._by_subject.get(principal.subject, [])
        if not applicable:
            return Decision(False, "no grant for this subject")
        for grant in applicable:
            if grant.covers(action, agent_id):
                return Decision(True, "granted by allowlist")
        return Decision(False, "no grant covers this action on this agent")

    @property
    def subjects(self) -> frozenset[str]:
        return frozenset(self._by_subject)


class GrantsConfigError(RuntimeError):
    """Raised for a grants file that cannot be honoured."""


def load_grants(path: Path) -> list[Grant]:
    """Read an allowlist.

    An action that is not one of ours raises rather than being ignored. A
    misspelled action in a grants file grants nothing, and the operator who
    wrote it would have no way to tell that from a grant that is working.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise GrantsConfigError(f"Cannot read the grants file at {path}.") from exc
    except yaml.YAMLError as exc:
        raise GrantsConfigError(f"The grants file at {path} is not valid YAML.") from exc

    entries = raw.get("grants")
    if not isinstance(entries, list) or not entries:
        raise GrantsConfigError(
            f"The grants file at {path} defines no grants. A file that grants "
            "nothing and an absent file are the same server; if that is what you "
            "want, leave the file out and the transport will refuse to start."
        )

    known = {str(action) for action in Action}
    grants: list[Grant] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise GrantsConfigError(f"Grant {index} in {path} is not a mapping.")
        subject = str(entry.get("subject", "")).strip()
        if not subject:
            raise GrantsConfigError(f"Grant {index} in {path} names no subject.")

        actions = {str(a).strip() for a in entry.get("actions", []) or []}
        unknown = actions - known - {WILDCARD}
        if unknown:
            raise GrantsConfigError(
                f"Grant {index} in {path} names actions that do not exist: "
                f"{sorted(unknown)}. Valid actions: {sorted(known)}."
            )
        if not actions:
            raise GrantsConfigError(f"Grant {index} in {path} lists no actions.")

        agents = {str(a).strip() for a in entry.get("agents", []) or []}
        if not agents:
            raise GrantsConfigError(
                f"Grant {index} in {path} lists no agents. Use ['*'] to mean all "
                "of them, so that granting everything is something someone wrote."
            )

        grants.append(Grant(subject=subject, agents=frozenset(agents), actions=frozenset(actions)))

    return grants
