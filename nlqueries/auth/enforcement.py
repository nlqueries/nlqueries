"""
nlqueries.auth.enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~
Putting the authorisation decision in front of every tool.

Step 3 of SEC-05. Tools are registered in one place -- ``add_tool`` in
``mcp_server/server.py`` -- so there is a single point every call passes
through, and the guard goes there rather than at nine call sites where the tenth
would be forgotten.

The wrapper preserves the wrapped function's signature. FastMCP derives each
tool's schema by inspecting it, so a wrapper that took ``*args, **kwargs`` would
publish nine tools that all claim to accept anything. That is asserted in the
tests, because it would not fail loudly: the tools would still be callable, and
would simply describe themselves wrongly to every client.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from nlqueries.auth.admission import AdmissionControl, TooManyRequests
from nlqueries.auth.authorizer import AgentAuthorizer
from nlqueries.auth.principal import (
    AGENTLESS_ACTIONS,
    TOOL_ACTIONS,
    Action,
    AuditEvent,
    Principal,
    record,
)

logger = logging.getLogger(__name__)


class NotAuthorized(Exception):
    """Raised when a principal may not perform an action.

    The message names the action and the agent, and never the reason the grant
    failed: a caller learning which agents exist by the wording of a refusal is
    the disclosure this control is meant to prevent.
    """


def _agent_of(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    """The agent this call is about, or None.

    Bound through the signature rather than read from kwargs, so a positional
    call is understood the same way as a keyword one.
    """
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except TypeError:
        return None
    value = bound.arguments.get("agent_id")
    return str(value) if value is not None else None


def _decide(
    authorizer: AgentAuthorizer,
    principal: Principal,
    action: Action,
    agent_id: str | None,
) -> None:
    """Authorise, record, and raise if refused."""
    if action in AGENTLESS_ACTIONS:
        agent_id = None
    elif agent_id is None:
        # An agent-scoped action whose agent could not be determined. Passing
        # None on would authorise it against no agent, and a grant covering any
        # one agent would then cover this call -- a subject granted
        # query:execute on sales alone would be let through. Refuse instead: an
        # agent we cannot name is not an agent we can check.
        record(
            AuditEvent(
                principal=principal.subject,
                source=principal.source,
                action=action,
                agent_id=None,
                decision=AuditEvent.DENY,
                reason="the agent for this call could not be determined",
            )
        )
        raise NotAuthorized(f"Not authorized to perform {action}.")

    decision = authorizer.authorize(principal, action, agent_id)
    record(
        AuditEvent(
            principal=principal.subject,
            source=principal.source,
            action=action,
            agent_id=agent_id,
            decision=AuditEvent.ALLOW if decision.allowed else AuditEvent.DENY,
            reason=decision.reason,
        )
    )
    if not decision.allowed:
        target = f" on {agent_id!r}" if agent_id else ""
        raise NotAuthorized(f"Not authorized to perform {action}{target}.")


def _admit(admission: AdmissionControl | None, principal: Principal, action: Action) -> None:
    """Take an admission slot, recording a refusal as its own kind of denial."""
    if admission is None:
        return
    try:
        admission.acquire(principal.subject)
    except TooManyRequests as exc:
        record(
            AuditEvent(
                principal=principal.subject,
                source=principal.source,
                action=action,
                decision=AuditEvent.DENY,
                reason=f"admission: {exc}",
            )
        )
        raise


def guard(
    func: Callable[..., Any],
    authorizer: AgentAuthorizer,
    principal_for_call: Callable[[], Principal],
    admission: AdmissionControl | None = None,
) -> Callable[..., Any]:
    """Wrap *func* so the call is authorised, and admitted, first.

    *principal_for_call* is resolved per call rather than passed in, because the
    identity belongs to the request and the wrapper is built once at startup.

    Authorisation runs before admission. A caller who may not do this at all
    should be told so however many times they ask, rather than having their
    refusals rationed and eventually replaced by a different refusal.
    """
    action = TOOL_ACTIONS.get(func.__name__)
    if action is None:  # pragma: no cover - the registry test prevents this
        raise KeyError(
            f"{func.__name__} has no action in TOOL_ACTIONS, so nobody has decided who may call it."
        )

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_guarded(*args: Any, **kwargs: Any) -> Any:
            principal = principal_for_call()
            _decide(authorizer, principal, action, _agent_of(func, args, kwargs))
            _admit(admission, principal, action)
            try:
                return await func(*args, **kwargs)
            finally:
                # In `finally`, so a tool that raises does not leak the slot. A
                # leaked slot is permanent: the caller's concurrency allowance
                # shrinks by one for the life of the process.
                if admission is not None:
                    admission.release(principal.subject)

        return async_guarded

    @functools.wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        principal = principal_for_call()
        _decide(authorizer, principal, action, _agent_of(func, args, kwargs))
        _admit(admission, principal, action)
        try:
            return func(*args, **kwargs)
        finally:
            if admission is not None:
                admission.release(principal.subject)

    return guarded
