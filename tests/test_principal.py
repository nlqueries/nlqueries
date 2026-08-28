"""
The vocabulary authorisation will be expressed in (SEC-05, step 1).

Nothing here enforces anything yet. What these tests hold is the two properties
the later steps depend on: that every tool reachable over the transport has an
action someone can write a rule about, and that an audit record never carries
the things that must not reach a log.
"""

from __future__ import annotations

import logging

import pytest
from nlqueries.auth.principal import (
    AGENTLESS_ACTIONS,
    TOOL_ACTIONS,
    Action,
    AuditEvent,
    Principal,
    Source,
    local_principal,
    record,
)


class TestTheVocabularyCoversTheSurface:
    def test_every_registered_tool_has_an_action(self) -> None:
        """A tool added without one would be a tool nobody wrote a rule for.

        This reads the real registry rather than a copy, so adding a tool and
        forgetting the rule fails here rather than shipping.
        """
        from nlqueries.mcp_server.server import _ALL_TOOLS

        registered = {fn.__name__ for fn in _ALL_TOOLS}

        assert registered == set(TOOL_ACTIONS)

    def test_no_action_is_defined_for_a_tool_that_does_not_exist(self) -> None:
        """The other direction: a stale entry suggests a permission that
        authorises nothing."""
        from nlqueries.mcp_server.server import _ALL_TOOLS

        assert set(TOOL_ACTIONS) <= {fn.__name__ for fn in _ALL_TOOLS}

    def test_the_actions_that_run_or_read_customer_data_are_agent_scoped(self) -> None:
        """Decision A. An authorizer that could only answer "may this subject
        query" would be useless to anyone with one agent over sales and another
        over payroll."""
        for action in (
            Action.QUERY_EXECUTE,
            Action.SCHEMA_READ,
            Action.HISTORY_READ,
            Action.CACHE_INVALIDATE,
            Action.FEEDBACK_SUBMIT,
        ):
            assert action not in AGENTLESS_ACTIONS

    def test_the_agentless_actions_name_no_agent(self) -> None:
        assert set(Action) >= AGENTLESS_ACTIONS


class TestPrincipal:
    def test_a_principal_must_name_someone(self) -> None:
        """The empty-subject case #157 closed in the verifier, held here too so
        it cannot re-enter by another route."""
        with pytest.raises(ValueError, match="subject"):
            Principal(subject="", source=Source.OIDC)

    def test_a_blank_subject_is_not_a_subject(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            Principal(subject="   ", source=Source.OIDC)

    def test_the_local_principal_is_the_process_owner(self) -> None:
        principal = local_principal("alice")

        assert principal.is_local
        assert principal.subject == "alice"

    def test_claims_are_not_in_the_representation(self) -> None:
        """`repr` reaches logs and tracebacks. Claims carry names, addresses and
        group membership."""
        principal = Principal(
            subject="user-1",
            source=Source.OIDC,
            claims={"email": "alice@example.com", "groups": ["finance"]},
        )

        assert "alice@example.com" not in repr(principal)
        assert "finance" not in repr(principal)


class TestAuditRecords:
    def test_a_record_carries_the_decision_and_its_subject(self) -> None:
        event = AuditEvent(
            principal="user-1",
            source=Source.OIDC,
            action=Action.QUERY_EXECUTE,
            agent_id="sales",
            decision=AuditEvent.DENY,
            reason="not granted",
        )

        payload = event.as_dict()

        assert payload["decision"] == "deny"
        assert payload["action"] == "query:execute"
        assert payload["agent_id"] == "sales"

    def test_a_record_carries_no_claims(self) -> None:
        """It travels into support bundles."""
        event = AuditEvent(
            principal="user-1",
            source=Source.OIDC,
            action=Action.QUERY_EXECUTE,
            decision=AuditEvent.ALLOW,
        )

        assert "claims" not in event.as_dict()

    def test_both_outcomes_are_recorded(self, caplog) -> None:
        """A denial nobody wrote down is a control nobody can evidence; an allow
        nobody wrote down leaves an incident unable to say what was reached."""
        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            record(
                AuditEvent(
                    principal="u",
                    source=Source.LOCAL,
                    action=Action.HEALTH_READ,
                    decision=AuditEvent.ALLOW,
                )
            )
            record(
                AuditEvent(
                    principal="u",
                    source=Source.LOCAL,
                    action=Action.QUERY_EXECUTE,
                    agent_id="payroll",
                    decision=AuditEvent.DENY,
                )
            )

        levels = [r.levelno for r in caplog.records]

        assert logging.INFO in levels
        assert logging.WARNING in levels

    def test_a_denial_is_a_warning_so_it_can_be_routed(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="nlqueries.audit"):
            record(
                AuditEvent(
                    principal="u",
                    source=Source.OIDC,
                    action=Action.CACHE_INVALIDATE,
                    agent_id="sales",
                    decision=AuditEvent.DENY,
                )
            )

        assert caplog.records[0].levelno == logging.WARNING
        assert caplog.records[0].audit["agent_id"] == "sales"
