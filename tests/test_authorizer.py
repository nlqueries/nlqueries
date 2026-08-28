"""
Deciding what an authenticated caller may do (SEC-05, step 3).

Authentication established who is calling. This is the other half: which agents,
and which actions on them. The default is refusal, so a deployment that
configures authentication and forgets the grants gets a server where nothing
works rather than one where everything does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nlqueries.auth.authorizer import (
    ConfigAllowlistAuthorizer,
    DenyAll,
    Grant,
    GrantsConfigError,
    LocalAuthorizer,
    load_grants,
)
from nlqueries.auth.principal import Action, Principal, Source, local_principal

ALICE = Principal(subject="alice", source=Source.OIDC)
OPERATOR = Principal(subject="operator", source=Source.STATIC_TOKEN)


def _grant(subject: str, agents: list[str], actions: list[str]) -> Grant:
    return Grant(subject=subject, agents=frozenset(agents), actions=frozenset(actions))


class TestTheDefault:
    def test_nothing_configured_refuses_everything(self) -> None:
        decision = DenyAll().authorize(ALICE, Action.QUERY_EXECUTE, "sales")

        assert not decision.allowed
        assert decision.reason

    def test_it_refuses_the_harmless_actions_too(self) -> None:
        """A default that allowed the read-only ones would be a judgement about
        which disclosures are acceptable, and that judgement is the operator's."""
        assert not DenyAll().authorize(ALICE, Action.AGENTS_LIST, None).allowed
        assert not DenyAll().authorize(ALICE, Action.HEALTH_READ, None).allowed


class TestLocal:
    def test_the_process_owner_may_act(self) -> None:
        decision = LocalAuthorizer().authorize(
            local_principal("alice"), Action.QUERY_EXECUTE, "sales"
        )

        assert decision.allowed

    def test_a_remote_principal_is_refused(self) -> None:
        """The local authorizer is for the local profile. If it ever reached a
        networked transport, it must not act as an allow-all."""
        decision = LocalAuthorizer().authorize(ALICE, Action.QUERY_EXECUTE, "sales")

        assert not decision.allowed


class TestAllowlist:
    def test_a_granted_action_on_a_granted_agent_is_allowed(self) -> None:
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["sales"], ["query:execute"])])

        assert authorizer.authorize(ALICE, Action.QUERY_EXECUTE, "sales").allowed

    def test_the_same_action_on_another_agent_is_refused(self) -> None:
        """Decision A. This is the whole reason actions are agent-scoped."""
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["sales"], ["query:execute"])])

        assert not authorizer.authorize(ALICE, Action.QUERY_EXECUTE, "payroll").allowed

    def test_another_action_on_the_granted_agent_is_refused(self) -> None:
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["sales"], ["query:execute"])])

        assert not authorizer.authorize(ALICE, Action.CACHE_INVALIDATE, "sales").allowed

    def test_a_subject_with_no_grant_is_refused(self) -> None:
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["sales"], ["*"])])

        assert not authorizer.authorize(OPERATOR, Action.AGENTS_LIST, None).allowed

    def test_a_wildcard_agent_covers_any_agent(self) -> None:
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["*"], ["query:execute"])])

        assert authorizer.authorize(ALICE, Action.QUERY_EXECUTE, "anything").allowed

    def test_a_wildcard_action_covers_any_action(self) -> None:
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["sales"], ["*"])])

        assert authorizer.authorize(ALICE, Action.CACHE_INVALIDATE, "sales").allowed

    def test_an_agentless_action_needs_no_agent_grant(self) -> None:
        """`health` and `agents:list` name no agent, so an agent grant cannot be
        what qualifies them."""
        authorizer = ConfigAllowlistAuthorizer([_grant("alice", ["sales"], ["health:read"])])

        assert authorizer.authorize(ALICE, Action.HEALTH_READ, None).allowed

    def test_several_grants_for_one_subject_are_combined(self) -> None:
        authorizer = ConfigAllowlistAuthorizer(
            [
                _grant("alice", ["sales"], ["query:execute"]),
                _grant("alice", ["payroll"], ["schema:read"]),
            ]
        )

        assert authorizer.authorize(ALICE, Action.QUERY_EXECUTE, "sales").allowed
        assert authorizer.authorize(ALICE, Action.SCHEMA_READ, "payroll").allowed
        assert not authorizer.authorize(ALICE, Action.QUERY_EXECUTE, "payroll").allowed


class TestLoadingTheFile:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "grants.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_well_formed_file_loads(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "grants:\n"
            "  - subject: alice\n"
            "    agents: ['sales']\n"
            "    actions: ['query:execute', 'schema:read']\n",
        )

        grants = load_grants(path)

        assert len(grants) == 1
        assert grants[0].subject == "alice"

    def test_a_misspelled_action_is_refused(self, tmp_path: Path) -> None:
        """It would grant nothing, and the operator who wrote it would have no
        way to tell that from a grant that works."""
        path = self._write(
            tmp_path,
            "grants:\n  - subject: alice\n    agents: ['sales']\n    actions: ['query:exec']\n",
        )

        with pytest.raises(GrantsConfigError, match="do not exist"):
            load_grants(path)

    def test_a_grant_with_no_agents_is_refused(self, tmp_path: Path) -> None:
        """Granting everything should be something someone wrote, not something
        an empty list happened to mean."""
        path = self._write(
            tmp_path, "grants:\n  - subject: alice\n    actions: ['query:execute']\n"
        )

        with pytest.raises(GrantsConfigError, match="no agents"):
            load_grants(path)

    def test_a_grant_with_no_subject_is_refused(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path, "grants:\n  - agents: ['sales']\n    actions: ['query:execute']\n"
        )

        with pytest.raises(GrantsConfigError, match="no subject"):
            load_grants(path)

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "grants: []\n")

        with pytest.raises(GrantsConfigError, match="no grants"):
            load_grants(path)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(GrantsConfigError, match="Cannot read"):
            load_grants(tmp_path / "absent.yaml")

    def test_invalid_yaml_is_refused(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "grants: [unclosed\n")

        with pytest.raises(GrantsConfigError, match="valid YAML"):
            load_grants(path)

    def test_every_real_action_is_accepted(self, tmp_path: Path) -> None:
        """The validation must not reject the vocabulary it validates against."""
        listed = "\n".join(f"      - '{action}'" for action in Action)
        path = self._write(
            tmp_path, f"grants:\n  - subject: alice\n    agents: ['*']\n    actions:\n{listed}\n"
        )

        assert load_grants(path)[0].actions == {str(a) for a in Action}

    def test_a_scalar_where_a_list_belongs_is_refused(self, tmp_path: Path) -> None:
        """A YAML scalar is iterable. `agents: "prod-*"` read as a sequence
        yields its characters, one of which is `*`, so the grant it produced
        covered every agent — the opposite of what was written, reached in
        silence. `agents: "*"` happens to behave correctly, which makes the
        scalar form look workable.
        """
        path = self._write(
            tmp_path,
            "grants:\n  - subject: alice\n    agents: 'prod-*'\n    actions: ['query:execute']\n",
        )

        with pytest.raises(GrantsConfigError, match="must be a list"):
            load_grants(path)

    def test_a_scalar_action_is_refused_too(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "grants:\n  - subject: alice\n    agents: ['sales']\n    actions: 'query:execute'\n",
        )

        with pytest.raises(GrantsConfigError, match="must be a list"):
            load_grants(path)

    def test_a_document_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        """`raw.get` on a list raised AttributeError rather than saying what was
        wrong with the file."""
        path = self._write(tmp_path, "- subject: alice\n")

        with pytest.raises(GrantsConfigError, match="must be a mapping"):
            load_grants(path)
