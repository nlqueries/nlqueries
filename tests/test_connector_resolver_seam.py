"""The query path must resolve a connector class the same way every other path does.

``CONNECTOR_REGISTRY`` holds one class per db-type, which is the whole answer
for the connectors this package ships and not the whole answer for a deployment
that adds an authentication method to one of them. A key-pair or token sign-in
differs from a password sign-in only in the driver handshake, so the natural
implementation is a subclass overriding ``connect()`` -- and a registry keyed by
db-type has nowhere to put a second class for the same type.

Before the seam, a deployment could select its own class everywhere it
controlled the call and still lose it on the query path, because
``open_connector_for_agent`` read the registry directly. The connector created
green, tested green, and then answered every question through a class that could
not perform its sign-in. That is the same shape as the TLS defect this area has
been fixing: a setting honoured on the paths that connect from a request body
and dropped on the one that connects from the projected file.

These tests are therefore about ``open_connector_for_agent`` -- the class that is
actually *connected* -- rather than about the helper's return value, which could
be right while the loader went on ignoring it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nlqueries.cli.main as cli_main
import pytest
import yaml
from nlqueries import config, connectors
from nlqueries.connectors import loader
from nlqueries.execution import ExecutionPolicy

_AGENT = "postgres:localhost:db"


class _Registered:
    """Stands in for the class the registry holds for a db-type."""

    def __init__(self) -> None:
        self.credentials: dict[str, Any] | None = None

    closed = False

    def connect(self, credentials: dict[str, Any]) -> None:
        self.credentials = credentials

    def close(self) -> None:
        # The cache disposes what it evicts, and nothing else closes a pooled
        # connector -- so this is how a leak is observed.
        self.closed = True

    def test_connection(self) -> bool:
        # `nlqueries connect` validates the connection it just opened, so a stub
        # without this fails the command for a reason unrelated to resolution.
        return True


class _Resolved(_Registered):
    """Stands in for the subclass a deployment resolves for an auth method."""


@pytest.fixture(autouse=True)
def _no_resolver_leaks():
    """A resolver is process-wide, so a test that installs one and does not
    remove it would silently re-route every connector opened afterwards -- in
    this file and in every other."""
    connectors.set_connector_resolver(None)
    loader.invalidate_connector_cache()
    yield
    connectors.set_connector_resolver(None)
    loader.invalidate_connector_cache()


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary connectors file, redirected in *both* places that name it.

    `nlqueries.cli.main` binds `CONNECTORS_FILE` at import, so patching
    `config.CONNECTORS_FILE` alone redirects every reader and none of the CLI's
    writes. That is not a cosmetic gap: `_save_connector` reads through
    `load_connectors_for_update`, which *is* dynamic and returns `{}` for the
    empty temporary path, then writes the result to the module-level name -- so a
    test driving `nlqueries connect` would replace the operator's real
    `~/.nlqueries/connectors.yaml` with its own single entry, destroying every
    connector registered on the machine running the suite.

    `tests/test_loader_reports_why.py` rebinds both for the same reason.
    """
    path = tmp_path / "connectors.yaml"
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    monkeypatch.setattr(cli_main, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)
    return path


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> type[_Registered]:
    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Registered)
    return _Registered


def _write(path: Path, **entry: Any) -> None:
    base = {"db_type": "postgres", "url": "postgresql://user:hunter2@localhost:5432/db"}
    base.update(entry)
    path.write_text(yaml.safe_dump({_AGENT: base}))


def _opened(agent: str = _AGENT):
    """The connector object the loader built, unwrapped from its policy wrapper.

    The loader returns a ``PermittedConnector``, so ``type(...)`` on the return
    value is the wrapper for every case and would make each assertion below pass
    or fail together regardless of what was resolved.
    """
    permitted = loader.open_connector_for_agent(agent, ExecutionPolicy.execute_read_only())
    return None if permitted is None else permitted._inner  # noqa: SLF001


# ---------------------------------------------------------------------------
# The default: nothing installed, nothing changed
# ---------------------------------------------------------------------------


def test_with_no_resolver_the_registry_class_is_built(connectors_file, registered) -> None:
    """The baseline every other test here is measured against.

    Without this, a seam that resolved nothing at all -- or one that returned the
    subclass unconditionally -- would satisfy the assertions below just as well.
    """
    _write(connectors_file)
    assert type(_opened()) is _Registered


def test_the_helper_and_the_loader_agree_by_default(registered) -> None:
    assert connectors.connector_class_for("postgres") is _Registered
    assert connectors.connector_class_for("postgres", {}) is _Registered


def test_an_unresolvable_db_type_is_still_none(connectors_file) -> None:
    """Unchanged behaviour, and the reason callers keep reporting it themselves:
    the seam returns ``None`` for a type nothing resolves, exactly as
    ``CONNECTOR_REGISTRY.get`` did."""
    assert connectors.connector_class_for("teradata", {"db_type": "teradata"}) is None


# ---------------------------------------------------------------------------
# An installed resolver
# ---------------------------------------------------------------------------


def test_the_resolved_class_is_the_one_connected(connectors_file, registered) -> None:
    """The property that matters. Not "the resolver was called" and not "the
    helper returned the subclass" -- the object the loader hands back, having
    called ``connect()`` on it, must be the resolved one."""
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)

    opened = _opened()
    assert type(opened) is _Resolved
    assert opened.credentials is not None, "the resolved class must have been connected"


def test_the_resolver_sees_the_entry_it_must_decide_from(connectors_file, registered) -> None:
    """The whole point of passing *cfg*: the deciding field is one the registry
    knows nothing about, so a seam that passed only the db-type would be unable
    to express the selection it exists for."""
    seen: list[tuple[str, dict[str, Any]]] = []

    def resolver(db_type: str, cfg: Any) -> type | None:
        seen.append((db_type, dict(cfg)))
        return _Resolved if cfg.get("auth_method") == "keypair" else None

    connectors.set_connector_resolver(resolver)
    _write(connectors_file, auth_method="keypair")

    assert type(_opened()) is _Resolved
    assert seen and seen[0][0] == "postgres"
    assert seen[0][1]["auth_method"] == "keypair"


def test_a_resolver_with_no_opinion_falls_through_to_the_registry(
    connectors_file, registered
) -> None:
    """``None`` means "no opinion", not "no connector". A resolver that answers
    for one auth method must leave every other connector in the deployment
    opening exactly as it did before."""
    connectors.set_connector_resolver(
        lambda db_type, cfg: _Resolved if cfg.get("auth_method") == "keypair" else None
    )
    _write(connectors_file, auth_method="password")

    assert type(_opened()) is _Registered


def test_removing_the_resolver_restores_the_registry(connectors_file, registered) -> None:
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)
    assert type(_opened()) is _Resolved

    connectors.set_connector_resolver(None)
    assert type(_opened()) is _Registered


# ---------------------------------------------------------------------------
# A resolver that misbehaves
# ---------------------------------------------------------------------------


def test_a_raising_resolver_does_not_take_the_query_path_down(
    connectors_file, registered, caplog
) -> None:
    """It is consulted for every connector, including the ones it has no opinion
    about. A fault in it must not stop a password-authenticated Postgres
    connector that would otherwise have opened -- that would turn an extension
    bug into a total outage."""

    def resolver(db_type: str, cfg: Any) -> type | None:
        raise RuntimeError("resolver is broken")

    connectors.set_connector_resolver(resolver)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=connectors.logger.name):
        assert type(_opened()) is _Registered
    assert "connector resolver raised" in caplog.text
    assert "resolver is broken" in caplog.text, "the resolver's own words are the diagnosis"


def test_a_working_resolver_logs_no_warning(connectors_file, registered, caplog) -> None:
    """Negative control for the test above, which would pass just as well
    against a seam that warned on every call -- and a warning on every connector
    open is one people learn to filter out, taking the real one with it."""
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=connectors.logger.name):
        assert type(_opened()) is _Resolved
    assert caplog.records == []


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


def test_a_changed_auth_method_is_re_resolved_through_the_cache(
    connectors_file, registered, monkeypatch
) -> None:
    """Resolution happens on a cache miss, so the cache is what decides whether a
    connector switched to key-pair authentication actually starts using it.

    The configuration entry is part of the cache fingerprint, so changing
    ``auth_method`` is a miss by itself and the rebuild goes through the
    resolver. This is the other half of the seam's documented "install it before
    the first connector is opened" caveat: settings that do not change are served
    from the cache and not re-resolved.
    """
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    connectors.set_connector_resolver(
        lambda db_type, cfg: _Resolved if cfg.get("auth_method") == "keypair" else None
    )

    _write(connectors_file, auth_method="password")
    assert type(_opened()) is _Registered

    _write(connectors_file, auth_method="keypair")
    assert type(_opened()) is _Resolved, "the entry changed, so the cached class must not be reused"


def test_removal_does_not_reach_a_cached_connector(
    connectors_file, registered, monkeypatch
) -> None:
    """The other direction of the cache caveat, and the reason it is documented.

    The fingerprint covers the entry and the password, not which resolver was
    installed. So a connector already built through a resolver keeps being served
    under that class after the resolver is removed -- removal is not a rebuild.
    The test above it passes only because its fixture disables the cache, which
    is exactly the kind of guard that reads like a proof and is not one.

    Asserted rather than assumed: this is what makes
    ``invalidate_connector_cache`` part of the documented procedure instead of an
    optimisation.
    """
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)
    assert type(_opened()) is _Resolved

    connectors.set_connector_resolver(None)
    assert type(_opened()) is _Resolved, (
        "removal alone must not be claimed to restore the registry: the cached "
        "connector is served under the resolved class until something invalidates it"
    )

    loader.invalidate_connector_cache()
    assert type(_opened()) is _Registered, (
        "invalidating the cache is what actually restores the registry class"
    )


# ---------------------------------------------------------------------------
# Registration, where there is no entry yet
# ---------------------------------------------------------------------------


def test_registration_resolves_from_an_entry_shaped_mapping(
    connectors_file, registered, monkeypatch
) -> None:
    """``nlqueries connect`` is the one site with no entry to hand the resolver.

    It builds one instead of passing its own option dict, so that a resolver
    never has to ask which caller it is serving. Nothing else in the suite
    invokes this command, so without this test an edit dropping ``db_type`` or
    ``url`` from that mapping -- or this site quietly reverting to
    ``CONNECTOR_REGISTRY.get`` -- would leave the suite green while
    reintroducing exactly the split this change exists to remove: the connection
    validated through one class and every later path using another.

    What is asserted is the class ``connect`` actually built, plus the shape the
    resolver was given, since a resolver that ignored *cfg* would pass on the
    class assertion alone.
    """
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    seen: list[dict[str, Any]] = []

    def resolver(db_type: str, cfg: Any) -> type | None:
        seen.append(dict(cfg))
        return _Resolved if cfg.get("db_type") == "postgres" else None

    connectors.set_connector_resolver(resolver)
    monkeypatch.setattr("nlqueries.cli.main._save_password", lambda *a, **k: False)

    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "postgres",
            "--host",
            "db.internal",
            "--database",
            "analytics",
            "--user",
            "alice",
            "--password",
            "hunter2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen, "the seam was not consulted at registration"
    _assert_registration_stayed_in(connectors_file)

    entry = seen[0]
    assert entry["db_type"] == "postgres", "a resolver keys on the db-type; it must be there"
    assert entry["url"].startswith("postgresql"), "an entry always carries its URL"
    assert "password" not in entry, (
        "an entry has no discrete password -- it lives inside the URL, and this "
        "site must not be the one place a resolver sees it separately"
    )
    assert "hunter2" in entry["url"], "the password is reachable where it always is"


def test_registration_falls_back_to_the_registry(connectors_file, registered, monkeypatch) -> None:
    """The control. A resolver with no opinion at registration must leave the
    command validating through the registry class, exactly as before the seam."""
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    built: list[type] = []

    class _Recording(_Registered):
        def __init__(self) -> None:
            super().__init__()
            built.append(type(self))

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Recording)
    monkeypatch.setattr("nlqueries.cli.main._save_password", lambda *a, **k: False)
    connectors.set_connector_resolver(lambda db_type, cfg: None)

    result = CliRunner().invoke(
        cli, ["connect", "postgres", "--database", "analytics", "--user", "a", "--password", "p"]
    )

    assert result.exit_code == 0, result.output
    assert built == [_Recording]
    _assert_registration_stayed_in(connectors_file)


def _assert_registration_stayed_in(path: Path) -> None:
    """`connect` writes, so a test driving it must be held to writing here.

    `nlqueries.cli.main` binds `CONNECTORS_FILE` at import and the fixture has to
    rebind both names. Miss one and the command reads the temporary file, finds
    it empty, and writes its single entry over the operator's real
    `~/.nlqueries/connectors.yaml` -- destroying every connector on the machine
    running the suite, while the test passes. Nothing else in this file writes at
    all, so this is the one place that failure can be caught.
    """
    written = yaml.safe_load(path.read_text()) or {}
    assert written, "the command wrote nothing here, so it wrote somewhere else"
    assert path == config.CONNECTORS_FILE
    assert path == cli_main.CONNECTORS_FILE, (
        "the CLI's own binding was not redirected; its write escaped the fixture"
    )


def test_a_transient_resolver_fault_is_not_cached(connectors_file, registered, monkeypatch) -> None:
    """A fallback taken because the resolver raised must not be pinned by the cache.

    The fingerprint covers the entry and the password, not how the class was
    chosen. So without this, a resolver that fails for one call -- one consulting
    a configuration service, say -- would hold the registry class for the whole
    TTL of every connector its fallback can still open, and the only trace would
    be a single warning. One momentary fault would reinstate exactly the split
    this seam removes.

    Uncached, the next query resolves again and corrects itself.
    """
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    calls: list[int] = []

    def resolver(db_type: str, cfg: Any) -> type | None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("configuration service is briefly unreachable")
        return _Resolved

    connectors.set_connector_resolver(resolver)
    _write(connectors_file)

    assert type(_opened()) is _Registered, "the fallback is the registry class"
    assert type(_opened()) is _Resolved, (
        "the second attempt must resolve again rather than be served the cached fallback"
    )


def test_a_resolver_that_declines_is_still_cached(connectors_file, registered, monkeypatch) -> None:
    """The control. Declining is an answer -- "use the registry" -- and is as
    cacheable as any other, so the exception path must be what skips the cache
    and not merely "the registry class was used"."""
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    calls: list[int] = []

    def resolver(db_type: str, cfg: Any) -> type | None:
        calls.append(1)
        return None

    connectors.set_connector_resolver(resolver)
    _write(connectors_file)

    assert type(_opened()) is _Registered
    assert type(_opened()) is _Registered
    assert calls == [1], "the second open must have been served from the cache"


def test_a_degraded_connector_is_disposed_rather_than_leaked(
    connectors_file, registered, monkeypatch
) -> None:
    """Not caching a degraded connector would have leaked it, one per query.

    Callers are told not to close a pooled connector, so the only things that
    ever close one are `_cache_get` finding it stale and `_cache_put` replacing
    it. A connector that reaches neither lives until garbage collection -- so a
    resolver failing *persistently* rather than transiently would have every
    query build an engine that nothing closes, which is the connection churn the
    cache exists to remove.

    Holding it under an unmatchable fingerprint keeps both properties: never
    reused, always disposed by the open that follows.
    """
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)

    def always_raises(db_type: str, cfg: Any) -> type | None:
        raise RuntimeError("the configuration service is down and staying down")

    connectors.set_connector_resolver(always_raises)
    _write(connectors_file)

    first = _opened()
    assert type(first) is _Registered
    assert not first.closed

    second = _opened()
    assert second is not first, "a degraded connector must never be reused"
    assert first.closed, "the previous degraded connector was not disposed"


def test_a_resolver_cannot_mutate_the_entry(connectors_file, registered, caplog) -> None:
    """The entry is already fingerprinted and credentials are built from it after.

    A resolver adding a default or rewriting `url` would change what the
    connector is built from without changing the fingerprint it is cached under,
    so the cache would describe a configuration the connector does not have. The
    `Mapping` annotation says not to; this makes it impossible, and a resolver
    that tries is reported like any other misbehaving one rather than silently
    half-applied.
    """

    def mutating(db_type: str, cfg: Any) -> type | None:
        cfg["url"] = "postgresql://smuggled/in"  # noqa: TRY301
        return _Resolved

    connectors.set_connector_resolver(mutating)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=connectors.logger.name):
        opened = _opened()

    assert type(opened) is _Registered, "the mutation attempt is a resolver fault"
    assert "connector resolver raised" in caplog.text
    assert opened.credentials["url"] == "postgresql://user:hunter2@localhost:5432/db", (
        "the entry the connector was built from must be the one that was read"
    )


def test_a_resolver_cannot_mutate_a_nested_value_either(connectors_file, registered) -> None:
    """The read-only wrapper is skin deep; the copy underneath it is not.

    `MappingProxyType` refuses `cfg["url"] = ...` and does nothing whatever about
    `cfg["options"]["sslmode"] = ...`, which reaches the loader's own dict through
    the shared nested value and raises nothing on the way. That is the same
    failure the wrapper exists to prevent, arriving one level down and more
    quietly, since there is no warning to notice.

    No entry written by `_save_connector` nests today. Entries are also
    hand-edited YAML, and the enterprise projection passes through whatever a
    connector was configured with, so this is a guarantee rather than a
    restatement of current shapes.
    """
    seen: list[Any] = []

    def nesting(db_type: str, cfg: Any) -> type | None:
        cfg["options"]["sslmode"] = "disable"
        seen.append(cfg["options"]["sslmode"])
        return _Resolved

    connectors.set_connector_resolver(nesting)
    _write(connectors_file, options={"sslmode": "verify-full"})

    opened = _opened()

    assert seen == ["disable"], "the resolver's own copy is writable, so it did not raise"
    assert type(opened) is _Resolved, "and the resolution itself still succeeded"
    assert opened.credentials["options"] == {"sslmode": "verify-full"}, (
        "the connector must be built from the entry as read, not as the resolver left it"
    )


def test_a_resolver_returning_an_instance_is_reported_as_a_resolver_fault(
    connectors_file, registered, caplog
) -> None:
    """`return MyConnector()` for `return MyConnector` is the easy slip.

    Without a check it passes straight through to `connector_cls()` in the
    loader, where the TypeError is caught by the same handler that reports a
    missing driver or a malformed entry -- "usually its configuration rather than
    the server". That sends the operator to the connectors file, and nothing in
    the message mentions a resolver at all, which is the opposite of the
    diagnosis the raising path goes to such trouble to produce.

    A resolver can be wrong in more than one way and all of them should read the
    same, so this is handled exactly as raising is: named, fallen back from, and
    not reused.
    """
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved())
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=connectors.logger.name):
        assert type(_opened()) is _Registered
    assert "not a class" in caplog.text
    assert "_Resolved" in caplog.text, "the message must name what was returned"


def test_a_non_class_result_is_degraded_and_so_is_never_reused(
    connectors_file, registered, monkeypatch
) -> None:
    """The control for treating it 'exactly as raising is treated'.

    Warning about it and then caching the fallback would leave the wrong class
    serving that connector for the whole TTL, which is the failure the degraded
    path exists to prevent -- so the classification matters, not just the message.
    """
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    calls: list[int] = []

    def resolver(db_type: str, cfg: Any) -> Any:
        calls.append(1)
        return _Resolved() if len(calls) == 1 else _Resolved

    connectors.set_connector_resolver(resolver)
    _write(connectors_file)

    assert type(_opened()) is _Registered
    assert type(_opened()) is _Resolved, "a non-class result must not be cached and reused"


# ---------------------------------------------------------------------------
# Every CLI site, not just the ones a test happened to touch
# ---------------------------------------------------------------------------

#: Every connector class the CLI built during a case, in order.
_built: list[type] = []


class _Recorded(_Registered):
    """Records its own construction, so a case can say which class was built."""

    def __init__(self) -> None:
        super().__init__()
        _built.append(type(self))

    def extract_schema(self) -> Any:
        raise RuntimeError("stop here; what was built is what this measures")

    def extract_query_history(self, days: int = 30, limit: int = 500) -> Any:
        raise RuntimeError("stop here; what was built is what this measures")


class _RecordedRegistered(_Recorded):
    """What the registry holds."""


class _RecordedResolved(_Recorded):
    """What a resolver returns instead."""


def _doctor() -> None:
    from nlqueries.cli import main as cli_main

    cli_main._check_connectors(None)  # noqa: SLF001


def _run(*args: str) -> None:
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    CliRunner().invoke(cli, list(args))


def _redirect_kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the knowledge-base path somewhere disposable, in *both* places.

    `cli/main.py` binds `KB_PATH` at import alongside `CONNECTORS_FILE`, so
    patching `config.KB_PATH` alone leaves the commands using the operator's real
    `~/.nlqueries/knowledge_base`: `export-kb` builds its output path from the
    module-level name and creates that directory before any connector is built,
    and `kb-stats` reads whatever is already there. The same two-binding trap the
    `connectors_file` fixture above exists for, one constant along, and
    `tests/test_kb_stats.py` rebinds both for exactly this reason.
    """
    kb = tmp_path / "kb"
    monkeypatch.setattr(config, "KB_PATH", kb)
    monkeypatch.setattr(cli_main, "KB_PATH", kb)


#: One entry per resolution site in ``cli/main.py`` that reads an entry.
#:
#: ``connect`` is covered above and is absent here because it has no entry to
#: read; the ``query`` command's site is covered by
#: ``tests/security/test_execution_boundary.py``.
_CLI_SITES = {
    "doctor": _doctor,
    "extract-schema": lambda: _run("extract-schema", _AGENT),
    "process-history": lambda: _run(
        "process-history", _AGENT, "--days", "30", "--no-annotate", "--no-embed"
    ),
    "export-kb": lambda: _run("export-kb", _AGENT),
    "kb-stats": lambda: _run("kb-stats", _AGENT),
}


@pytest.mark.parametrize("site", sorted(_CLI_SITES))
def test_every_cli_site_builds_the_resolved_class(
    site: str, connectors_file, monkeypatch, tmp_path: Path
) -> None:
    """The invariant this whole seam exists to establish, at each site that has one.

    Each of these was a single mechanical line, and each would survive a revert to
    ``CONNECTOR_REGISTRY.get`` with the suite green: the tests that touch them
    substitute into the registry itself, so they pass whichever way the class is
    found, and nothing installs a resolver on those paths. A change that means
    "resolution happens in one place" is worth only as much as the coverage that
    would notice it stopping.

    What is asserted is the class the command actually constructed, so a site that
    resolved correctly and then built something else would still fail. The
    commands are allowed to fail afterwards -- the stub refuses to extract a
    schema -- because what happens after the connector is built is not what this
    measures.
    """
    _redirect_kb(tmp_path, monkeypatch)
    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _RecordedRegistered)
    _write(connectors_file)
    _built.clear()

    connectors.set_connector_resolver(lambda db_type, cfg: _RecordedResolved)
    _CLI_SITES[site]()

    assert _built, f"{site} never built a connector, so it cannot be measuring resolution"
    assert _built == [_RecordedResolved] * len(_built), (
        f"{site} built {_built}, so it did not go through the seam"
    )


@pytest.mark.parametrize("site", sorted(_CLI_SITES))
def test_every_cli_site_falls_back_to_the_registry(
    site: str, connectors_file, monkeypatch, tmp_path: Path
) -> None:
    """The control. A case that never reached resolution would pass the test above
    against a resolver returning the subclass for everything, and would pass this
    one too -- but only one of the two can pass if the built class is asserted
    both ways, which is why both exist."""
    _redirect_kb(tmp_path, monkeypatch)
    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _RecordedRegistered)
    _write(connectors_file)
    _built.clear()

    connectors.set_connector_resolver(lambda db_type, cfg: None)
    _CLI_SITES[site]()

    assert _built == [_RecordedRegistered] * len(_built)
    assert _built, f"{site} never built a connector"
