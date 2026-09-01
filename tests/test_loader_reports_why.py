"""``open_connector_for_agent`` must say why it returned ``None``.

It has five ways of failing — no connector id for the agent, no connectors file,
the connector absent from that file, an unregistered ``db_type``, and a refused
connection — and every one of them returned a bare ``None``. The caller cannot
tell them apart, and each has a different fix.

Downstream that silence becomes an answer with no rows and no error: the SQL
path reads a ``None`` connector as "nothing to execute", so a misconfigured
agent is indistinguishable from a query that legitimately matched nothing. That
is the shape of the defect this whole area has been fixing, with the reason not
merely dropped at a boundary but never written down at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from nlqueries import config
from nlqueries.connectors import loader
from nlqueries.execution import ExecutionPolicy

_CONNECTOR_ID = "postgres:localhost:db"


@pytest.fixture(autouse=True)
def _clean_cache():
    loader.invalidate_connector_cache()
    yield
    loader.invalidate_connector_cache()


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "connectors.yaml"
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)
    return path


def _write(path: Path, **entry: Any) -> None:
    base = {"db_type": "postgres", "url": "postgresql://user:hunter2@localhost:5432/db"}
    base.update(entry)
    path.write_text(yaml.safe_dump({_CONNECTOR_ID: base}))


def _open(agent: str = _CONNECTOR_ID):
    return loader.open_connector_for_agent(agent, ExecutionPolicy.execute_read_only())


def test_a_successful_open_says_nothing(connectors_file, caplog, monkeypatch) -> None:
    """Canary. Every assertion below looks for a warning, and all of them would
    be satisfied by a function that warns on every call — which would bury the
    real ones and train people to filter the lot out."""

    class _Connector:
        def connect(self, credentials: dict[str, Any]) -> None:
            return None

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Connector)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is not None
    assert caplog.records == []


def test_no_connectors_file_says_so(connectors_file, caplog) -> None:
    """The file is absent entirely — nothing has ever been projected."""
    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "holds no usable connector configuration" in caplog.text


def test_an_agent_with_no_entry_is_told_what_is_there(connectors_file, caplog) -> None:
    """The file exists and nothing in it matches — a different fault from the
    file being missing, with a different fix. Listing what *is* there is what
    makes it actionable: the usual cause is an id that does not match, and the
    right one is in that list."""
    connectors_file.write_text(yaml.safe_dump({"some-other-agent": {"db_type": "postgres"}}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "no entry" in caplog.text
    assert "some-other-agent" in caplog.text, "the message must name what is available"


def test_an_empty_entry_says_so(connectors_file, caplog) -> None:
    """`agent-a:` with no value under it parses to None. Distinct from a missing
    entry, and it was previously indistinguishable from one."""
    connectors_file.write_text(yaml.safe_dump({_CONNECTOR_ID: None}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "is empty" in caplog.text


def test_an_unregistered_db_type_names_what_is_registered(connectors_file, caplog) -> None:
    """Almost always a missing driver extra, and the fix is unguessable without
    knowing what *is* available."""
    # The agent id deliberately shares no substring with any registered type.
    # With the usual `postgres:localhost:db` the assertion below passes whether
    # or not the message lists anything, because the id itself contains
    # "postgres" -- right and wrong look identical, which is no test at all.
    agent = "warehouse-nine"
    connectors_file.write_text(
        yaml.safe_dump({agent: {"db_type": "teradata", "url": "teradata://h/db"}})
    )

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open(agent) is None
    assert "not registered" in caplog.text
    assert "teradata" in caplog.text
    assert "Registered types:" in caplog.text
    assert "postgres" in caplog.text, "the message must list what is registered"


def test_a_refused_connection_reports_the_drivers_own_words(
    connectors_file, caplog, monkeypatch
) -> None:
    """When a connector does refuse at `connect` time, its own words survive.

    The exception's message is the diagnosis and any summary written here would
    be a worse one — "server does not support SSL, but SSL was required" is the
    whole answer to the defect this area has been fixing. Note that the real
    instance of it was raised on the execute path rather than this one, since
    `create_engine` opens no socket; this holds the reporting, not the claim
    that refusals usually land here.
    """

    class _Refusing:
        def connect(self, credentials: dict[str, Any]) -> None:
            raise OSError("server does not support SSL, but SSL was required")

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Refusing)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "No connector could be opened" in caplog.text
    assert "server does not support SSL" in caplog.text, "the driver's message must survive"


def test_a_configuration_fault_is_reported_here_too(connectors_file, caplog) -> None:
    """What actually reaches this branch, most of the time.

    `connect` on the SQLAlchemy-backed connectors only builds an engine —
    `create_engine` opens no socket — so a server refusing us surfaces later in
    `execute_query`, not here. This block covers `credentials_for` as well, and
    an entry with no URL raises `KeyError: 'url'` inside it. The message must not
    send the reader looking at the network for a fault in the file.
    """
    connectors_file.write_text(yaml.safe_dump({_CONNECTOR_ID: {"db_type": "postgres"}}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "No connector could be opened" in caplog.text
    assert "connection attempt failed" not in caplog.text, (
        "this branch is mostly configuration; naming the network misdirects"
    )
    assert "KeyError" in caplog.text, "the traceback carries the actual cause"


def test_the_password_is_not_logged(connectors_file, caplog, monkeypatch) -> None:
    """These lines carry a connector id, a db_type and a driver message. The
    entry they are derived from holds a password in its URL, so this holds the
    line that none of that reaches the log."""

    class _Refusing:
        def connect(self, credentials: dict[str, Any]) -> None:
            raise OSError('connection to server at "localhost", port 5432 failed')

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Refusing)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "hunter2" not in caplog.text


def test_a_non_string_key_does_not_break_the_diagnostic(connectors_file, caplog) -> None:
    """A diagnostic must not be the thing that breaks.

    YAML keys are not necessarily strings: an unquoted `2024:` in a hand-edited
    file parses to an int. That raises `TypeError` in `_find_connector_id`'s
    sanitising loop, and again in the branch listing what the file holds — so a
    lookup that should return `None` cleanly would instead throw into the
    orchestrator's `except` and reach the caller as an opaque error, caused by
    the very code added to explain failures.
    """
    connectors_file.write_text(
        yaml.safe_dump({2024: {"db_type": "postgres", "url": "postgresql://u:p@h/db"}})
    )

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open("no-such-agent") is None  # must not raise
    assert "no entry" in caplog.text
    assert "2024" in caplog.text, "the key should still be listed, as a string"


def test_a_non_string_key_resolves_to_a_string_id(connectors_file) -> None:
    """`_find_connector_id` is annotated `str | None` and must honour it.

    Comparing `str(key)` while returning the raw key made an agent id of "2024"
    match an unquoted `2024:` and come back as an int. That value keys the
    connector cache, so `invalidate_connector_cache("2024")` — the enterprise
    credential-rotation hook — would miss the entry and leave the old
    credential live until the TTL expired. Normalising in `_load_connectors`
    is what keeps the annotation true, so this asserts the type, not the match.
    """
    connectors_file.write_text(
        yaml.safe_dump({2024: {"db_type": "postgres", "url": "postgresql://u:p@h/db"}})
    )

    found = loader._find_connector_id("2024")
    assert found == "2024"
    assert isinstance(found, str), "an int here silently becomes an unclearable cache key"


def test_the_id_listing_is_capped(connectors_file, caplog) -> None:
    """The enumeration is a sample for spotting a near-match, not an inventory.

    In an enterprise deployment this file is the whole instance's agent set, and
    a mismatched id is the misconfiguration this message targets — so an
    unbounded list writes every agent to the log on every such request, growing
    with the deployment rather than with the fault.
    """
    many = {
        f"agent-{i:03d}": {"db_type": "postgres", "url": "postgresql://u:p@h/db"} for i in range(50)
    }
    connectors_file.write_text(yaml.safe_dump(many))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open("no-such-agent") is None
    assert "The file holds 50" in caplog.text, "the count must still be exact"
    assert "(30 more)" in caplog.text
    assert "agent-019" in caplog.text
    assert "agent-020" not in caplog.text, "the listing must stop at the cap"


def test_a_short_listing_is_not_marked_truncated(connectors_file, caplog) -> None:
    """Canary: the cap must not announce itself when it did not bite."""
    few = {f"agent-{i}": {"db_type": "postgres", "url": "postgresql://u:p@h/db"} for i in range(3)}
    connectors_file.write_text(yaml.safe_dump(few))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open("no-such-agent") is None
    assert "more)" not in caplog.text
    assert "agent-2" in caplog.text


@pytest.mark.parametrize(
    ("content", "shape"),
    [("- one\n- two\n", "a sequence"), ("just a string\n", "a scalar"), ("null\n", "null")],
)
def test_a_file_that_is_not_a_mapping_is_reported_not_raised(
    connectors_file, caplog, content: str, shape: str
) -> None:
    """`.items()` assumes the root is a mapping, and a hand-edited file need not be.

    Left unguarded this raises `AttributeError` out of `open_connector_for_agent`,
    which the multi-agent path surfaces as
    `{"error": "'list' object has no attribute 'items'"}` and `binding_for_agent`
    swallows into an empty fingerprint. The documented contract is that a
    connector which cannot be found returns `None`, so this must land on the
    existing warning rather than escaping as an exception.
    """
    connectors_file.write_text(content)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None  # must not raise, whatever the file holds
    assert "holds no usable connector configuration" in caplog.text, shape


def test_a_malformed_url_does_not_put_the_password_in_the_log(connectors_file, caplog) -> None:
    """`credentials_for` calls `make_url` on the password-resolved URL, so an
    entry that does not parse decides whether a credential reaches the log.

    SQLAlchemy 1.4 embedded the offending string in that `ArgumentError`; 2.0
    removed it, and this project floors at `sqlalchemy>=2.0`. So the leak cannot
    happen today — but the floor has no ceiling, and a note saying "I checked
    once" rots. This holds the property instead of asserting the version.
    """
    connectors_file.write_text(
        yaml.safe_dump(
            {_CONNECTOR_ID: {"db_type": "postgres", "url": "postgresql:/u:hunter2@h/db"}}
        )
    )

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "No connector could be opened" in caplog.text
    assert "ArgumentError" in caplog.text, "the traceback should still name the cause"
    assert "hunter2" not in caplog.text


@pytest.mark.parametrize(
    ("entry", "described"),
    [("postgresql://host/db", "str"), ([1, 2], "list")],
)
def test_an_entry_that_is_not_a_mapping_is_reported_not_raised(
    connectors_file, caplog, entry, described: str
) -> None:
    """The root guard covers the file; this covers one entry.

    `agent-a: postgresql://host/db` — an id whose value is the URL rather than a
    mapping containing it — is a plausible hand-edit. It passes the emptiness
    check and then `cfg.get` raises `AttributeError` out of the function: the
    same opaque error the root guard was added to remove, one level down.
    """
    connectors_file.write_text(yaml.safe_dump({_CONNECTOR_ID: entry}))

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None  # must not raise
    assert "not a mapping" in caplog.text
    assert described in caplog.text, "the message should name what was found instead"


def test_a_db_type_key_with_no_value_is_reported_not_raised(connectors_file, caplog) -> None:
    """`db_type:` with nothing under it parses to None. `.get("db_type", "")`
    returns that None happily — the default covers an absent key, not a present
    one holding nothing — and `.lower()` then raises."""
    connectors_file.write_text(
        yaml.safe_dump({_CONNECTOR_ID: {"db_type": None, "url": "postgresql://u:p@h/db"}})
    )

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None  # must not raise
    assert "not registered" in caplog.text


@pytest.mark.parametrize(
    ("content", "found"), [("- one\n- two\n", "list"), ("just a string\n", "str")]
)
def test_a_malformed_root_says_what_it_found(connectors_file, caplog, content, found) -> None:
    """A file that is present and full must not be reported as missing.

    Both cases reach the caller as an empty mapping, so only `_load_connectors`
    can tell them apart — and telling an operator their file is missing while it
    sits there with content in it is the misdirection this module is being
    changed to remove, reintroduced by the guard that fixed the crash.
    """
    connectors_file.write_text(content)

    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "does not contain a mapping of connector ids" in caplog.text
    assert f"root is a {found}" in caplog.text


def test_a_genuinely_absent_file_is_not_called_malformed(connectors_file, caplog) -> None:
    """Canary for the above: the shape complaint must fire on shape, not on
    every empty result, or it becomes the misdirection in the other direction."""
    with caplog.at_level(logging.WARNING, logger=loader.logger.name):
        assert _open() is None
    assert "does not contain a mapping" not in caplog.text
    assert "holds no usable connector configuration" in caplog.text


def test_the_cli_reads_through_the_same_hardened_reader(connectors_file) -> None:
    """One reader, because two drifted the moment this one was hardened.

    The CLI had its own `_load_connectors` with the same `yaml.safe_load(...) or
    {}` idiom, so `nlqueries connectors` still raised AttributeError on the very
    file this change was written to handle, and still compared raw YAML keys.
    """
    from nlqueries.cli.main import _load_connectors as cli_load

    connectors_file.write_text("- not\n- a mapping\n")
    assert cli_load() == {}  # must not raise

    connectors_file.write_text(
        yaml.safe_dump({2024: {"db_type": "postgres", "url": "postgresql://u:p@h/db"}})
    )
    assert list(cli_load()) == ["2024"], "the CLI must see normalised keys too"
