"""A connectors-file entry that is not a mapping, outside the loader.

``agent-a: postgresql://host/db`` -- an id whose value is the URL rather than a
mapping containing it -- is a plausible hand-edit. ``open_connector_for_agent``
already reports it. The other readers called ``cfg.get`` on it and raised
AttributeError: alias resolution (so every command given an alias),
``_require_connector`` (so every command that opens one), ``kb-stats``,
``doctor``'s database checks, ``nlqueries connectors`` and the MCP
``list_connectors`` tool.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import _check_connectors, _require_connector, _resolve_alias, cli
from rich.console import Console

CONNECTORS: dict[str, Any] = {
    "bad": "postgresql://h/db",
    "good": {"db_type": "postgres", "alias": "shop", "host": "h", "database": "db"},
}


def test_an_alias_still_resolves_past_a_malformed_entry() -> None:
    with patch.object(cli_main, "_load_connectors", return_value=CONNECTORS):
        assert _resolve_alias("shop") == "good"
        # Not an alias of anything: returned unchanged, for the caller's own
        # "not found", rather than raising on the way.
        assert _resolve_alias("nothing") == "nothing"


def test_the_connectors_listing_names_the_malformed_entry() -> None:
    out = io.StringIO()
    with (
        patch.object(cli_main, "_load_connectors", return_value=CONNECTORS),
        patch.object(cli_main, "console", Console(file=out, width=200)),
    ):
        result = CliRunner().invoke(cli, ["connectors"])
    assert result.exit_code == 0, result.output
    text = out.getvalue()
    assert "bad" in text and "not a mapping (str)" in text
    assert "good" in text and "shop" in text


def test_doctor_reports_the_malformed_entry_and_checks_the_rest() -> None:
    with (
        patch.object(cli_main, "_load_connectors", return_value=CONNECTORS),
        # The well-formed entry is checked as usual; no class keeps it offline.
        patch.object(cli_main, "connector_class_for", return_value=None),
    ):
        results = _check_connectors(None)
    by_service = {r.service: r for r in results}
    bad = by_service["Database (bad)"]
    assert bad.status == "fail"
    assert "is a str, not a mapping" in bad.detail
    assert by_service["Database (shop)"].status == "skip"


def test_the_mcp_listing_names_the_malformed_entry(tmp_path: Path) -> None:
    from nlqueries.mcp_server.server import list_connectors

    f = tmp_path / "connectors.yaml"
    f.write_text(
        "bad: postgresql://h/db\ngood:\n  db_type: postgres\n  host: h\n  database: db\n",
        encoding="utf-8",
    )
    with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
        out = list_connectors()
    assert "**bad** (not a mapping of connection settings: str)" in out
    assert "**good** (postgres)" in out


def test_the_mcp_listing_refuses_a_file_that_is_not_a_mapping(tmp_path: Path) -> None:
    from nlqueries.mcp_server.server import list_connectors

    f = tmp_path / "connectors.yaml"
    f.write_text("- db_type: postgres\n", encoding="utf-8")
    with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
        out = list_connectors()
    assert out.startswith("Failed to read connectors file: its top level is a list")


def test_a_command_that_opens_the_connector_names_the_bad_entry() -> None:
    """One message, with the fix, for every command that opens a connector."""
    with (
        patch.object(cli_main, "_load_connectors", return_value=CONNECTORS),
        pytest.raises(click.ClickException, match="'bad' .* is a str, not a mapping"),
    ):
        _require_connector("bad")
    with patch.object(cli_main, "_load_connectors", return_value=CONNECTORS):
        assert _require_connector("good") is CONNECTORS["good"]


def test_kb_stats_skips_the_connection_for_a_malformed_entry(tmp_path: Path) -> None:
    """Best-effort: it carries on to its own report, here the missing KB."""
    with (
        patch.object(cli_main, "_load_connectors", return_value=CONNECTORS),
        patch.object(cli_main, "KB_PATH", tmp_path),
    ):
        result = CliRunner().invoke(cli, ["kb-stats", "bad"])
    assert not isinstance(result.exception, AttributeError), result.exception
    assert result.exit_code == 1


# --- An empty entry, `agent-a:` with nothing after it, reads as empty ----------

EMPTY: dict[str, Any] = {"blank": None, **CONNECTORS}


def test_an_empty_entry_is_called_empty_not_a_nonetype() -> None:
    """The loader calls this file state empty; "a NoneType" is Python's word."""
    with (
        patch.object(cli_main, "_load_connectors", return_value=EMPTY),
        pytest.raises(click.ClickException) as refused,
    ):
        _require_connector("blank")
    assert "is empty" in refused.value.message
    assert "NoneType" not in refused.value.message

    with (
        patch.object(cli_main, "_load_connectors", return_value=EMPTY),
        patch.object(cli_main, "connector_class_for", return_value=None),
    ):
        blank = {r.service: r for r in _check_connectors(None)}["Database (blank)"]
    assert blank.status == "fail" and blank.detail.endswith("is empty")


def test_the_listings_call_an_empty_entry_empty(tmp_path: Path) -> None:
    out = io.StringIO()
    with (
        patch.object(cli_main, "_load_connectors", return_value=EMPTY),
        patch.object(cli_main, "console", Console(file=out, width=200)),
    ):
        CliRunner().invoke(cli, ["connectors"])
    assert "empty" in out.getvalue() and "NoneType" not in out.getvalue()

    from nlqueries.mcp_server.server import list_connectors

    f = tmp_path / "connectors.yaml"
    f.write_text("blank:\ngood:\n  db_type: postgres\n", encoding="utf-8")
    with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
        listed = list_connectors()
    assert "**blank** (empty entry)" in listed
