"""`nlqueries agent docs`: the document sources an agent may answer from.

The list lives on the agent's entry in the connectors file, so the CLI's
connector ID and the MCP server's sanitised form read the same one
(`document_sources_for`). Registering the connector again keeps it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from nlqueries import config
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import cli
from nlqueries.connectors.loader import document_sources_for

_AGENT = "postgres:localhost:shop"


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One connector, in a file both the CLI's writer and its readers use."""
    path = tmp_path / "connectors.yaml"
    path.write_text(
        yaml.dump({_AGENT: {"db_type": "postgres", "database": "shop", "alias": "shop"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_main, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    return path


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The sources Qdrant holds, without asking a Qdrant."""
    held: set[str] = {"q1-report"}
    monkeypatch.setattr(cli_main, "_source_is_stored", lambda source_id: source_id in held)
    return held


def _entry(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))[_AGENT]
    return entry


def test_add_lists_the_sources_once_and_keeps_the_rest_of_the_entry(
    connectors_file: Path, stored: set[str]
) -> None:
    runner = CliRunner()

    first = runner.invoke(cli, ["agent", "docs", "add", "shop", "q1-report", "eng-wiki"])
    again = runner.invoke(cli, ["agent", "docs", "add", _AGENT, "q1-report"])

    assert first.exit_code == 0, first.output
    assert again.exit_code == 0, again.output
    assert "already listed" in again.output
    entry = _entry(connectors_file)
    assert entry["document_sources"] == ["q1-report", "eng-wiki"]
    assert entry["alias"] == "shop" and entry["database"] == "shop"


def test_a_source_not_ingested_yet_is_added_with_a_warning(
    connectors_file: Path, stored: set[str]
) -> None:
    result = CliRunner().invoke(cli, ["agent", "docs", "add", "shop", "eng-wiki"])

    assert result.exit_code == 0, result.output
    assert "not ingested yet" in result.output
    assert _entry(connectors_file)["document_sources"] == ["eng-wiki"]


def test_when_qdrant_cannot_say_the_source_is_still_added(
    connectors_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_main, "_source_is_stored", lambda source_id: None)

    result = CliRunner().invoke(cli, ["agent", "docs", "add", "shop", "q1-report"])

    assert result.exit_code == 0, result.output
    assert "could not be asked" in result.output


def test_remove_drops_only_what_it_names(connectors_file: Path, stored: set[str]) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["agent", "docs", "add", "shop", "q1-report", "eng-wiki"])

    result = runner.invoke(cli, ["agent", "docs", "remove", "shop", "q1-report"])

    assert result.exit_code == 0, result.output
    assert _entry(connectors_file)["document_sources"] == ["eng-wiki"]


def test_removing_a_source_not_listed_is_refused(connectors_file: Path, stored: set[str]) -> None:
    result = CliRunner().invoke(cli, ["agent", "docs", "remove", "shop", "nope"])

    assert result.exit_code != 0
    assert "Not listed" in result.output
    assert "document_sources" not in _entry(connectors_file)


def test_list_shows_the_sources_or_says_there_are_none(
    connectors_file: Path, stored: set[str]
) -> None:
    runner = CliRunner()
    empty = runner.invoke(cli, ["agent", "docs", "list", "shop"])
    runner.invoke(cli, ["agent", "docs", "add", "shop", "q1-report"])
    listed = runner.invoke(cli, ["agent", "docs", "list", "shop"])

    assert "has no document sources" in empty.output
    assert "q1-report" in listed.output


def test_an_unknown_agent_is_refused(connectors_file: Path, stored: set[str]) -> None:
    result = CliRunner().invoke(cli, ["agent", "docs", "add", "nobody", "q1-report"])

    assert result.exit_code != 0
    assert "not found" in result.output


def test_both_forms_of_the_id_read_the_same_list(connectors_file: Path, stored: set[str]) -> None:
    """The CLI uses the connector ID, the MCP server its sanitised form."""
    CliRunner().invoke(cli, ["agent", "docs", "add", "shop", "q1-report"])

    assert document_sources_for(_AGENT) == ["q1-report"]
    assert document_sources_for("postgres_localhost_shop") == ["q1-report"]


@pytest.mark.parametrize(
    "raw", [None, "q1-report", {"a": 1}, [None, 3, " ", "q1-report", "q1-report"]]
)
def test_a_hand_edited_list_reads_as_what_it_validly_holds(
    connectors_file: Path, raw: object
) -> None:
    data = yaml.safe_load(connectors_file.read_text(encoding="utf-8"))
    data[_AGENT]["document_sources"] = raw
    connectors_file.write_text(yaml.dump(data), encoding="utf-8")

    expected = ["q1-report"] if isinstance(raw, list) else []
    assert document_sources_for(_AGENT) == expected


def test_an_agent_with_no_entry_has_no_sources(connectors_file: Path) -> None:
    assert document_sources_for("someone-else") == []


def test_registering_the_connector_again_keeps_its_sources(
    connectors_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new password or a moved host rebuilds the entry; the sources are not
    connection settings and carry over."""
    from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector

    monkeypatch.setattr(cli_main, "_save_password", lambda connector_id, password: True)
    monkeypatch.setattr(SQLAlchemyConnector, "connect", lambda self, credentials: None)
    monkeypatch.setattr(SQLAlchemyConnector, "connection_problem", lambda self: None)
    cid = "sqlalchemy:postgresql:db.example:shop"
    data = yaml.safe_load(connectors_file.read_text(encoding="utf-8"))
    data[cid] = {"db_type": "sqlalchemy", "document_sources": ["q1-report"]}
    connectors_file.write_text(yaml.dump(data), encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["connect", "sqlalchemy", "--url", "postgresql://u:pw@db.example/shop"]
    )

    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(connectors_file.read_text(encoding="utf-8"))[cid]
    assert entry["document_sources"] == ["q1-report"]
    assert entry["db_type"] == "sqlalchemy" and "registered" in entry
