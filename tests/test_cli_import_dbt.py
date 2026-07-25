"""
tests.test_cli_import_dbt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``nlqueries import-dbt`` command (gap-bridging Block C, KB-side funnel
feature). Drives the CLI end to end with a real fixture manifest and a small KB
YAML, checking the merge writes dbt docs, respects a manual edit, and reports
the metrics it found.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner
from nlqueries.cli.main import cli

FIXTURES = Path(__file__).parent / "fixtures" / "dbt"


def _write_kb(path: Path) -> None:
    kb = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "",
                    "columns": [
                        {"name": "order_id", "description": ""},
                        {
                            "name": "customer_id",
                            "description": "Hand-written; keep me.",
                            "description_source": "manual",
                        },
                    ],
                }
            ]
        }
    }
    path.write_text(yaml.safe_dump(kb, sort_keys=False), encoding="utf-8")


def test_import_dbt_merges_docs_and_preserves_manual(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.yaml"
    _write_kb(kb_path)

    result = CliRunner().invoke(
        cli, ["import-dbt", str(FIXTURES / "manifest.json"), "--kb", str(kb_path)]
    )
    assert result.exit_code == 0, result.output

    kb = yaml.safe_load(kb_path.read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in kb["schema"]["tables"][0]["columns"]}
    # Empty field filled from dbt.
    assert cols["order_id"]["description"] == "Surrogate key for the order."
    assert cols["order_id"]["description_source"] == "dbt"
    # Manual edit untouched, even though dbt has a doc for it.
    assert cols["customer_id"]["description"] == "Hand-written; keep me."
    assert cols["customer_id"]["description_source"] == "manual"
    # Table description filled from dbt.
    assert kb["schema"]["tables"][0]["description"] == "One row per confirmed customer order."


def test_import_dbt_reports_metrics_from_a_directory(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.yaml"
    _write_kb(kb_path)
    result = CliRunner().invoke(cli, ["import-dbt", str(FIXTURES), "--kb", str(kb_path)])
    assert result.exit_code == 0, result.output
    assert "revenue: sum(amount)" in result.output
    assert "3 metrics" in result.output


def test_import_dbt_writes_to_a_separate_output(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.yaml"
    out_path = tmp_path / "merged.yaml"
    _write_kb(kb_path)
    result = CliRunner().invoke(
        cli,
        ["import-dbt", str(FIXTURES / "manifest.json"), "--kb", str(kb_path), "-o", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    # The source KB is left untouched when --output is given.
    original = yaml.safe_load(kb_path.read_text(encoding="utf-8"))
    assert original["schema"]["tables"][0]["columns"][0]["description"] == ""
