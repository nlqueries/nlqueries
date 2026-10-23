"""The CLI does not fail on a character its output's encoding lacks.

Redirected or piped on Windows, Python writes in the locale's code page
(cp1252), which has no check mark. Forced here with PYTHONIOENCODING, which does
the same on any platform.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "PYTHONPATH": str(ROOT),
        "NLQ_STATE_DIR": str(home / ".nlqueries"),
        # Set too, as the suite's conftest exports its own and this would inherit it.
        "CONNECTORS_FILE": str(home / ".nlqueries" / "connectors.yaml"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
    }
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, "-c", "from nlqueries.cli.main import cli; cli()", *args],
        capture_output=True,
        env=env,
        timeout=180,
        check=False,
    )


def test_connect_with_output_in_a_code_page_without_its_marks_saves_the_connector(
    tmp_path: Path,
) -> None:
    """`connect` prints a check mark after testing the connection and before
    saving it; strict cp1252 raised there, and the connector was never saved."""
    db = tmp_path / "shop.db"
    sqlite3.connect(db).close()

    ran = _run_cli(tmp_path, "connect", "sqlite", "--database", str(db), "--alias", "shop")

    stderr = ran.stderr.decode("cp1252", "replace")
    assert ran.returncode == 0, stderr
    assert "UnicodeEncodeError" not in stderr
    connectors = tmp_path / "home" / ".nlqueries" / "connectors.yaml"
    assert connectors.exists()
    assert "alias: shop" in connectors.read_text(encoding="utf-8")
    # Written, with what cp1252 lacks replaced rather than dropped.
    assert b"Connection successful" in ran.stdout


def test_a_json_payload_survives_any_output_encoding() -> None:
    """Replacement suits the text people read; a payload a program parses must
    not quietly hold `?`. It is written in ASCII, escaped, so nothing is lost."""
    import json
    from unittest.mock import patch

    from click.testing import CliRunner
    from nlqueries.cli.main import _CheckResult, cli

    detail = "Zoë ✓ 東京"
    ok = _CheckResult("Qdrant", "ok", detail)
    with (
        patch("nlqueries.cli.main._check_qdrant", return_value=ok),
        patch("nlqueries.cli.main._check_connectors", return_value=[]),
        patch("nlqueries.cli.main._check_llm", return_value=ok),
        patch("nlqueries.cli.main._check_embedding", return_value=ok),
        patch("nlqueries.cli.main._check_config", return_value=ok),
    ):
        result = CliRunner().invoke(cli, ["health", "--json"])

    assert result.exit_code == 0, result.output
    assert result.output.isascii()
    payload = json.loads(result.output)
    assert detail in [check.get("detail") for check in payload["checks"]]


def test_a_diagnostic_on_stderr_keeps_what_it_cannot_encode_recoverable(tmp_path: Path) -> None:
    """stderr is escaped, not replaced: a database's message with a character
    the code page lacks should still say what it said."""
    ran = _run_cli(tmp_path, "connect", "sqlite", "--database", str(tmp_path / "missing.db"))

    assert ran.returncode != 0
    assert b"Connection failed" in ran.stderr
    assert b"\\u2717" in ran.stderr  # the cross mark, escaped
    assert b"? Connection failed" not in ran.stderr
