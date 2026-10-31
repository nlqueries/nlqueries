"""Google's end-of-life notice for the running Python version is not shown.

The BigQuery connector imports Google's client libraries whenever the
connectors load, and on Python 3.11 they warned twice at the start of every
``nlqueries`` command and of the MCP server, about a version the person
running it did not choose.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest
from nlqueries.warning_filters import quiet_google_python_version_notice

_NOTICE = (
    "You are using a Python version (3.11.17) which Google will stop supporting in new "
    "releases of google.api_core once it reaches its end of life (2027-10-24)."
)
_GOOGLE_MODULE = "google.api_core._python_version_support"


def _shown(message: str, category: type[Warning], module: str) -> bool:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        quiet_google_python_version_notice()
        warnings.warn_explicit(message, category, "x.py", 1, module=module)
    return bool(seen)


def test_the_notice_is_not_shown() -> None:
    assert not _shown(_NOTICE, FutureWarning, _GOOGLE_MODULE)


@pytest.mark.parametrize(
    ("message", "category", "module"),
    [
        ("Some other change is coming.", FutureWarning, _GOOGLE_MODULE),
        (_NOTICE, FutureWarning, "nlqueries.connectors.bigquery"),
        (_NOTICE, DeprecationWarning, _GOOGLE_MODULE),
    ],
    ids=["another notice from that module", "the text from elsewhere", "another category"],
)
def test_nothing_else_is_hidden(message: str, category: type[Warning], module: str) -> None:
    assert _shown(message, category, module)


def _stderr_of(code: str) -> str:
    return subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=180,
    ).stderr


@pytest.mark.skipif(
    sys.version_info >= (3, 12), reason="Google gives the notice on Python 3.11 only"
)
@pytest.mark.parametrize(
    "entry",
    [
        "nlqueries.cli.main",
        # The server module does not load the connectors itself; the server
        # does once it starts, and the filter is in place by then.
        "nlqueries.mcp_server, nlqueries.connectors",
    ],
    ids=["cli", "mcp server"],
)
def test_the_entry_points_start_without_it(entry: str) -> None:
    """Through a fresh interpreter, as each entry point starts: the filter has
    to be in place before the connectors import Google's libraries."""
    if "which Google will stop supporting" not in _stderr_of("import google.cloud.bigquery"):
        pytest.skip("this install's Google libraries do not give the notice")
    assert "which Google will stop supporting" not in _stderr_of(f"import {entry}")
