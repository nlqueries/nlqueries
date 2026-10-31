# nlqueries-core — OSS (BSL 1.1)
"""Warnings the entry points do not show, installed before anything imports them.

Google's client libraries (``google.api_core``, and ``google.cloud.bigquery``
through it) warn at import that they will stop supporting the running Python
version once it reaches end of life. The BigQuery connector imports them
whenever the connectors load, so every ``nlqueries`` command and the MCP
server's start printed two such warnings on Python 3.11. Nothing the person
running the command does changes it: the Python version is the install's, and
the published image's. Moving the image to a newer Python is what ends it.
"""

from __future__ import annotations

import warnings

#: The start of Google's notice, as ``google.api_core._python_version_support``
#: words it for every package it speaks for.
_GOOGLE_PYTHON_VERSION_NOTICE = (
    r"You are using a Python version .* which Google will stop supporting"
)


def quiet_google_python_version_notice() -> None:
    """Ignore Google's end-of-life notice for the running Python version.

    Only that notice, from the module that raises it: any other warning from
    Google's libraries, and any other ``FutureWarning``, still shows.
    """
    warnings.filterwarnings(
        "ignore",
        message=_GOOGLE_PYTHON_VERSION_NOTICE,
        category=FutureWarning,
        module=r"google\.api_core\._python_version_support",
    )
