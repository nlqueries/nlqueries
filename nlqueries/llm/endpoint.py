# nlqueries-core — OSS (BSL 1.1)
"""A warning for an LLM endpoint reached over plain http.

Every prompt carries the schema and the question, and every request carries
the API key, so the transport to the model matters. A gateway inside the
cluster is often plain http and that is a legitimate deployment, which is why
this warns and does not refuse.
"""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: Endpoints already warned about. Clients are built per request when an
#: override is in play, and a warning that floods is a warning nobody reads.
_WARNED: set[str] = set()


def _is_loopback(host: str) -> bool:
    """``localhost``, or an address ``ipaddress`` calls loopback. Nothing else.

    Not a name lookup: a hostname that happens to resolve to 127.0.0.1 today is
    still a name, and what it resolves to can change without this code knowing.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def warn_if_plain_http(base_url: str | None, setting: str) -> None:
    """Log a warning, once per endpoint per process, for a non-loopback http URL.

    *setting* names where the URL came from (``ANTHROPIC_BASE_URL``,
    ``api_base``) so an operator knows what to change. Only the scheme, host
    and port are logged: a base URL can carry credentials in its userinfo or
    query string, and those do not belong in a log line.
    """
    if not base_url:
        return
    try:
        parts = urlsplit(base_url.strip())
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        # A malformed port or bracket; the client's own connection attempt
        # reports that better than a warning about transport would.
        return
    if parts.scheme.lower() != "http" or _is_loopback(host):
        return
    shown = f"http://[{host}]" if ":" in host else f"http://{host}"
    if port is not None:
        shown = f"{shown}:{port}"
    if shown in _WARNED:
        return
    _WARNED.add(shown)
    logger.warning(
        "%s points the LLM client at %s over plain http. Prompts and the API "
        "key will cross the network unencrypted. Use https unless this gateway "
        "is only reachable on a trusted network.",
        setting,
        shown,
    )
