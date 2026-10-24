"""
nlqueries.embeddings.qdrant_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One place that builds a Qdrant client, and one rule about authentication.

A security audit reproduced this chain end to end: an anonymous writer put a
forged entry into the semantic cache, the cache returned it on a matching
question, and the SQL inside it ran against the customer's database. Qdrant is
therefore not a cache in the security sense. It is an input, and one that
required no authentication to write.

The rule is that anything not on the loopback interface must present an API key.
A Qdrant on `localhost` belongs to the running process; a Qdrant on any other
address is reachable by other parties, and
"reachable by something else" is exactly the condition the audit exploited. A
private Docker network is not an exception: the chain needs one compromised
neighbour, not an internet route.

There is deliberately no opt-out switch. A flag that turns this off would be
selected by whoever meets the error first, at which point the control is
decorative. If the strictness proves wrong in practice, adding an escape hatch
later is easy; taking one away is not.

Every call site that used to build a client itself goes through here, which is
the other half of the point: the previous arrangement had the same credential
decision written out several times, and a rule expressed three times is a rule
that will eventually be enforced twice. The verified-examples lookup was the
one left out, and a test now fails if any other module constructs a client.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, urlparse

from nlqueries import config

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

#: URLs already warned about for carrying an API key over plain http.
_PLAIN_HTTP_WARNED: set[str] = set()


class QdrantAuthenticationRequired(RuntimeError):
    """Raised when a non-loopback Qdrant is configured with no API key."""


def _parsed(url: str) -> ParseResult:
    """*url* as qdrant-client reads it.

    qdrant-client adds ``http://`` to a URL without a scheme, so ``qdrant:6333``
    reaches the host ``qdrant`` over plain http. Read without that, ``urlparse``
    takes ``qdrant`` for the scheme and finds no host at all.
    """
    return urlparse(url if "://" in url else f"http://{url}")


def _host(url: str) -> str:
    """The host *url* connects to, as qdrant-client reads it."""
    return (_parsed(url).hostname or "").lower()


def is_loopback(url: str) -> bool:
    """True only when *url* certainly points at this machine.

    ``localhost``, or an IP address in the loopback range (``127.0.0.0/8``,
    ``::1``, and IPv4-mapped ``::ffff:127.x.x.x``). Anything else, including a
    URL whose host cannot be read and a DNS name that merely starts with
    ``127.``, is treated as remote: a mistake then costs a demand for an API key,
    not an unauthenticated connection to somewhere else.
    """
    host = _host(url)
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def _warn_on_plain_http(url: str) -> None:
    """Warn, once per URL, when an API key would cross the network in clear.

    Allowed rather than refused: a private network between this process and
    Qdrant is a common deployment. But the key is the whole of the protection
    against anonymous writes, and over plain http anyone on that path reads it.
    """
    # The scheme qdrant-client will use, not the one written: `qdrant:6333`
    # is plain http too.
    if _parsed(url).scheme.lower() != "http" or is_loopback(url) or url in _PLAIN_HTTP_WARNED:
        return
    _PLAIN_HTTP_WARNED.add(url)
    logger.warning(
        "Qdrant at %s is reached over plain http, so QDRANT_API_KEY is sent "
        "unencrypted to a host that is not this machine. Use https, or keep "
        "Qdrant on a network nothing else can read.",
        url,
    )


def require_qdrant_auth(url: str | None = None, api_key: str | None = None) -> None:
    """Raise unless *url* is loopback or *api_key* is set.

    Arguments default to the configured values; they are parameters so this can
    be tested without reaching into module state.
    """
    resolved_url = config.QDRANT_URL if url is None else url
    resolved_key = config.QDRANT_API_KEY if api_key is None else api_key

    if resolved_key:
        _warn_on_plain_http(resolved_url)
        return
    if is_loopback(resolved_url):
        return

    raise QdrantAuthenticationRequired(
        f"Qdrant at {resolved_url} is not on loopback and no QDRANT_API_KEY is set. "
        "An unauthenticated vector store is a write path into the semantic cache, "
        "and cached SQL is executed against your database. Set QDRANT_API_KEY on "
        "both this process and the Qdrant service "
        "(QDRANT__SERVICE__API_KEY) — generate one with: openssl rand -hex 32"
    )


def build_qdrant_client(**kwargs: object) -> QdrantClient:
    """A configured, authenticated Qdrant client.

    Raises:
        QdrantAuthenticationRequired: the configuration would talk to a remote
            Qdrant anonymously.
    """
    require_qdrant_auth()

    # Deferred, as it was at each of the three call sites this replaces: the
    # qdrant_client package is a heavy import and this module is reachable from
    # start-up paths that never touch a vector store.
    from qdrant_client import QdrantClient  # noqa: PLC0415

    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY or None,
        **kwargs,  # type: ignore[arg-type]
    )
