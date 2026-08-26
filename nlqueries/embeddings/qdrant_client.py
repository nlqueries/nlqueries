"""
nlqueries.embeddings.qdrant_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One place that builds a Qdrant client, and one rule about authentication.

A security audit reproduced this chain end to end: an anonymous writer put a
forged entry into the semantic cache, the cache returned it on a matching
question, and the SQL inside it ran against the customer's database. Qdrant is
therefore not a cache in the security sense — it is an input, and an input
nobody had to authenticate to write.

The rule is that anything not on the loopback interface must present an API key.
A Qdrant on `localhost` is a developer's own process and needs no ceremony; a
Qdrant anywhere else is reachable by something other than this process, and
"reachable by something else" is exactly the condition the audit exploited. A
private Docker network is not an exception: the chain needs one compromised
neighbour, not an internet route.

There is deliberately no opt-out switch. A flag that turns this off would be
selected by whoever meets the error first, at which point the control is
decorative. If the strictness proves wrong in practice, adding an escape hatch
later is easy; taking one away is not.

The three call sites that used to build clients themselves all go through here,
which is the other half of the point: the previous arrangement had the same
credential decision written out three times, and a rule expressed three times is
a rule that will eventually be enforced twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from nlqueries import config

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


class QdrantAuthenticationRequired(RuntimeError):
    """Raised when a non-loopback Qdrant is configured with no API key."""


def is_loopback(url: str) -> bool:
    """True when *url* points at this machine.

    IPv6 literals arrive bracketed (``http://[::1]:6333``); urlparse strips the
    brackets for us. An empty host is what a malformed URL parses to, and is
    treated as local rather than remote so a configuration mistake fails on the
    connection attempt with a useful error rather than here with a misleading
    one about authentication.
    """
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def require_qdrant_auth(url: str | None = None, api_key: str | None = None) -> None:
    """Raise unless *url* is loopback or *api_key* is set.

    Arguments default to the configured values; they are parameters so this can
    be tested without reaching into module state.
    """
    resolved_url = config.QDRANT_URL if url is None else url
    resolved_key = config.QDRANT_API_KEY if api_key is None else api_key

    if resolved_key or is_loopback(resolved_url):
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
