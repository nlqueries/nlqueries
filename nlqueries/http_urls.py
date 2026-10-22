# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.http_urls
~~~~~~~~~~~~~~~~~~~
Only ``http`` and ``https`` URLs are opened.

``urllib.request.urlopen`` opens ``file:``, ``ftp:`` and ``data:`` URLs as
readily as web ones. Every URL this package opens is a service it speaks HTTP
to -- Qdrant, the embedding daemon -- and comes from configuration, so any
other scheme is a misconfiguration at best, and is refused before ``urlopen``
sees it: ``QDRANT_URL=file:///etc/passwd`` would otherwise be read.
"""

from __future__ import annotations

from urllib.parse import urlsplit

#: The schemes ``urlopen`` is given.
ALLOWED_SCHEMES = frozenset({"http", "https"})


class NotAnHttpUrl(ValueError):
    """A URL with a scheme other than ``http`` or ``https``."""


def http_url(url: str) -> str:
    """*url*, if its scheme is ``http`` or ``https``; otherwise :class:`NotAnHttpUrl`."""
    scheme = urlsplit(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        shown = f"'{scheme}:'" if scheme else "none"
        raise NotAnHttpUrl(f"{url!r} is not an http or https URL (its scheme is {shown})")
    return url
