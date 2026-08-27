"""
Qdrant must be authenticated unless it is loopback (SEC-09).

A security audit reproduced the full chain: an anonymous writer put a forged
entry into the semantic cache, the cache returned it for a matching question,
and the SQL inside it executed against the customer's database. Qdrant is an
input rather than a cache, and it accepted writes without authentication.
"""

from __future__ import annotations

import pytest
from nlqueries.embeddings.qdrant_client import (
    QdrantAuthenticationRequired,
    is_loopback,
    require_qdrant_auth,
)


class TestLoopbackDetection:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:6333",
            "http://127.0.0.1:6333",
            "http://127.1.2.3:6333",
            "https://LOCALHOST:6333",
            "http://[::1]:6333",
        ],
    )
    def test_local_urls_are_loopback(self, url: str) -> None:
        assert is_loopback(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://qdrant:6333",  # the compose service name — not loopback
            "http://10.0.0.5:6333",
            "https://vectors.example.com",
            "http://192.168.1.10:6333",
        ],
    )
    def test_everything_else_is_not(self, url: str) -> None:
        assert not is_loopback(url)


class TestAuthRequirement:
    def test_loopback_needs_no_key(self) -> None:
        """A developer's own Qdrant is not a security boundary worth ceremony."""
        require_qdrant_auth("http://localhost:6333", "")

    def test_remote_without_a_key_is_refused(self) -> None:
        with pytest.raises(QdrantAuthenticationRequired) as excinfo:
            require_qdrant_auth("http://qdrant:6333", "")

        # The message has to carry the fix, because the person who hits this is
        # deploying, not reading source.
        message = str(excinfo.value)
        assert "QDRANT_API_KEY" in message
        assert "QDRANT__SERVICE__API_KEY" in message
        assert "openssl rand -hex 32" in message

    def test_remote_with_a_key_is_allowed(self) -> None:
        require_qdrant_auth("http://qdrant:6333", "a-real-key")

    def test_a_private_network_is_not_an_exception(self) -> None:
        """The audit's chain needs one compromised neighbour, not an internet
        route, so 'it is only on the internal network' is not a reason to skip
        authentication."""
        with pytest.raises(QdrantAuthenticationRequired):
            require_qdrant_auth("http://10.1.2.3:6333", "")


def test_there_is_no_way_to_turn_this_off() -> None:
    """Deliberately no opt-out switch.

    A flag that disables this would be found and set by whoever meets the error
    first, at which point the control is decorative. Adding an escape hatch
    later is easy; removing one is not. If this ever grows an env var, this test
    should fail and make someone argue for it.
    """
    import inspect

    from nlqueries.embeddings import qdrant_client

    source = inspect.getsource(qdrant_client)
    for escape in ("ALLOW_UNAUTHENTICATED", "SKIP_AUTH", "INSECURE"):
        assert escape not in source.upper()
