"""Trust in what comes back from the vector store.

Three things decide it: that a Qdrant anyone else can reach is only read with an
API key, that the key signing cache entries is worth its name, and that a cache
entry is bound to the schema of the agent it was made for.
"""

from __future__ import annotations

import ast
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.cache import envelope
from nlqueries.embeddings import qdrant_client as qc
from nlqueries.embeddings.qdrant_client import (
    QdrantAuthenticationRequired,
    is_loopback,
    require_qdrant_auth,
)
from nlqueries.orchestrator import prompt_assembly

PACKAGE = Path(__file__).resolve().parents[1] / "nlqueries"
GOOD_KEY = "k" * envelope.MIN_KEY_BYTES

# ---------------------------------------------------------------------------
# Which Qdrant is local
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:6333",
        "http://LOCALHOST:6333",
        "localhost:6333",
        "http://127.0.0.1:6333",
        "http://127.1.2.3:6333",
        "http://[::1]:6333",
        "http://[::ffff:127.0.0.1]:6333",
    ],
)
def test_these_are_this_machine(url: str) -> None:
    assert is_loopback(url)


@pytest.mark.parametrize(
    "url",
    [
        # qdrant-client adds http:// to these and connects to the named host.
        "qdrant:6333",
        "qdrant",
        "10.0.0.5:6333",
        # A DNS name is not an address, whatever it starts with.
        "http://127.example.com:6333",
        "http://127.0.0.1.example.com:6333",
        # Unreadable is remote: the mistake should cost a key, not a connection.
        "",
        "http://",
    ],
)
def test_these_are_not(url: str) -> None:
    assert not is_loopback(url)


def test_a_scheme_less_remote_url_needs_a_key() -> None:
    with pytest.raises(QdrantAuthenticationRequired):
        require_qdrant_auth("qdrant:6333", "")


class TestPlainHttpWithAKey:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(qc, "_PLAIN_HTTP_WARNED", set())

    @pytest.mark.parametrize("url", ["http://qdrant:6333", "qdrant:6333"])
    def test_a_key_sent_over_plain_http_to_another_host_is_warned_about_once(
        self, url: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`qdrant:6333` is plain http too: qdrant-client adds the scheme."""
        with caplog.at_level(logging.WARNING, logger=qc.__name__):
            require_qdrant_auth(url, GOOD_KEY)
            require_qdrant_auth(url, GOOD_KEY)
        warned = [r for r in caplog.records if "plain http" in r.getMessage()]
        assert len(warned) == 1

    @pytest.mark.parametrize("url", ["https://qdrant:6333", "http://localhost:6333"])
    def test_https_or_this_machine_is_not(self, url: str, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=qc.__name__):
            require_qdrant_auth(url, GOOD_KEY)
        assert not [r for r in caplog.records if "plain http" in r.getMessage()]


# ---------------------------------------------------------------------------
# One place builds a client
# ---------------------------------------------------------------------------


def _client_constructions(path: Path) -> list[int]:
    """Lines in *path* that call ``QdrantClient(...)`` directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "QdrantClient":
            lines.append(node.lineno)
    return lines


def test_only_the_builder_constructs_a_qdrant_client() -> None:
    """A client built anywhere else skips the API-key rule. The verified-
    examples lookup did, and read anonymously from a remote store."""
    builder = PACKAGE / "embeddings" / "qdrant_client.py"
    # The scan has to find the one construction that is allowed, or it is not
    # looking at anything.
    assert _client_constructions(builder)

    elsewhere = {
        str(path.relative_to(PACKAGE)): found
        for path in PACKAGE.rglob("*.py")
        if path != builder and (found := _client_constructions(path))
    }
    assert elsewhere == {}


class TestVerifiedExamplesLookup:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompt_assembly, "_VERIFIED_AUTH_WARNED", [])

    def test_a_remote_store_without_a_key_is_not_read(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")
        constructor = MagicMock()
        with (
            patch("qdrant_client.QdrantClient", constructor),
            caplog.at_level(logging.WARNING, logger=prompt_assembly.__name__),
        ):
            first = prompt_assembly._search_verified("agent_sales_schema", "q", vector=[0.1])
            second = prompt_assembly._search_verified("agent_sales_schema", "q", vector=[0.1])

        assert first == [] and second == []
        constructor.assert_not_called()
        warned = [r for r in caplog.records if "verified examples" in r.getMessage()]
        assert len(warned) == 1

    def test_a_remote_store_with_a_key_is_read_with_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "QDRANT_URL", "https://qdrant:6333")
        monkeypatch.setattr(config, "QDRANT_API_KEY", GOOD_KEY)
        client = MagicMock()
        client.get_collections.return_value.collections = []
        with patch("qdrant_client.QdrantClient", return_value=client) as constructor:
            assert prompt_assembly._search_verified("agent_sales_schema", "q", vector=[0.1]) == []
        assert constructor.call_args.kwargs["api_key"] == GOOD_KEY


# ---------------------------------------------------------------------------
# The signing key
# ---------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.delenv(envelope._KEY_ENV, raising=False)
    monkeypatch.delenv(envelope._KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(envelope, "_ephemeral_key", None)
    monkeypatch.setattr(envelope, "_KEY_ERROR_LOGGED", [])
    return tmp_path


class TestSigningKey:
    def test_a_short_key_in_the_environment_is_refused(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(envelope._KEY_ENV, "x")
        with pytest.raises(envelope.SigningKeyError, match="at least 32"):
            envelope.signing_key()

    def test_a_key_of_the_minimum_length_is_accepted(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(envelope._KEY_ENV, GOOD_KEY)
        assert envelope.signing_key() == GOOD_KEY.encode()

    def test_a_short_mounted_key_file_is_refused(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mounted = state_dir / "mounted"
        mounted.write_text("short\n", encoding="utf-8")
        monkeypatch.setenv(envelope._KEY_FILE_ENV, str(mounted))
        with pytest.raises(envelope.SigningKeyError):
            envelope.signing_key()

    def test_an_unreadable_mounted_key_file_takes_the_same_paths_as_a_short_key(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A root-owned secret in a container running as another user is as
        likely as an empty one: refused at startup, a miss on the query path."""
        monkeypatch.setenv(envelope._KEY_FILE_ENV, str(state_dir / "missing"))
        with pytest.raises(envelope.SigningKeyError, match="cannot be read"):
            envelope.signing_key()
        binding = envelope.CacheBinding("a", "c", "postgres", "s", "3")
        signed = envelope.sign({"question": "q"}, binding, key=GOOD_KEY.encode())
        assert envelope.verify(signed, binding) is False

    def test_an_unreadable_stored_key_takes_the_same_paths(self, state_dir: Path) -> None:
        """The state-directory key, unreadable (written by another user, say),
        is the same condition as an unreadable mounted one."""
        (state_dir / "cache_signing_key").mkdir()  # exists, and cannot be read as a file
        with pytest.raises(envelope.SigningKeyError, match="cannot be read"):
            envelope.signing_key()
        binding = envelope.CacheBinding("a", "c", "postgres", "s", "3")
        signed = envelope.sign({"question": "q"}, binding, key=GOOD_KEY.encode())
        assert envelope.verify(signed, binding) is False

    def test_an_empty_stored_key_is_refused_not_used(self, state_dir: Path) -> None:
        """A crash between creating and writing the file used to leave it
        empty, and an empty key signed entries anyone could reproduce."""
        (state_dir / "cache_signing_key").write_bytes(b"")
        with pytest.raises(envelope.SigningKeyError, match="delete it"):
            envelope.signing_key()

    def test_a_generated_key_is_stored_whole_and_private(self, state_dir: Path) -> None:
        key = envelope.signing_key()
        stored = state_dir / "cache_signing_key"

        assert len(key) == 64 and stored.read_bytes().strip() == key
        # No temporary file is left behind.
        assert [p.name for p in state_dir.iterdir()] == ["cache_signing_key"]
        if sys.platform != "win32":
            assert stored.stat().st_mode & 0o777 == 0o600

    def test_a_key_another_process_published_first_is_the_one_used(self, state_dir: Path) -> None:
        stored = state_dir / "cache_signing_key"
        theirs = b"t" * 64
        stored.write_bytes(theirs)

        assert envelope._publish_key(stored, b"m" * 64) == theirs
        assert stored.read_bytes() == theirs
        assert [p.name for p in state_dir.iterdir()] == ["cache_signing_key"]

    def test_losing_the_race_to_create_the_key_uses_the_winners(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every process sharing the state directory must sign with one key, so
        the one that loses keeps the winner's rather than its own."""
        stored = state_dir / "cache_signing_key"
        theirs = b"t" * 64
        real_link = os.link

        def another_process_wins(src: str, dst: str) -> None:
            Path(dst).write_bytes(theirs)
            real_link(src, dst)  # raises FileExistsError, as the real race would

        monkeypatch.setattr(envelope.os, "link", another_process_wins)

        assert envelope.signing_key() == theirs
        assert stored.read_bytes() == theirs

    def test_without_hard_links_the_key_is_still_stored(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAT, exFAT and some SMB and FUSE mounts have no hard links. The key
        has to be stored there all the same, or every restart signs with a new
        one and nothing in the cache survives it."""

        def no_links(src: str, dst: str) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(envelope.os, "link", no_links)

        key = envelope.signing_key()
        stored = state_dir / "cache_signing_key"
        assert stored.read_bytes().strip() == key
        assert envelope._ephemeral_key is None
        assert [p.name for p in state_dir.iterdir()] == ["cache_signing_key"]

    def test_without_hard_links_a_key_published_meanwhile_is_kept_and_used(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Windows, renaming onto a key another process has just published,
        and still holds, is refused. Retrying the rename would replace a key that
        process already signs with; the winner's key is the one to use."""
        stored = state_dir / "cache_signing_key"
        theirs = b"t" * 64

        def no_links(src: str, dst: str) -> None:
            raise PermissionError(1, "Operation not permitted")

        def published_meanwhile(src: str, dst: str) -> None:
            Path(dst).write_bytes(theirs)
            raise PermissionError(13, "Access is denied")

        monkeypatch.setattr(envelope.os, "link", no_links)
        monkeypatch.setattr(envelope.os, "replace", published_meanwhile)

        assert envelope._publish_key(stored, b"m" * 64) == theirs
        assert stored.read_bytes() == theirs
        assert [p.name for p in state_dir.iterdir()] == ["cache_signing_key"]

    def test_check_signing_key_refuses_a_bad_key_at_startup(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(envelope._KEY_ENV, "x")
        with pytest.raises(envelope.SigningKeyError):
            envelope.check_signing_key()

    def test_on_the_read_path_a_bad_key_is_a_miss_logged_once(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A query is still answered, without the cache; the reason is logged."""
        binding = envelope.CacheBinding("a", "c", "postgres", "s", "3")
        signed = envelope.sign({"question": "q"}, binding, key=GOOD_KEY.encode())
        monkeypatch.setenv(envelope._KEY_ENV, "x")

        with caplog.at_level(logging.ERROR, logger=envelope.__name__):
            assert envelope.verify(signed, binding) is False
            assert envelope.verify(signed, binding) is False

        assert len([r for r in caplog.records if "Semantic cache disabled" in r.getMessage()]) == 1

    def test_the_mcp_server_refuses_to_start_with_a_bad_key(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nlqueries.mcp_server import server

        monkeypatch.setenv(envelope._KEY_ENV, "x")
        run = MagicMock()
        monkeypatch.setattr(server.mcp, "run", run)
        # A message, as the server's other startup refusals give, not a traceback.
        with pytest.raises(SystemExit, match="at least 32"):
            server.main(transport="stdio")
        run.assert_not_called()

    def test_with_a_bad_key_cache_writes_are_skipped_not_raised(
        self, state_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The write side of the read path's miss: a query that would succeed
        must not fail on its way out because the cache is unusable."""
        from nlqueries.cache import semantic_cache as sc

        monkeypatch.setenv(envelope._KEY_ENV, "x")
        binding = envelope.CacheBinding("agent1", "c", "postgres", "s", "3")
        result = MagicMock(
            resolved_question="how many orders",
            agent_type="sql",
            answer="There were 42 orders.",
            sql="SELECT count(*) FROM orders",
        )
        client = MagicMock()
        with (
            patch.object(sc, "_get_client", return_value=client),
            patch.object(sc, "embed_text", return_value=[0.1] * 384),
            patch.object(sc, "ensure_collection"),
            caplog.at_level(logging.ERROR, logger=envelope.__name__),
        ):
            sc.SemanticCache("agent1", binding=binding).put("how many orders", result)
            sc.SemanticCache("agent1", binding=binding).put("how many orders", result)

        client.upsert.assert_not_called()
        assert len([r for r in caplog.records if "Semantic cache disabled" in r.getMessage()]) == 1


# ---------------------------------------------------------------------------
# The schema binding
# ---------------------------------------------------------------------------


def test_a_dotted_agent_id_is_bound_to_its_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`connect` makes ids like `postgres:db.example.com:shop`, and `export-kb`
    writes their knowledge base keeping the dots. The binding dropped them,
    fingerprinted a file that does not exist, and bound every such agent's
    cache to no schema at all."""
    monkeypatch.setattr(config, "KB_PATH", tmp_path)
    agent = "postgres:db.example.com:shop"
    kb = tmp_path / "postgres_db.example.com_shop.yaml"
    kb.write_text("tables: []\n", encoding="utf-8")

    before = envelope.binding_for_agent(agent, "postgres").schema_fingerprint
    kb.write_text("tables: [orders]\n", encoding="utf-8")
    os.utime(kb, ns=(kb.stat().st_mtime_ns + 1_000_000, kb.stat().st_mtime_ns + 1_000_000))
    after = envelope.binding_for_agent(agent, "postgres").schema_fingerprint

    assert before and after and before != after


def test_the_binding_finds_the_knowledge_base_by_the_shared_storage_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By `state_files.agent_storage_id`, not a copy of its pattern: two copies
    are how the binding once fingerprinted a file `export-kb` never wrote."""
    from nlqueries import state_files

    monkeypatch.setattr(config, "KB_PATH", tmp_path)
    monkeypatch.setattr(state_files, "agent_storage_id", lambda agent_id: "stored-as")
    (tmp_path / "stored-as.yaml").write_text("tables: []\n", encoding="utf-8")

    assert envelope.binding_for_agent("any agent", "postgres").schema_fingerprint
