"""
The probes that read Qdrant without a client send its API key.

``health``, the MCP ``health`` tool, process-history's ``--embed`` preflight
and kb-stats' cache count read Qdrant's REST API directly and sent no key. On a
Qdrant that requires one -- every remote deployment, since the client refuses
a remote one without a key -- the health check said "connection refused" (a
401 is a URLError) and the cache count showed n/a with nothing said (from
#343's review). Each now sends ``QDRANT_API_KEY`` as the client does.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nlqueries import config
from nlqueries.embeddings.qdrant_client import qdrant_auth_headers, qdrant_rest_request

_KEY = "qdrant-test-key-1f0e"


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _recording(body: dict[str, Any]) -> tuple[list[Any], Any]:
    """An urlopen stand-in that answers *body* and keeps each request it got."""
    seen: list[Any] = []

    def _urlopen(req: Any, timeout: float = 0) -> _Response:
        seen.append(req)
        return _Response(json.dumps(body).encode())

    return seen, _urlopen


def _key_of(req: Any) -> str | None:
    """The API key a request carries; urllib stores header names capitalised."""
    return req.get_header("Api-key") if hasattr(req, "get_header") else None


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(config, "QDRANT_API_KEY", _KEY)


def test_the_request_carries_the_key_when_one_is_set(keyed: None) -> None:
    req = qdrant_rest_request("http://localhost:6333/collections/x")

    assert _key_of(req) == _KEY
    assert qdrant_auth_headers() == {"api-key": _KEY}


def test_and_nothing_when_none_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")

    assert _key_of(qdrant_rest_request("http://localhost:6333/")) is None
    assert qdrant_auth_headers() == {}


def test_the_cli_health_check_sends_it(keyed: None) -> None:
    from nlqueries.cli import main as cli_main

    seen, urlopen = _recording({"version": "1.18.2"})
    with patch("urllib.request.urlopen", urlopen):
        result = cli_main._check_qdrant("http://localhost:6333")

    assert result.status == "ok", result.detail
    assert [_key_of(r) for r in seen] == [_KEY]


def test_the_mcp_health_tool_sends_it(keyed: None) -> None:
    from nlqueries.mcp_server import server

    seen, urlopen = _recording({})
    with patch("urllib.request.urlopen", urlopen):
        server.health()

    # Qdrant's, not the embedding daemon's /healthz, which is no Qdrant.
    qdrant = [
        r for r in seen if hasattr(r, "full_url") and r.full_url == "http://localhost:6333/healthz"
    ]
    assert [_key_of(r) for r in qdrant] == [_KEY]


def test_kb_stats_counts_the_cache_with_it(keyed: None, tmp_path: Path) -> None:
    from nlqueries.knowledge.kb_stats import compute_kb_stats

    kb_path = tmp_path / "kb.yaml"
    kb_path.write_text(
        yaml.dump({"schema": {"tables": []}, "query_capsules": [], "business_context": {}}),
        encoding="utf-8",
    )
    seen, urlopen = _recording({"result": {"vectors_count": 7}})
    with patch("urllib.request.urlopen", urlopen):
        stats = compute_kb_stats("agent1", kb_path)

    assert stats.cache_entries == 7
    assert [_key_of(r) for r in seen] == [_KEY]


def test_process_historys_preflight_sends_it(keyed: None, tmp_path: Path) -> None:
    """The --embed preflight reads /healthz with httpx, before anything runs."""
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    got: list[dict[str, str]] = []

    def _get(url: str, headers: dict[str, str] | None = None, timeout: float = 0) -> Any:
        got.append(dict(headers or {}))
        raise RuntimeError("stop after the preflight")

    with (
        patch("httpx.get", _get),
        patch("nlqueries.cli.main._require_connector", return_value=MagicMock()),
    ):
        CliRunner().invoke(cli, ["process-history", "agent1", "--no-annotate", "--embed"])

    assert got == [{"api-key": _KEY}]
