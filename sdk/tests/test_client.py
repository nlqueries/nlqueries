"""Tests for nlqueries_sdk.client (Task 24.1).

All HTTP and WebSocket calls are mocked — no live server is required.
"""

from __future__ import annotations

import asyncio
import json
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries_sdk.client import (
    AgentQueryResult,
    AuthenticationError,
    NLQueriesClient,
    PlanLimitError,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
API_KEY = "nlq_abc1234567890abcdef1234567890ab"


def _make_success_response(body: dict[str, Any]) -> MagicMock:
    """Build a mock httpx.Response that looks like a successful API response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _default_query_body() -> dict[str, Any]:
    return {
        "question": "How many orders were placed last month?",
        "answer": "There were 1,243 orders placed last month.",
        "agent_type": "sql",
        "sql": "SELECT COUNT(*) FROM orders WHERE ...",
        "sql_result": {"columns": ["count"], "rows": [[1243]], "row_count": 1},
        "citations": [],
        "session_id": "sess-abc",
        "latency_ms": 1840,
        "from_cache": False,
    }


class _FakeWS:
    """Minimal fake WebSocket that replaces ``websockets.connect``."""

    def __init__(self, messages: list[str]) -> None:
        self._messages = messages
        self._index: int = 0
        self.sent_payloads: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent_payloads.append(msg)

    def __aiter__(self) -> _FakeWS:
        self._index = 0
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        token = self._messages[self._index]
        self._index += 1
        return token

    async def __aenter__(self) -> _FakeWS:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# test_query_sends_correct_request
# ---------------------------------------------------------------------------


class TestQuerySendsCorrectRequest:
    def test_query_sends_correct_request(self) -> None:
        """query() POSTs to /api/v1/query/{agent_id} with the correct path,
        X-Api-Key header, and JSON body; returns a populated AgentQueryResult."""
        mock_resp = _make_success_response(_default_query_body())

        with patch("nlqueries_sdk.client.httpx.Client") as MockClient:
            mock_http = MockClient.return_value
            mock_http.post.return_value = mock_resp

            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)

            # Verify the httpx.Client was created with the API key header
            MockClient.assert_called_once_with(
                base_url=BASE_URL,
                headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
                timeout=60,
            )

            result = client.query(
                agent_id="sales-agent",
                question="How many orders were placed last month?",
            )

        # Assert the POST was called with the correct path and body
        mock_http.post.assert_called_once_with(
            "/api/v1/query/sales-agent",
            json={
                "question": "How many orders were placed last month?",
                "session_id": None,
                "execute_sql": True,
                "timeout_seconds": 30,
            },
        )

        assert isinstance(result, AgentQueryResult)
        assert result.answer == "There were 1,243 orders placed last month."
        assert result.agent_type == "sql"
        assert result.sql == "SELECT COUNT(*) FROM orders WHERE ..."
        assert result.latency_ms == 1840
        assert result.from_cache is False

    def test_query_passes_session_id_and_execute_sql_false(self) -> None:
        """Optional parameters are forwarded verbatim in the request body."""
        mock_resp = _make_success_response({**_default_query_body(), "sql_result": None})

        with patch("nlqueries_sdk.client.httpx.Client") as MockClient:
            mock_http = MockClient.return_value
            mock_http.post.return_value = mock_resp
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)
            client.query(
                "sales-agent",
                "Show me the SQL only",
                session_id="sess-xyz",
                execute_sql=False,
                timeout_seconds=10,
            )

        _, kwargs = mock_http.post.call_args
        body = kwargs["json"]
        assert body["session_id"] == "sess-xyz"
        assert body["execute_sql"] is False
        assert body["timeout_seconds"] == 10


# ---------------------------------------------------------------------------
# test_query_raises_on_401
# ---------------------------------------------------------------------------


class TestQueryRaisesOn401:
    def test_raises_authentication_error(self) -> None:
        """query() raises AuthenticationError when the server returns HTTP 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("nlqueries_sdk.client.httpx.Client") as MockClient:
            mock_http = MockClient.return_value
            mock_http.post.return_value = mock_resp
            client = NLQueriesClient(base_url=BASE_URL, api_key="nlq_bad_key")

            with pytest.raises(AuthenticationError, match="Authentication failed"):
                client.query("agent", "any question")

    def test_list_agents_raises_authentication_error(self) -> None:
        """list_agents() also raises AuthenticationError on HTTP 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("nlqueries_sdk.client.httpx.Client") as MockClient:
            mock_http = MockClient.return_value
            mock_http.get.return_value = mock_resp
            client = NLQueriesClient(base_url=BASE_URL, api_key="nlq_bad")

            with pytest.raises(AuthenticationError):
                client.list_agents()


# ---------------------------------------------------------------------------
# test_query_raises_on_402
# ---------------------------------------------------------------------------


class TestQueryRaisesOn402:
    def test_raises_plan_limit_error(self) -> None:
        """query() raises PlanLimitError when the server returns HTTP 402."""
        mock_resp = MagicMock()
        mock_resp.status_code = 402

        with patch("nlqueries_sdk.client.httpx.Client") as MockClient:
            mock_http = MockClient.return_value
            mock_http.post.return_value = mock_resp
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)

            with pytest.raises(PlanLimitError, match="Plan limit exceeded"):
                client.query("agent", "any question")


# ---------------------------------------------------------------------------
# test_query_iter_yields_tokens_then_sets_last_result
# ---------------------------------------------------------------------------


class TestQueryIterYieldsTokensThenSetsLastResult:
    def test_yields_text_tokens_and_sets_last_result(self) -> None:
        """query_iter() yields text tokens from the WebSocket stream, then sets
        ``last_result`` from the final structured JSON frame."""
        final_chunk = json.dumps(
            {
                "agent_type": "sql",
                "sql": "SELECT COUNT(*) FROM orders",
                "citations": [],
                "session_id": "sess-stream-123",
                "latency_ms": 420,
                "from_cache": False,
            }
        )
        messages = ["There are ", "42 ", "orders.", final_chunk]
        fake_ws = _FakeWS(messages)

        with (
            patch("nlqueries_sdk.client.httpx.Client"),
            patch("websockets.connect", return_value=fake_ws),
        ):
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)
            tokens = list(client.query_iter("sales-agent", "How many orders?"))

        # Text tokens yielded (final JSON frame is NOT yielded)
        assert tokens == ["There are ", "42 ", "orders."]

        # last_result is populated from the final frame
        assert client.last_result is not None
        assert client.last_result.agent_type == "sql"
        assert client.last_result.sql == "SELECT COUNT(*) FROM orders"
        assert client.last_result.session_id == "sess-stream-123"
        assert client.last_result.latency_ms == 420
        assert client.last_result.from_cache is False
        assert client.last_result.answer == "There are 42 orders."

    def test_query_iter_sends_correct_ws_payload(self) -> None:
        """query_iter() sends the question, agent_id, and session_id over the WS."""
        fake_ws = _FakeWS([])  # no messages — last_result stays None

        with (
            patch("nlqueries_sdk.client.httpx.Client"),
            patch("websockets.connect", return_value=fake_ws),
        ):
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)
            list(client.query_iter("my-agent", "Revenue this quarter?", session_id="s1"))

        assert len(fake_ws.sent_payloads) == 1
        sent = json.loads(fake_ws.sent_payloads[0])
        assert sent["question"] == "Revenue this quarter?"
        assert sent["agent_id"] == "my-agent"
        assert sent["session_id"] == "s1"

    def test_query_iter_no_final_chunk_yields_all_tokens(self) -> None:
        """If no structured final chunk is detected, all messages are yielded raw."""
        messages = ["Hello ", "world"]  # no JSON final frame

        with (
            patch("nlqueries_sdk.client.httpx.Client"),
            patch("websockets.connect", return_value=_FakeWS(messages)),
        ):
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)
            tokens = list(client.query_iter("agent", "question"))

        assert tokens == ["Hello ", "world"]
        assert client.last_result is None


# ---------------------------------------------------------------------------
# test_async_query_works
# ---------------------------------------------------------------------------


class TestAsyncQueryWorks:
    def test_async_query_returns_result(self) -> None:
        """query_async() returns an AgentQueryResult when the server responds 200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _default_query_body()
        mock_resp.raise_for_status.return_value = None

        mock_async_client = MagicMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("nlqueries_sdk.client.httpx.Client"),
            patch(
                "nlqueries_sdk.client.httpx.AsyncClient",
                return_value=mock_async_client,
            ),
        ):
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)
            result = asyncio.run(client.query_async("sales-agent", "How many orders?"))

        mock_async_client.post.assert_called_once_with(
            "/api/v1/query/sales-agent",
            json={
                "question": "How many orders?",
                "session_id": None,
                "execute_sql": True,
                "timeout_seconds": 30,
            },
        )
        assert isinstance(result, AgentQueryResult)
        assert result.answer == "There were 1,243 orders placed last month."
        assert result.agent_type == "sql"

    def test_async_query_raises_on_401(self) -> None:
        """query_async() raises AuthenticationError on HTTP 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        mock_async_client = MagicMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("nlqueries_sdk.client.httpx.Client"),
            patch(
                "nlqueries_sdk.client.httpx.AsyncClient",
                return_value=mock_async_client,
            ),
        ):
            client = NLQueriesClient(base_url=BASE_URL, api_key="nlq_bad")

            with pytest.raises(AuthenticationError):
                asyncio.run(client.query_async("agent", "question"))

    def test_async_query_raises_on_402(self) -> None:
        """query_async() raises PlanLimitError on HTTP 402."""
        mock_resp = MagicMock()
        mock_resp.status_code = 402

        mock_async_client = MagicMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("nlqueries_sdk.client.httpx.Client"),
            patch(
                "nlqueries_sdk.client.httpx.AsyncClient",
                return_value=mock_async_client,
            ),
        ):
            client = NLQueriesClient(base_url=BASE_URL, api_key=API_KEY)

            with pytest.raises(PlanLimitError):
                asyncio.run(client.query_async("agent", "question"))
