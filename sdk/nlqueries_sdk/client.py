"""NLQueries SDK client — thin HTTP/WebSocket wrapper around the NLQueries REST API.

This module is self-contained: it does not import from ``nlqueries-core``
or ``enterprise``.  It communicates with NLQueries deployments exclusively
over HTTP and WebSocket.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
import websockets

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NLQueriesSDKError(Exception):
    """Base exception for all NLQueries SDK errors."""


class AuthenticationError(NLQueriesSDKError):
    """Raised when the server returns HTTP 401 Unauthorized."""


class PlanLimitError(NLQueriesSDKError):
    """Raised when the server returns HTTP 402 (plan limit exceeded)."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentQueryResult:
    """Structured result returned by a completed agent query."""

    question: str
    answer: str
    agent_type: str  # "sql" | "document" | "hybrid"
    sql: str | None
    sql_result: dict[str, Any] | None
    citations: list[dict[str, Any]]
    session_id: str | None
    latency_ms: int
    from_cache: bool = field(default=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_response(data: dict[str, Any]) -> AgentQueryResult:
    """Deserialise an API response dict into an :class:`AgentQueryResult`."""
    return AgentQueryResult(
        question=str(data.get("question", "")),
        answer=str(data.get("answer", "")),
        agent_type=str(data.get("agent_type", "sql")),
        sql=data.get("sql") or None,
        sql_result=data.get("sql_result") or None,
        citations=list(data.get("citations") or []),
        session_id=data.get("session_id") or None,
        latency_ms=int(data.get("latency_ms", 0)),
        from_cache=bool(data.get("from_cache", False)),
    )


def _http_to_ws(url: str) -> str:
    """Convert an ``http``/``https`` URL scheme to ``ws``/``wss``."""
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


def _check_status(response: httpx.Response) -> None:
    """Raise SDK-specific errors for known HTTP status codes."""
    if response.status_code == 401:
        raise AuthenticationError(
            "Authentication failed — verify your API key and that it is active."
        )
    if response.status_code == 402:
        raise PlanLimitError("Plan limit exceeded. Please upgrade your NLQueries plan to continue.")
    response.raise_for_status()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class NLQueriesClient:
    """Synchronous and asynchronous client for the NLQueries REST API.

    Quickstart::

        from nlqueries_sdk import NLQueriesClient

        client = NLQueriesClient(
            base_url="https://my-nlqueries.company.com",
            api_key="nlq_abc123",
        )
        result = client.query(agent_id="sales-agent", question="How many deals closed?")
        print(result.answer)
        print(result.sql)

    Args:
        base_url:  Full URL of the NLQueries server (no trailing slash).
        api_key:   Agent API key (format ``nlq_<32 hex chars>``).
        timeout:   Default HTTP timeout in seconds (default 60).
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._headers: dict[str, str] = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        }
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=timeout,
        )
        self.last_result: AgentQueryResult | None = None

    # ------------------------------------------------------------------
    # Synchronous query
    # ------------------------------------------------------------------

    def query(
        self,
        agent_id: str,
        question: str,
        session_id: str | None = None,
        execute_sql: bool = True,
        timeout_seconds: int = 30,
    ) -> AgentQueryResult:
        """Synchronous query — calls ``POST /api/v1/query/{agent_id}``.

        Args:
            agent_id:        Target agent identifier.
            question:        Natural-language question to ask.
            session_id:      Optional conversation session UUID for multi-turn chat.
            execute_sql:     When ``True`` (default), the generated SQL is executed
                             and ``sql_result`` is populated.  Set to ``False`` to
                             receive the SQL for preview without running it.
            timeout_seconds: Server-side timeout (max 120; default 30).

        Returns:
            :class:`AgentQueryResult` with the complete answer.

        Raises:
            :class:`AuthenticationError`: on HTTP 401.
            :class:`PlanLimitError`:       on HTTP 402.
            ``httpx.HTTPStatusError``:    on other 4xx/5xx responses.
        """
        payload: dict[str, Any] = {
            "question": question,
            "session_id": session_id,
            "execute_sql": execute_sql,
            "timeout_seconds": timeout_seconds,
        }
        response = self._http.post(f"/api/v1/query/{agent_id}", json=payload)
        _check_status(response)
        return _parse_response(response.json())

    # ------------------------------------------------------------------
    # Async query
    # ------------------------------------------------------------------

    async def query_async(
        self,
        agent_id: str,
        question: str,
        session_id: str | None = None,
        execute_sql: bool = True,
        timeout_seconds: int = 30,
    ) -> AgentQueryResult:
        """Async version of :meth:`query` — uses ``httpx.AsyncClient``.

        Suitable for callers that already have a running event loop (e.g.
        FastAPI route handlers, Jupyter notebooks with ``asyncio`` support).
        """
        payload: dict[str, Any] = {
            "question": question,
            "session_id": session_id,
            "execute_sql": execute_sql,
            "timeout_seconds": timeout_seconds,
        }
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        ) as client:
            response = await client.post(f"/api/v1/query/{agent_id}", json=payload)
        _check_status(response)
        return _parse_response(response.json())

    # ------------------------------------------------------------------
    # Streaming WebSocket query
    # ------------------------------------------------------------------

    async def _ws_stream(
        self,
        agent_id: str,
        question: str,
        session_id: str | None = None,
    ) -> list[str]:
        """Open a WebSocket connection and collect all streamed message frames."""
        ws_base = _http_to_ws(self._base_url)
        qs = urlencode({"api_key": self._api_key})
        ws_url = f"{ws_base}/api/v1/chat/ws/{agent_id}?{qs}"

        payload = json.dumps({"question": question, "agent_id": agent_id, "session_id": session_id})
        tokens: list[str] = []
        async with websockets.connect(ws_url) as ws:
            await ws.send(payload)
            async for msg in ws:
                tokens.append(str(msg))
        return tokens

    def query_iter(
        self,
        agent_id: str,
        question: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Streaming query via WebSocket — yields text tokens as they arrive.

        After the iterator is exhausted, :attr:`last_result` is set with the
        final structured :class:`AgentQueryResult` parsed from the closing
        JSON frame sent by the server.

        Example::

            for token in client.query_iter("sales-agent", "Monthly revenue?"):
                print(token, end="", flush=True)
            print()
            print(client.last_result.sql)

        Raises:
            :class:`StopIteration`:  when all tokens have been yielded.
        """
        tokens = asyncio.run(self._ws_stream(agent_id, question, session_id=session_id))

        text_tokens = tokens
        final: dict[str, Any] | None = None

        if tokens:
            try:
                parsed = json.loads(tokens[-1])
                if isinstance(parsed, dict) and "agent_type" in parsed:
                    final = parsed
                    text_tokens = tokens[:-1]
            except (json.JSONDecodeError, TypeError):
                pass

        if final is not None:
            self.last_result = AgentQueryResult(
                question=question,
                answer="".join(text_tokens),
                agent_type=str(final.get("agent_type", "sql")),
                sql=final.get("sql") or None,
                sql_result=final.get("sql_result") or None,
                citations=list(final.get("citations") or []),
                session_id=final.get("session_id") or None,
                latency_ms=int(final.get("latency_ms", 0)),
                from_cache=bool(final.get("from_cache", False)),
            )

        yield from text_tokens

    # ------------------------------------------------------------------
    # Agent listing
    # ------------------------------------------------------------------

    def list_agents(self) -> list[dict[str, Any]]:
        """``GET /api/v1/agents`` — return the list of configured agents.

        Note:
            This endpoint is authenticated with a **user JWT**, not an agent
            API key.  Pass the JWT as the ``api_key`` argument if your
            deployment requires it, or configure a bearer token via a custom
            ``httpx.Client`` before calling this method.
        """
        response = self._http.get("/api/v1/agents")
        _check_status(response)
        data: list[dict[str, Any]] = response.json()
        return data
