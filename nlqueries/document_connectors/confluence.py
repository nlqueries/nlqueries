"""
nlqueries.document_connectors.confluence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Confluence document connector using ``atlassian-python-api`` for page retrieval,
``beautifulsoup4`` for HTML stripping, and a built-in recursive character chunker.

Chunking strategy: for each page in the given space, the HTML storage body is
stripped to plain text and split with
``RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)``.
Chunk indices are sequential across all pages in the space.

Confluence has no document-page concept analogous to PDF pages, so
``page_number`` is always ``None``.

Incremental sync: pass ``since`` to restrict the CQL query to pages modified
after the given timestamp.

Requires the ``wiki`` optional dependency group:
    pip install "nlqueries-core[wiki]"
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.chunker import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100


def _make_chunk_id(source_id: str, chunk_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:None:chunk_index)[:16]."""
    raw = f"{source_id}:None:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


#: Results asked for per CQL request while reading a whole space.
_SEARCH_PAGE_SIZE = 100


class IncompleteSpaceRead(RuntimeError):
    """The search stopped returning new pages while Confluence reported more.

    A sync stores what it read as the whole space and removes everything
    else, so a partial read is refused rather than returned.
    """


def _result_id(result: dict[str, Any]) -> str:
    content: dict[str, Any] = result.get("content", result)
    return str(content.get("id", "")) or repr(sorted(result.items()))


def _next_link(response: dict[str, Any]) -> str:
    return str((response.get("_links") or {}).get("next") or "")


def _cursor_of(next_link: str) -> str | None:
    """The ``cursor`` query parameter of a search response's next link, if any."""
    values = parse_qs(urlsplit(next_link).query).get("cursor") if next_link else None
    return values[0] if values else None


def _accepts_cursor(method: Any) -> bool:
    """Whether the installed client's ``cql`` can be given a ``cursor``.

    The ``atlassian-python-api`` range the wiki extra allows differs: the
    3.x/4.x and server ``cql(cql, start, limit, ...)`` takes no cursor, the 5.x
    Cloud client forwards keyword arguments to the search.
    """
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "cursor" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


class ConfluenceConnector(DocumentConnector):
    """Extract and chunk text from a Confluence space using atlassian-python-api.

    Uses CQL to list all pages in the space, fetches each page body via the
    Confluence REST API, strips HTML with BeautifulSoup, and splits with
    ``RecursiveCharacterTextSplitter``.  ``page_number`` is always ``None``
    because Confluence spaces have no single-document page concept.
    """

    def __init__(self, base_url: str, username: str, api_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._api_token = api_token

    def _client(self) -> Any:
        try:
            from atlassian import Confluence
        except ImportError as exc:
            raise ImportError(
                "atlassian-python-api is required for ConfluenceConnector. "
                "Install it with: pip install 'nlqueries-core[wiki]'"
            ) from exc
        return Confluence(
            url=self._base_url,
            username=self._username,
            password=self._api_token,
            cloud=True,
        )

    @staticmethod
    def _cql(space_key: str, since: datetime | None) -> str:
        """CQL for the space's pages, optionally only those modified after *since*."""
        if since is None:
            return f'space = "{space_key}" AND type = page ORDER BY lastModified DESC'
        since_str = since.strftime("%Y-%m-%d %H:%M")
        return (
            f'space = "{space_key}" AND type = page'
            f' AND lastModified > "{since_str}"'
            f" ORDER BY lastModified DESC"
        )

    @staticmethod
    def _search_all(client: Any, cql: str) -> list[dict[str, Any]]:
        """Every result of *cql*, read a page at a time.

        One request used to be the whole read, capped at 200 results, so a
        larger space was cut short with no sign of it. Confluence Cloud's search
        is cursor-paginated and honours ``start`` only so far, so the cursor in
        the response's next link is followed when the client can pass one, and
        ``start`` is used otherwise.

        Progress is the number of raw results consumed, compared with
        ``totalSize`` (or, without one, the presence of a next link): offset
        paging over ``lastModified`` repeats a page that is edited during the
        read, so the de-duplicated count can end short of ``totalSize`` on a
        read that finished. A request that brings nothing new while more is
        reported -- empty, or only repeats, as from a server that ignores
        ``start`` -- raises :class:`IncompleteSpaceRead` instead of returning a
        partial list.
        """
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        consumed = 0
        cursor: str | None = None
        use_cursor = _accepts_cursor(client.cql)
        while True:
            if cursor is not None:
                response: dict[str, Any] = client.cql(cql, limit=_SEARCH_PAGE_SIZE, cursor=cursor)
            else:
                response = client.cql(cql, start=consumed, limit=_SEARCH_PAGE_SIZE)
            batch: list[dict[str, Any]] = response.get("results") or []
            new = [r for r in batch if _result_id(r) not in seen]
            for result in new:
                seen.add(_result_id(result))
                results.append(result)
            consumed += len(batch)
            next_link = _next_link(response)
            total = response.get("totalSize")
            more = consumed < total if isinstance(total, int) else bool(next_link)
            if not more:
                return results
            if not new:
                raise IncompleteSpaceRead(
                    f"the search stopped advancing after {len(results)} page(s) while "
                    "Confluence reports more, so the space could not be read whole"
                )
            cursor = _cursor_of(next_link) if use_cursor else None

    def has_changes_since(self, space_key: str, since: datetime) -> bool:
        """Whether any page in *space_key* was modified after *since*.

        One CQL search for a single result, with no page bodies fetched, so an
        incremental sync can decide whether to re-read the space for the cost
        of one request.
        """
        results: dict[str, Any] = self._client().cql(self._cql(space_key, since), limit=1)
        return bool(results.get("results"))

    def ingest(
        self,
        source: str | Path,
        source_id: str,
        *,
        since: datetime | None = None,
    ) -> list[DocumentChunk]:
        """Fetch all pages in a Confluence space and return text chunks.

        Args:
            source: Confluence space key (e.g. ``"ENG"``).
            source_id: Opaque identifier for this source (used to generate
                       deterministic chunk IDs and for Qdrant filtering).
            since: If provided, restrict the CQL query to pages last modified
                   strictly after this timestamp (incremental sync).

        Returns:
            Ordered list of :class:`DocumentChunk` objects.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for ConfluenceConnector. "
                "Install it with: pip install 'nlqueries-core[wiki]'"
            ) from exc

        space_key = str(source)
        client = self._client()
        page_results = self._search_all(client, self._cql(space_key, since))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )

        all_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page_result in page_results:
            # CQL results may wrap the content under a "content" key.
            content: dict[str, Any] = page_result.get("content", page_result)
            page_id = str(content.get("id", ""))
            page_title: str = content.get("title", "")
            web_ui: str = content.get("_links", {}).get("webui", "")
            page_url = f"{self._base_url}/wiki{web_ui}" if web_ui else ""

            if not page_id:
                continue

            # Fetch the full page body (HTML storage format).
            page_data: dict[str, Any] = client.get_page_by_id(page_id, expand="body.storage")
            html_body: str = page_data.get("body", {}).get("storage", {}).get("value", "")

            if not html_body:
                continue

            # Strip HTML tags to obtain plain text.
            soup = BeautifulSoup(html_body, "html.parser")
            plain_text = soup.get_text(separator="\n", strip=True)

            if not plain_text:
                continue

            text_chunks = splitter.split_text(plain_text)
            for chunk_text in text_chunks:
                chunk_id = _make_chunk_id(source_id, chunk_index)
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_id=source_id,
                        source_name=space_key,
                        page_number=None,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        metadata={
                            "connector": "confluence",
                            "space_key": space_key,
                            "page_id": page_id,
                            "page_title": page_title,
                            "page_url": page_url,
                        },
                    )
                )
                chunk_index += 1

        return all_chunks

    def supports(self, source: str | Path) -> bool:
        """Confluence sources are identified by space key, not file extension."""
        return True
