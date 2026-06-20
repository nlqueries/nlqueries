"""
nlqueries.document_connectors.confluence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Confluence document connector using ``atlassian-python-api`` for page retrieval,
``beautifulsoup4`` for HTML stripping, and ``langchain_text_splitters`` for chunking.

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
from datetime import datetime
from pathlib import Path
from typing import Any

from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100


def _make_chunk_id(source_id: str, chunk_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:None:chunk_index)[:16]."""
    raw = f"{source_id}:None:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
            from atlassian import Confluence
        except ImportError as exc:
            raise ImportError(
                "atlassian-python-api is required for ConfluenceConnector. "
                "Install it with: pip install 'nlqueries-core[wiki]'"
            ) from exc

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for ConfluenceConnector. "
                "Install it with: pip install 'nlqueries-core[wiki]'"
            ) from exc

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "langchain-text-splitters is required for ConfluenceConnector. "
                "Install it with: pip install 'nlqueries-core[wiki]'"
            ) from exc

        space_key = str(source)
        client = Confluence(
            url=self._base_url,
            username=self._username,
            password=self._api_token,
            cloud=True,
        )

        # Build CQL query; optionally restrict to pages modified after `since`.
        cql = f'space = "{space_key}" AND type = page ORDER BY lastModified DESC'
        if since is not None:
            since_str = since.strftime("%Y-%m-%d %H:%M")
            cql = (
                f'space = "{space_key}" AND type = page'
                f' AND lastModified > "{since_str}"'
                f" ORDER BY lastModified DESC"
            )

        search_results: dict[str, Any] = client.cql(cql, limit=200)
        page_results: list[dict[str, Any]] = search_results.get("results", [])

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
