"""
nlqueries.document_connectors.notion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Notion document connector using ``notion-client`` for page/block retrieval and
a built-in recursive character chunker.

Chunking strategy: all text blocks from a Notion page are concatenated into a
single body and split with ``RecursiveCharacterTextSplitter(chunk_size=800,
chunk_overlap=100)``.  Supported block types: paragraph, heading_1, heading_2,
heading_3, bulleted_list_item, numbered_list_item, quote.

Notion has no concept of page numbers, so ``page_number`` is always ``None``.

Requires the ``wiki`` optional dependency group:
    pip install "nlqueries-core[wiki]"
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.chunker import RecursiveCharacterTextSplitter

_TEXT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "quote",
    }
)

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100


def _make_chunk_id(source_id: str, chunk_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:None:chunk_index)[:16]."""
    raw = f"{source_id}:None:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_rich_text(rich_text: list[dict[str, Any]]) -> str:
    return "".join(rt.get("plain_text", "") for rt in rich_text)


def _extract_block_text(block: dict[str, Any]) -> str:
    block_type = block.get("type", "")
    if block_type not in _TEXT_BLOCK_TYPES:
        return ""
    block_content: dict[str, Any] = block.get(block_type, {})
    return _extract_rich_text(block_content.get("rich_text", []))


def _extract_page_title(page: dict[str, Any]) -> str:
    properties: dict[str, Any] = page.get("properties", {})
    for prop_data in properties.values():
        if prop_data.get("type") == "title":
            return _extract_rich_text(prop_data.get("title", []))
    return ""


class NotionConnector(DocumentConnector):
    """Extract and chunk text from a Notion page using notion-client.

    All text blocks from the page are concatenated and split with
    ``RecursiveCharacterTextSplitter``.  ``page_number`` is always ``None``
    because Notion has no page concept.
    """

    def __init__(self, api_token: str) -> None:
        self._api_token = api_token

    def ingest(
        self,
        source: str | Path,
        source_id: str,
        *,
        since: datetime | None = None,
    ) -> list[DocumentChunk]:
        """Fetch a Notion page and return text chunks.

        Args:
            source: Notion page ID or database ID.
            source_id: Opaque identifier for this source (used to generate
                       deterministic chunk IDs and for Qdrant filtering).
            since: If provided, skip pages whose ``last_edited_time`` is not
                   strictly after this timestamp (incremental sync).

        Returns:
            Ordered list of :class:`DocumentChunk` objects, or an empty list
            if the page was not modified after ``since``.
        """
        try:
            from notion_client import Client
        except ImportError as exc:
            raise ImportError(
                "notion-client is required for NotionConnector. "
                "Install it with: pip install 'nlqueries-core[wiki]'"
            ) from exc

        client = Client(auth=self._api_token)
        page_id = str(source)

        # Retrieve page metadata (title and last_edited_time).
        page: dict[str, Any] = client.pages.retrieve(page_id=page_id)
        last_edited_time_str: str = page.get("last_edited_time", "")
        page_title = _extract_page_title(page)

        # Incremental sync: skip if the page was not modified after `since`.
        if since is not None and last_edited_time_str:
            last_edited_dt = datetime.fromisoformat(last_edited_time_str.replace("Z", "+00:00"))
            if last_edited_dt <= since:
                return []

        # Fetch all blocks, following Notion's pagination cursor.
        all_text_parts: list[str] = []
        has_more = True
        start_cursor: str | None = None

        while has_more:
            kwargs: dict[str, Any] = {"block_id": page_id}
            if start_cursor:
                kwargs["start_cursor"] = start_cursor

            response: dict[str, Any] = client.blocks.children.list(**kwargs)
            for block in response.get("results", []):
                text = _extract_block_text(block)
                if text:
                    all_text_parts.append(text)

            has_more = bool(response.get("has_more", False))
            start_cursor = response.get("next_cursor")

        full_text = "\n".join(all_text_parts).strip()
        if not full_text:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        text_chunks = splitter.split_text(full_text)

        chunks: list[DocumentChunk] = []
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_id = _make_chunk_id(source_id, chunk_index)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    source_name=page_id,
                    page_number=None,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    metadata={
                        "connector": "notion",
                        "page_id": page_id,
                        "page_title": page_title,
                        "last_edited_time": last_edited_time_str,
                    },
                )
            )

        return chunks

    def supports(self, source: str | Path) -> bool:
        """Notion sources are identified by page ID, not file extension."""
        return True
