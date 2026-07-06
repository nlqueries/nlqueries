"""
nlqueries.document_connectors.chunker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dependency-free recursive-character text chunker used by the document
connectors (PDF, Word, Notion, Confluence) in place of
``langchain_text_splitters.RecursiveCharacterTextSplitter``.
"""

from __future__ import annotations

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
_JOINER = "\n"


def _split_recursive(fragment: str, chunk_size: int, sep_index: int) -> list[str]:
    if not fragment:
        return []
    if len(fragment) <= chunk_size:
        return [fragment]
    if sep_index >= len(_SEPARATORS):
        # Separator cascade exhausted: hard-split to guarantee every leaf <= chunk_size.
        return [fragment[i : i + chunk_size] for i in range(0, len(fragment), chunk_size)]

    separator = _SEPARATORS[sep_index]
    raw_parts = fragment.split(separator)
    if len(raw_parts) <= 1:
        # Separator genuinely absent (checked pre-filter, since filtering empty
        # strings first would misclassify e.g. "\n\nHello" as "no separator").
        return _split_recursive(fragment, chunk_size, sep_index + 1)

    parts = [p for p in raw_parts if p]
    leaves: list[str] = []
    for part in parts:
        leaves.extend(_split_recursive(part, chunk_size, sep_index + 1))
    return leaves


def _merge_leaves(leaves: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if not leaves:
        return []

    effective_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    chunks: list[str] = [leaves[0]]

    for piece in leaves[1:]:
        current = chunks[-1]
        if len(current) + len(_JOINER) + len(piece) <= chunk_size:
            chunks[-1] = f"{current}{_JOINER}{piece}"
            continue

        # Start a new chunk, seeded with as much overlap as fits without
        # exceeding chunk_size -- overlap degrades toward zero rather than
        # ever violating the size cap.
        available = chunk_size - len(piece) - len(_JOINER)
        usable_overlap = max(0, min(effective_overlap, available))
        overlap_text = current[-usable_overlap:] if usable_overlap else ""
        chunks.append(f"{overlap_text}{_JOINER}{piece}" if overlap_text else piece)

    return chunks


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Recursively splits on paragraph, line, sentence, then word boundaries
    (falling back to a hard character split if none apply), then greedily
    merges adjacent pieces back together up to ``chunk_size``, seeding each
    new chunk with up to ``chunk_overlap`` trailing characters of the
    previous one.

    Args:
        text: The text to split.
        chunk_size: Maximum length of each returned chunk.
        chunk_overlap: Number of trailing characters from the previous chunk
            to prepend to the next one. Clamped so no chunk ever exceeds
            ``chunk_size``.

    Returns:
        A list of non-empty chunk strings, each at most ``chunk_size``
        characters long. Empty or whitespace-only input returns ``[]``.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or ``chunk_overlap``
            is negative.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if not text or not text.strip():
        return []

    leaves = [p.strip() for p in _split_recursive(text, chunk_size, 0)]
    leaves = [p for p in leaves if p]
    if not leaves:
        return []

    return _merge_leaves(leaves, chunk_size, chunk_overlap)


class RecursiveCharacterTextSplitter:
    """Drop-in, dependency-free replacement for
    ``langchain_text_splitters.RecursiveCharacterTextSplitter``."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        return split_text(text, self._chunk_size, self._chunk_overlap)
