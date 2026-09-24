"""
nlqueries.document_connectors.text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Plain-text (.txt) and Markdown (.md, .markdown) document connectors.

No third-party dependency: the file is read as UTF-8 and chunked with the
built-in splitter, so these work on a base install without the ``docs`` extra.

**Markdown** is split into sections at level-1 and level-2 headings -- the
same boundary ``WordConnector`` uses for Heading 1 / Heading 2 -- and each
section's heading travels in the chunk metadata. Both heading styles count:
ATX (``# Title``, ``## Section``) and setext, a paragraph underlined with
``===`` or ``---``, which READMEs still use. Details a naive split gets wrong:

* ``---`` after a blank line is a thematic break, not a heading, and under a
  list item it is a break too; only paragraph text directly above makes it an
  underline.
* A YAML front-matter block at the top of the file (``---``, ``key: value``
  lines, ``---``) is skipped. Its closing ``---`` would otherwise read as a
  setext underline and make the block every chunk's heading.

* A ``#`` line inside a fenced code block (```` ``` ```` or ``~~~``) is a shell
  or Python comment, not a heading. Fences are tracked, and nothing inside one
  starts a section.
* Deeper headings (``###`` and below) stay in their section's text. Splitting
  on every level produces chunks too small to carry an answer.

**Plain text** has no structure to follow, so it is one section, split by the
recursive splitter on paragraphs, then lines, then words.

Text that is not UTF-8 is refused with ``UnicodeDecodeError`` rather than
decoded with a guessed encoding: a wrong guess ingests mojibake that embeds and
retrieves as if it were content, and the person uploading it never finds out.
A UTF-8 byte-order mark is accepted and dropped.

A file larger than the per-document limit (``MAX_EXPANDED_BYTES``, the cap a
zip-based document's expansion is held to) is refused with
``DocumentTooComplexError`` before it is read, since nothing can interrupt the
read once it starts.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from nlqueries.document_connectors import _limits
from nlqueries.document_connectors._limits import DocumentTooComplexError, ExtractionBudget
from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector
from nlqueries.document_connectors.chunker import RecursiveCharacterTextSplitter

_SPLIT_THRESHOLD = 1_200
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

#: ATX headings at level 1 or 2. Up to three leading spaces is still a heading
#: in CommonMark; four is an indented code block.
_SECTION_HEADING = re.compile(r"^ {0,3}(#{1,2})[ \t]+(.+?)[ \t]*#*[ \t]*$")
#: A setext underline: `===` makes the paragraph above it a level-1 heading,
#: `---` a level-2 one. Only directly under paragraph text -- after a blank
#: line, `---` is a thematic break and stays text.
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
#: A thematic break: three or more `-`, `*` or `_`, optionally spaced. It is a
#: block of its own, so a paragraph above an underline stops at one.
_THEMATIC_BREAK = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
#: A list item. `- item` followed by `---` is a list and a thematic break, not
#: a heading.
_LIST_ITEM = re.compile(r"^ {0,3}([-*+]|\d{1,9}[.)])([ \t]|$)")
#: A code fence opener or closer: three or more backticks or tildes.
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_TEXT_SUFFIXES = frozenset({".txt"})


def _make_chunk_id(source_id: str, chunk_index: int) -> str:
    """Deterministic 16-char hex ID: sha256(source_id:None:chunk_index)[:16]."""
    raw = f"{source_id}:None:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _read_utf8(path: Path) -> str:
    """The file as text. ``utf-8-sig`` drops a byte-order mark if there is one."""
    return path.read_bytes().decode("utf-8-sig")


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split *text* into ``(heading, body)`` pairs at level-1/2 headings.

    Text before the first heading is its own section, headed ``"untitled"``.
    Sections with no body are dropped; a heading followed directly by another
    heading has nothing to embed.
    """
    sections: list[tuple[str, str]] = []
    heading = "untitled"
    lines: list[str] = []
    # The open fence's character and length. CommonMark closes a fence only
    # with the same character, at least as many times -- a ```` fence is not
    # closed by ```, so a sample showing a fence inside a fence stays inside.
    fence: tuple[str, int] | None = None

    for line in _without_front_matter(text.splitlines()):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
            lines.append(line)
            continue

        heading_match = None if fence is not None else _SECTION_HEADING.match(line)
        if heading_match:
            body = "\n".join(lines).strip()
            if body:
                sections.append((heading, body))
            heading = heading_match.group(2).strip() or "untitled"
            lines = []
            continue

        setext = _setext_paragraph(lines) if fence is None and _SETEXT_UNDERLINE.match(line) else 0
        if setext:
            title = " ".join(ln.strip() for ln in lines[-setext:])
            body = "\n".join(lines[:-setext]).strip()
            if body:
                sections.append((heading, body))
            heading = title
            lines = []
        else:
            lines.append(line)

    body = "\n".join(lines).strip()
    if body:
        sections.append((heading, body))
    return sections


def _without_front_matter(lines: list[str]) -> list[str]:
    """*lines* without a YAML front-matter block, if the file opens with one.

    Obsidian, Hugo and Jekyll notes start with ``---``, a few ``key: value``
    lines, then ``---`` (or ``...``). Read as Markdown, that closing ``---``
    sits directly under text and is a setext underline, so the whole block
    became the heading of every chunk that followed. Only a block that starts on
    the very first line is front matter; a ``---`` anywhere else is Markdown's.
    An opening ``---`` that is never closed is left alone.
    """
    if not lines or lines[0].strip() != "---":
        return lines
    for end in range(1, len(lines)):
        if lines[end].strip() in ("---", "..."):
            return lines[end + 1 :]
    return lines


def _setext_paragraph(lines: list[str]) -> int:
    """How many trailing *lines* form the paragraph a setext underline titles.

    The paragraph is the run of non-blank lines directly above the underline,
    and in CommonMark all of it becomes the heading. Zero when there is none --
    a blank line above makes `---` a thematic break -- or when that run holds
    something that is not paragraph text: a list item, a fence line, or a line
    indented four spaces (code). A thematic break above ends the paragraph
    without being part of it, as a blank line does.
    """
    count = 0
    for line in reversed(lines):
        if not line.strip() or _THEMATIC_BREAK.match(line) or _SETEXT_UNDERLINE.match(line):
            break
        if _LIST_ITEM.match(line) or _FENCE.match(line) or line.startswith("    "):
            return 0
        count += 1
    return count


class _SectionedTextConnector(DocumentConnector):
    """Shared chunking: sections in, deterministic chunks out."""

    _connector_name: str
    _suffixes: frozenset[str]

    def _sections(self, text: str) -> list[tuple[str, str]]:
        raise NotImplementedError

    def ingest(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        source_path = Path(source)
        # Before the read, because the read is the expensive part: the whole
        # file, then its decoded text, then a per-line list while sectioning,
        # and the extraction budget cannot interrupt any of it. A plain file's
        # size on disk is its expanded size, so it is held to the same limit a
        # zip-based document's declared expansion is.
        size = source_path.stat().st_size
        # Read from `_limits` at call time, as `check_archive_expansion` does, so
        # the one knob that tightens every other connector tightens these too.
        limit = _limits.MAX_EXPANDED_BYTES
        if size > limit:
            raise DocumentTooComplexError(
                f"{source_path.name} is {size // (1024 * 1024)} MiB, over the "
                f"{limit // (1024 * 1024)} MiB limit for a single document."
            )
        budget = ExtractionBudget(name=source_path.name)
        text = _read_utf8(source_path)
        sections = self._sections(text)
        budget.check("reading the file")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        chunks: list[DocumentChunk] = []
        index = 0
        for number, (heading, body) in enumerate(sections, start=1):
            budget.check(f"{number - 1} of {len(sections)} sections")
            pieces = splitter.split_text(body) if len(body) > _SPLIT_THRESHOLD else [body]
            for piece in pieces:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_chunk_id(source_id, index),
                        source_id=source_id,
                        source_name=source_path.name,
                        page_number=None,
                        chunk_index=index,
                        text=piece,
                        metadata={
                            "connector": self._connector_name,
                            "file_path": str(source_path),
                            "section_heading": heading,
                        },
                    )
                )
                index += 1
        return chunks

    def supports(self, source: str | Path) -> bool:
        return Path(source).suffix.lower() in self._suffixes


class MarkdownConnector(_SectionedTextConnector):
    """Markdown files, sectioned at ``#`` and ``##`` headings outside code fences."""

    _connector_name = "markdown"
    _suffixes = _MARKDOWN_SUFFIXES

    def _sections(self, text: str) -> list[tuple[str, str]]:
        return markdown_sections(text)


class TextConnector(_SectionedTextConnector):
    """Plain-text files, as one section split on paragraphs, lines, then words."""

    _connector_name = "text"
    _suffixes = _TEXT_SUFFIXES

    def _sections(self, text: str) -> list[tuple[str, str]]:
        body = text.strip()
        return [("untitled", body)] if body else []
