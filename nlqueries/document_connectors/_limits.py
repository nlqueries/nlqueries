"""
nlqueries.document_connectors._limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bounds on what document extraction may consume.

Part of security-review finding 2.8. An upload is size-limited before it is
stored, and the request body is capped before it is parsed -- but neither bounds
what comes *out* of a parser. `.xlsx` and `.docx` are zip archives, and the
number that matters for memory is the expanded size, not the size on disk.

**Measured, not assumed.** On this machine, with ordinary content:

===========================  ==========  ==============  =====
input                        on disk     expands to      ratio
===========================  ==========  ==============  =====
200k rows x 10 cols          5.2 MiB     108.8 MiB       20.9x
300k rows of repeated text   7.1 MiB     231.2 MiB       33.0x
===========================  ==========  ==============  =====

Neither is crafted; both are things a user could produce in Excel. At 33x, a
file at the 20 MiB upload limit expands to roughly 654 MiB in a bulk worker.

A deliberate zip bomb does far better than 33x, which is the point: the ratio is
attacker-chosen and the size on disk says nothing about it.

**The check reads the central directory, and decompresses nothing.** Every zip
entry carries its uncompressed size in the directory at the end of the archive,
so the total is known before a parser touches the file. That figure is
self-reported and a malformed archive can lie -- which is why it is a cheap first
gate rather than the only one, and why the count limits in the connectors
(applied while reading) are the ones that bound a liar.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from nlqueries.config import (
    MAX_CONFLUENCE_BYTES,
    MAX_CONFLUENCE_PAGES,
    MAX_DOCUMENT_EXPANDED_BYTES,
    MAX_DOCUMENT_EXPANSION_RATIO,
    MAX_DOCUMENT_ROWS,
    MAX_EXTRACTION_SECONDS,
    MAX_NOTION_BLOCKS,
    MAX_NOTION_BYTES,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
)


class DocumentTooComplexError(Exception):
    """Raised when a document would cost more to extract than the limits allow.

    Deliberately not ``ValueError`` or ``MemoryError``: callers distinguish "this
    file is refused" from "this file is broken", and the enterprise ingestion
    task reports the two differently to the user.

    Raised directly only by the *deterministic* limits -- the expansion gate, the
    row cap, the PDF size and page caps and the wiki read caps. Each reads a
    property of the source, so a caller that retries gets the same answer and has
    spent a download and a parse to learn nothing.
    :class:`DocumentExtractionTimeout` is the partial exception to that -- read
    its docstring before retrying one, since it is not always transient -- and is
    a subclass precisely so an existing ``except DocumentTooComplexError`` keeps
    catching both.
    """


class DocumentExtractionTimeout(DocumentTooComplexError):
    """Raised when extraction ran out of wall-clock budget.

    Split from its parent because it is the one refusal here that *may* not be a
    property of the file. The budget measures elapsed time, so it is partly a
    function of what else the machine was doing: a document that extracts in
    100s on an idle worker can pass the 120s default on a busy one and fail it
    on the next. A caller that treats every ``DocumentTooComplexError`` as final
    therefore permanently rejects files that are fine, and tells their owner to
    split something that did not need splitting.

    **It may equally be the file, and that is the case this class exists for.**
    The budget is the only bound on what the expansion gate cannot reach: a
    ``.pdf`` is not a zip, Word has no count cap, and an archive that misstated
    its directory gets past the gate by construction. Such a document times out
    on every attempt.

    So a retry must be **bounded and counted per document**, not merely allowed.
    Retry this rather than its parent, but with a limit, and record a document
    that keeps timing out as finally refused -- an unbounded retry repeats a
    download and up to ``MAX_EXTRACTION_SECONDS`` of parsing for exactly the
    files these limits were added to contain, which makes the defence an
    amplifier. ``enterprise``'s ingestion task bounds it with
    ``INGEST_MAX_RETRIES``.
    """


#: Re-exported from :mod:`nlqueries.config` rather than read from the environment
#: here. ``config`` is where ``load_dotenv()`` runs and, by its own docstring, the
#: single source of truth for settings -- and nothing in this package imports it
#: otherwise, so reading ``os.getenv`` at import time meant that on any path where
#: ``config`` had not already been imported the ``.env`` file had not been read and
#: an operator's override was silently ignored.
#:
#: Bound as module attributes so they remain the names this module's callers and
#: tests refer to, and so a connector reads ``_limits.<NAME>`` at call time.
MAX_EXPANDED_BYTES: int = MAX_DOCUMENT_EXPANDED_BYTES
MAX_EXPANSION_RATIO: int = MAX_DOCUMENT_EXPANSION_RATIO

#: Smallest expanded size at which one archive member is held to
#: ``MAX_EXPANSION_RATIO`` on its own. The risk is the absolute expanded size,
#: not the ratio: a small repetitive member can't exhaust memory however well it
#: compresses, and ordinary writers produce such members (20,000 identical Word
#: paragraphs give a ``document.xml`` of about 4 MB at about 250x). The
#: whole-archive ratio and ``MAX_EXPANDED_BYTES`` apply to every archive
#: regardless.
MEMBER_RATIO_FLOOR_BYTES: int = 4 * 1024 * 1024
MAX_ROWS: int = MAX_DOCUMENT_ROWS
MAX_SECONDS: float = MAX_EXTRACTION_SECONDS
PDF_BYTES: int = MAX_PDF_BYTES
PDF_PAGES: int = MAX_PDF_PAGES
CONFLUENCE_PAGES: int = MAX_CONFLUENCE_PAGES
CONFLUENCE_BYTES: int = MAX_CONFLUENCE_BYTES
NOTION_BLOCKS: int = MAX_NOTION_BLOCKS
NOTION_BYTES: int = MAX_NOTION_BYTES


class ExtractionBudget:
    """A wall-clock deadline for extracting one document.

    The expansion gate reads what an archive declares about itself. This measures
    what extraction actually costs, so it bounds a document whose directory lied,
    a `.pdf` (which is not an archive and never passed through that gate at all),
    and the case no size predicts -- a small file that is simply expensive to
    parse.

    One budget across every connector, and the general bound on cost.
    Per-format counts are guesses about cost; a clock measures it. A 3,000-page
    PDF of scanned images and a 30-page one of dense tables cost very
    differently, and no page number tells them apart. The count caps -- PDF size
    and pages, spreadsheet rows -- refuse what is known to be too large up
    front; they do not stand in for this.

    :meth:`check` is called *between* units of work -- a page, every 1,000 Word
    paragraphs, a section, a batch of rows -- so the budget bounds a long document rather than
    interrupting one slow page part-way. A parser that hangs inside a single unit
    is not something this can stop, and is not what it claims to.
    """

    def __init__(self, seconds: float | None = None, *, name: str = "document") -> None:
        self._limit = MAX_SECONDS if seconds is None else seconds
        self._started = time.monotonic()
        self._name = name

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def check(self, progress: str) -> None:
        """Raise if the budget is spent. *progress* names how far it got."""
        if self.elapsed > self._limit:
            raise DocumentExtractionTimeout(
                f"Extracting {self._name} passed the {self._limit:g}s budget "
                f"after {progress} ({self.elapsed:.1f}s)."
            )


def _size(n: int) -> str:
    """*n* bytes for a message, in a unit that never rounds a non-zero size to zero."""
    if n < 1024:
        return f"{n:,} bytes"
    if n < 1024 * 1024:
        return f"{n / 1024:,.1f} KiB"
    return f"{n / (1024 * 1024):,.1f} MiB"


def _size_over(size: int, limit: int) -> tuple[str, str]:
    """*size* and the *limit* it exceeds, formatted so they read as different.

    :func:`_size` rounds to a tenth of the unit, so a size just over its limit
    can format the same as the limit ("100.0 MiB, over the 100.0 MiB limit").
    When it would, both are given in exact bytes instead.
    """
    shown, cap = _size(size), _size(limit)
    if shown == cap:
        return f"{size:,} bytes", f"{limit:,} bytes"
    return shown, cap


def check_archive_expansion(source: str | Path) -> None:
    """Refuse a zip-based document that would expand beyond the limits.

    A no-op for anything that is not a zip archive -- `.pdf` is not one, and a
    corrupt file is left for the parser to report, since "not a valid xlsx" is a
    better message than anything this function could invent.
    """
    path = Path(source)
    if not zipfile.is_zipfile(path):
        return

    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        expanded = sum(entry.file_size for entry in entries)
        compressed = sum(entry.compress_size for entry in entries)

    if expanded > MAX_EXPANDED_BYTES:
        shown, cap = _size_over(expanded, MAX_EXPANDED_BYTES)
        raise DocumentTooComplexError(
            f"{path.name} expands to {shown}, over the {cap} limit for a single document."
        )

    # Guarded against a zero denominator: an archive of stored (uncompressed)
    # entries reports compress_size == file_size, and an empty one reports both
    # as zero. Neither is an expansion.
    if compressed > 0:
        ratio = expanded / compressed
        if ratio > MAX_EXPANSION_RATIO:
            raise DocumentTooComplexError(
                f"{path.name} expands {ratio:.0f}x ({_size(compressed)} to "
                f"{_size(expanded)}), over the {MAX_EXPANSION_RATIO}x limit."
            )

    # Each member as well as the whole. The whole-archive ratio can be diluted:
    # one member that expands 1000x beside a large stored (incompressible) one
    # averages out under the limit, yet the parser still inflates the bomb in
    # full. Only members of at least MEMBER_RATIO_FLOOR_BYTES expanded, since a
    # smaller one is cheap to inflate whatever its ratio. Same zero-denominator
    # guard as above, for the same reasons.
    for entry in entries:
        if entry.file_size >= MEMBER_RATIO_FLOOR_BYTES and entry.compress_size > 0:
            member_ratio = entry.file_size / entry.compress_size
            if member_ratio > MAX_EXPANSION_RATIO:
                raise DocumentTooComplexError(
                    f"{path.name}: member {entry.filename!r} expands {member_ratio:.0f}x "
                    f"({_size(entry.compress_size)} to {_size(entry.file_size)}), over the "
                    f"{MAX_EXPANSION_RATIO}x limit."
                )
