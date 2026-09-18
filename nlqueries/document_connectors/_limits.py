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

import zipfile
from pathlib import Path

from nlqueries.config import MAX_DOCUMENT_EXPANDED_BYTES, MAX_DOCUMENT_EXPANSION_RATIO


class DocumentTooComplexError(Exception):
    """Raised when a document would cost more to extract than the limits allow.

    Deliberately not ``ValueError`` or ``MemoryError``: callers distinguish "this
    file is refused" from "this file is broken", and the enterprise ingestion
    task reports the two differently to the user.
    """


#: Re-exported from :mod:`nlqueries.config` rather than read from the environment
#: here. ``config`` is where ``load_dotenv()`` runs and, by its own docstring, the
#: single source of truth for settings -- and nothing in this package imports it
#: otherwise, so reading ``os.getenv`` at import time meant that on any path where
#: ``config`` had not already been imported the ``.env`` file had not been read and
#: an operator's override was silently ignored.
#:
#: Bound as module attributes so they remain the names this module's callers and
#: tests refer to.
MAX_EXPANDED_BYTES: int = MAX_DOCUMENT_EXPANDED_BYTES
MAX_EXPANSION_RATIO: int = MAX_DOCUMENT_EXPANSION_RATIO


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
        raise DocumentTooComplexError(
            f"{path.name} expands to {expanded // (1024 * 1024)} MiB, over the "
            f"{MAX_EXPANDED_BYTES // (1024 * 1024)} MiB limit for a single document."
        )

    # Guarded against a zero denominator: an archive of stored (uncompressed)
    # entries reports compress_size == file_size, and an empty one reports both
    # as zero. Neither is an expansion.
    if compressed > 0:
        ratio = expanded / compressed
        if ratio > MAX_EXPANSION_RATIO:
            raise DocumentTooComplexError(
                f"{path.name} expands {ratio:.0f}x ({compressed // 1024} KiB to "
                f"{expanded // (1024 * 1024)} MiB), over the {MAX_EXPANSION_RATIO}x limit."
            )
