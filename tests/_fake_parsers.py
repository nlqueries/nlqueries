"""Document connectors that misbehave, for tests/test_document_isolation.py.

In a module of their own because the child process imports the connector it
runs by name, so it must be importable, not defined inside a test.
"""

from __future__ import annotations

import time
from pathlib import Path

from nlqueries.document_connectors import _limits
from nlqueries.document_connectors._isolation import IsolatedIngest
from nlqueries.document_connectors._limits import DocumentTooComplexError
from nlqueries.document_connectors.base import DocumentChunk, DocumentConnector


def _chunk(source_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{source_id}-0",
        source_id=source_id,
        source_name="fake",
        page_number=None,
        chunk_index=0,
        text=text,
        metadata={},
    )


class _Fake(IsolatedIngest, DocumentConnector):
    def supports(self, source: str | Path) -> bool:
        return True


class Returns(_Fake):
    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        return [_chunk(source_id, f"parsed {Path(source).name}")]


class Hangs(_Fake):
    """Stuck inside one unit of work, where the in-process budget never looks."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        time.sleep(600)
        return []


class Allocates(_Fake):
    """Holds on to 1 GiB, in 64 MiB pieces, and returns if nothing stopped it.

    Bounded, so that a limit that is not applied fails the test rather than
    growing until the machine's own limit raises the same MemoryError.
    """

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        held = [bytearray(64 * 1024 * 1024) for _ in range(16)]
        return [_chunk(source_id, f"held {len(held)} pieces")]


class Spins(_Fake):
    """Burns CPU without returning to a point where the budget is checked."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        n = 0
        while True:
            n += 1


class LxmlRunsOutOfMemory(_Fake):
    """What python-docx raises when libxml2 cannot allocate: not MemoryError."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        from lxml import etree  # noqa: PLC0415

        raise etree.XMLSyntaxError("unknown error", etree.ErrorTypes.ERR_NO_MEMORY, 0, 0)


class LxmlRejectsTheXml(_Fake):
    """A malformed document, which is not a memory refusal."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        from lxml import etree  # noqa: PLC0415

        raise etree.XMLSyntaxError("Document is empty", etree.ErrorTypes.ERR_DOCUMENT_EMPTY, 1, 1)


class Refuses(_Fake):
    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        raise DocumentTooComplexError("fake.docx expands to 900 MiB, over the 128 MiB limit.")


class _Unpicklable(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.callback = lambda: None


class _NeedsArguments(Exception):
    """Pickles, but does not unpickle: its __init__ wants two arguments and
    Exception was given one. subprocess.CalledProcessError has this shape."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path


class RaisesWhatDoesNotUnpickle(_Fake):
    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        raise _NeedsArguments("/fonts/x.afm", "broken glyph table")


class RaisesUnpicklable(_Fake):
    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        raise _Unpicklable("the parser broke")


class Prints(_Fake):
    """Writes to stdout, which carries the result back to the parent.

    Enough to fill the stream's buffer, so it is written out before the result
    rather than after it.
    """

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        print("a parser's chatter " * 10_000)
        return [_chunk(source_id, "after the chatter")]


class WritesToTheDescriptor(_Fake):
    """Writes to descriptor 1 directly, as a C library or a child process would."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        os.write(1, b"written below Python's sys.stdout\n")
        assert sys.__stdout__ is not None
        sys.__stdout__.write("through sys.__stdout__\n")
        sys.__stdout__.flush()
        subprocess.run([sys.executable, "-c", "print('from a grandchild')"], check=True)
        return [_chunk(source_id, "after the noise")]


class ChattersThenDies(_Fake):
    """Logs far more than the parent keeps, then exits without a result."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        import os  # noqa: PLC0415

        for n in range(20_000):
            os.write(2, b"WARNING: unrecognised operator %d\n" % n)
        os.write(2, b"the last thing it said\n")
        os._exit(3)


class LeavesAGrandchild(_Fake):
    """Starts a process that outlives it holding stderr open; returns its pid."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        return [_chunk(source_id, str(grandchild.pid))]


class ReturnsMany(_Fake):
    """A result far larger than a pipe's buffer, as any real document's is."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"{source_id}-{n}",
                source_id=source_id,
                source_name="fake",
                page_number=None,
                chunk_index=n,
                text=f"chunk {n} " + "of a long document " * 5,
                metadata={},
            )
            for n in range(5000)
        ]


class KilledBySystem(_Fake):
    """Dies as the kernel's out-of-memory killer leaves a process: SIGKILL, no traceback."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        import os  # noqa: PLC0415
        import signal  # noqa: PLC0415

        os.kill(os.getpid(), signal.SIGKILL)  # type: ignore[attr-defined]
        return []


class ReportsLimits(_Fake):
    """Returns the limits it ran under."""

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        return [_chunk(source_id, f"{_limits.MAX_ROWS} {_limits.MAX_SECONDS}")]
