"""
nlqueries.document_connectors._isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PDF, Word and Excel parsing in a child process with time and memory limits.

The parsers are third-party code reading untrusted files. The in-process limits
in :mod:`._limits` check the wall clock only between pages, paragraphs and rows,
so a parser stuck inside one of them, or one that allocates without bound, held
the process that asked -- an API worker or an ingestion task. Here the parse runs
in a child, which is killed when the extraction deadline passes, and which runs
under a memory limit as well: ``RLIMIT_AS`` on POSIX, with ``RLIMIT_CPU`` beside
it, and a Job Object on Windows.

A child process, not a thread, because a thread cannot be stopped; and
``subprocess`` rather than ``multiprocessing``, because a worker that is itself a
daemonic process may not start ``multiprocessing`` children. The child receives
the connector, the file and the limits the parent is using, so a limit an
operator or a test set in the parent applies there too.
"""

from __future__ import annotations

import pickle  # noqa: S403 - both ends are this package; nothing else writes these pipes
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from nlqueries import config
from nlqueries.document_connectors import _limits
from nlqueries.document_connectors._limits import (
    DocumentExtractionTimeout,
    DocumentTooComplexError,
)
from nlqueries.document_connectors.base import DocumentChunk

#: The limits in :mod:`._limits` the child adopts from the parent.
FORWARDED_LIMITS = (
    "MAX_EXPANDED_BYTES",
    "MAX_EXPANSION_RATIO",
    "MEMBER_RATIO_FLOOR_BYTES",
    "MAX_ROWS",
    "MAX_SECONDS",
    "PDF_BYTES",
    "PDF_PAGES",
)

#: Allowance on top of the extraction deadline for the child to start: it is a
#: new interpreter importing the parser before the deadline's clock starts there.
CHILD_START_SECONDS = 15.0


class IsolatedIngest:
    """Gives a document connector an ``ingest`` that parses in a child process.

    The connector's own parsing is :meth:`_ingest_here`. With
    :data:`~nlqueries.config.DOCUMENT_PARSE_ISOLATION` off it is called in this
    process, as before.
    """

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        raise NotImplementedError

    def ingest(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        if not config.DOCUMENT_PARSE_ISOLATION:
            return self._ingest_here(source, source_id)
        return parse_in_child(type(self), source, source_id)


def parse_in_child(connector: type, source: str | Path, source_id: str) -> list[DocumentChunk]:
    """Run ``connector()._ingest_here(source, source_id)`` in a child process.

    Returns its chunks, or raises what it raised. A child still running at the
    extraction deadline (plus :data:`CHILD_START_SECONDS`) is killed and
    :class:`DocumentExtractionTimeout` raised; one stopped by its CPU limit
    raises the same, and one that ran out of memory raises
    :class:`DocumentTooComplexError`.
    """
    name = Path(source).name
    payload = {
        "connector": f"{connector.__module__}:{connector.__qualname__}",
        "source": str(source),
        "source_id": source_id,
        "limits": {limit: getattr(_limits, limit) for limit in FORWARDED_LIMITS},
        "memory_bytes": config.DOCUMENT_PARSE_MEMORY_BYTES,
        "cpu_seconds": config.DOCUMENT_PARSE_CPU_SECONDS,
    }
    deadline = float(_limits.MAX_SECONDS) + CHILD_START_SECONDS
    # `-P`: the child imports the installed package, never a module of the same
    # name in the working directory.
    child = subprocess.Popen(  # noqa: S603 - our own interpreter and module
        [sys.executable, "-P", "-m", "nlqueries.document_connectors._child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = child.communicate(pickle.dumps(payload), timeout=deadline)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise DocumentExtractionTimeout(
            f"Extracting {name} did not finish within {deadline:g}s and was stopped."
        ) from None

    if out:
        try:
            kind, value = pickle.loads(out)  # noqa: S301 - written by our own child
        except Exception:  # noqa: BLE001 - reported below as a failed parse
            kind, value = None, None
        if kind == "chunks":
            return list(value)
        if kind == "raised":
            if isinstance(value, MemoryError):
                raise _memory_refusal(name) from value
            raise value
        if kind == "unpicklable":
            raise RuntimeError(value)
    raise _death(name, child.returncode, err)


def _death(name: str, returncode: int | None, stderr: bytes) -> Exception:
    """What a child that exited without reporting a result ran into."""
    text = stderr.decode("utf-8", "replace")
    if "MemoryError" in text:
        return _memory_refusal(name)
    cpu_signal = getattr(signal, "SIGXCPU", None)
    if cpu_signal is not None and returncode == -int(cpu_signal):
        return DocumentExtractionTimeout(
            f"Extracting {name} used up its {config.DOCUMENT_PARSE_CPU_SECONDS}s of CPU time."
        )
    last = next((line for line in reversed(text.splitlines()) if line.strip()), "")
    return RuntimeError(f"Extracting {name} failed: the parser exited with {returncode}. {last}")


def _memory_refusal(name: str) -> DocumentTooComplexError:
    return DocumentTooComplexError(
        f"Extracting {name} ran out of the "
        f"{config.DOCUMENT_PARSE_MEMORY_BYTES // (1024 * 1024)} MiB memory limit."
    )


def ran_out_of_memory(exc: BaseException) -> bool:
    """Whether *exc*, or an exception it was raised from, is an allocation failing.

    ``MemoryError`` from Python code, and lxml's form of it, which python-docx
    and openpyxl parse through. When libxml2 cannot allocate, lxml raises
    ``XMLSyntaxError("unknown error")`` with ``code`` set to ``ERR_NO_MEMORY``,
    which would otherwise reach the caller as a corrupt document. Measured: a
    ``.docx`` parsed under ``RLIMIT_AS`` fails that way, not with ``MemoryError``.
    """
    # Not imported here: an lxml exception exists only if lxml was loaded.
    etree = sys.modules.get("lxml.etree")
    current: BaseException | None = exc
    for _ in range(10):  # a chain, not a cycle; bounded either way
        if current is None:
            return False
        if isinstance(current, MemoryError):
            return True
        if (
            etree is not None
            and isinstance(current, etree.XMLSyntaxError)
            and current.code == etree.ErrorTypes.ERR_NO_MEMORY
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def child_result(connector_path: str, source: str, source_id: str) -> tuple[str, Any]:
    """Run the parse in this process and describe its outcome for the parent."""
    import importlib  # noqa: PLC0415

    module_name, _, class_name = connector_path.partition(":")
    connector = getattr(importlib.import_module(module_name), class_name)
    try:
        return "chunks", connector()._ingest_here(source, source_id)
    except Exception as exc:  # noqa: BLE001 - handed to the parent to raise
        if ran_out_of_memory(exc):
            return "raised", MemoryError(f"{type(exc).__name__}: {exc}")
        try:
            pickle.dumps(exc)
        except Exception:  # noqa: BLE001 - an exception that cannot cross the pipe
            return "unpicklable", f"{type(exc).__name__}: {exc}"
        return "raised", exc
