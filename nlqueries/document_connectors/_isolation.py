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

import contextlib
import os
import pickle  # noqa: S403 - both ends are this package; nothing else writes these pipes
import signal
import subprocess
import sys
import threading
import traceback
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

#: How much of the end of the child's stderr is kept. What a parser logs there
#: is unbounded and depends on the document -- pdfminer warns once per
#: unrecognised operator -- and only the end is ever read: the last line, or the
#: traceback of a child that died. Held whole, it would hand the parent back
#: the memory the child's limit takes away.
STDERR_TAIL_BYTES = 64 * 1024

#: The exit code of a child whose own watchdog ended it: its parent was gone,
#: or failed to stop it at the deadline. Read as a timeout.
WATCHDOG_EXIT_CODE = 124

#: How long to wait, once the child has exited, for its output to finish
#: arriving. stdout ends with the child; stderr can be held open by a process
#: the parser started, and is not waited on past this.
_DRAIN_SECONDS = 5.0


class IsolatedIngest:
    """Gives a document connector an ``ingest`` that parses in a child process.

    The connector's own parsing is :meth:`_ingest_here`. With
    :data:`~nlqueries.config.DOCUMENT_PARSE_ISOLATION` off it is called in this
    process, as before.
    """

    def _ingest_here(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        raise NotImplementedError

    def _refuse_before_parsing(self, source: Path) -> None:
        """The connector's checks that read no more than a file's size or a zip's
        directory. Run here as well as in :meth:`_ingest_here`, so a document they
        refuse costs a ``stat`` rather than a child process."""

    def ingest(self, source: str | Path, source_id: str) -> list[DocumentChunk]:
        if not config.DOCUMENT_PARSE_ISOLATION:
            return self._ingest_here(source, source_id)
        self._refuse_before_parsing(Path(source))
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
    # The child holds the same deadline itself, so it ends even when this
    # process dies without stopping it -- killed outright, or by a task's hard
    # time limit. It starts counting later than this process does, so the
    # deadline here is the one that normally fires.
    payload["deadline_seconds"] = deadline
    # `-P`: the child imports the installed package, never a module of the same
    # name in the working directory. On POSIX it leads a session of its own, so
    # that what the parser starts can be ended with it (`_end_descendants`); on
    # Windows its Job Object does that.
    child = subprocess.Popen(  # noqa: S603 - our own interpreter and module
        [sys.executable, "-P", "-m", "nlqueries.document_connectors._child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=sys.platform != "win32",
    )
    output = _Output(child)
    try:
        try:
            _send(child, pickle.dumps(payload))
            child.wait(timeout=deadline)
            # Before collecting: a process the parser started can hold stderr
            # open, and the limits that bound the child are per process.
            _end_descendants(child)
        except subprocess.TimeoutExpired:
            _stop(child)
            raise DocumentExtractionTimeout(
                f"Extracting {name} did not finish within {deadline:g}s and was stopped."
            ) from None
        except BaseException:
            # The caller was stopped rather than the parse: a task's time limit,
            # a worker shutting down, an interrupt. The child would otherwise
            # outlive it holding the file and its memory, and on Windows with no
            # limit on its time at all, since the deadline above was the only one.
            _stop(child)
            raise
        out, err = output.collected()
    finally:
        output.close()

    if out:
        try:
            kind, value = pickle.loads(out)  # noqa: S301 - written by our own child
        except Exception as exc:  # noqa: BLE001 - reported as what it is, below
            if child.returncode == 0:
                # The child finished and said something this could not read:
                # not a death, and not to be reported as one.
                raise RuntimeError(
                    f"Extracting {name} failed: the parser's result could not be read "
                    f"({type(exc).__name__}: {exc})."
                ) from None
            # Cut short, as a child killed mid-write leaves it: its exit says why.
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


class _Output:
    """The child's stdout, whole, and the end of its stderr, read as they arrive.

    In place of ``communicate``, which holds all of both: stdout carries only
    the result, but stderr carries whatever the parser and its libraries write.
    """

    def __init__(self, child: subprocess.Popen[bytes]) -> None:
        self._child = child
        self._out = bytearray()
        self._err = bytearray()
        self._out_reader = threading.Thread(target=self._read_out, daemon=True)
        self._err_reader = threading.Thread(target=self._read_err, daemon=True)
        self._readers = [self._out_reader, self._err_reader]
        for reader in self._readers:
            reader.start()

    def _read_out(self) -> None:
        if self._child.stdout is not None:
            with contextlib.suppress(OSError, ValueError):
                self._out += self._child.stdout.read()

    def _read_err(self) -> None:
        stream = self._child.stderr
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            while block := os.read(stream.fileno(), 65536):
                self._err += block
                if len(self._err) > STDERR_TAIL_BYTES:
                    del self._err[:-STDERR_TAIL_BYTES]

    def collected(self) -> tuple[bytes, bytes]:
        for reader in self._readers:
            reader.join(_DRAIN_SECONDS)
        return bytes(self._out), bytes(self._err)

    def close(self) -> None:
        """Close the pipes, each once its reader has finished with it.

        A pipe still being read is left open: a process the parser started can
        keep stderr open after the child exits, and on Windows closing a pipe
        another thread is reading blocks until that read returns.
        """
        if self._child.stdin is not None:
            with contextlib.suppress(OSError):
                self._child.stdin.close()
        for stream, reader in (
            (self._child.stdout, self._out_reader),
            (self._child.stderr, self._err_reader),
        ):
            reader.join(_DRAIN_SECONDS)
            if stream is not None and not reader.is_alive():
                with contextlib.suppress(OSError):
                    stream.close()


def _send(child: subprocess.Popen[bytes], payload: bytes) -> None:
    """Hand the child its instructions. A child that has already gone is
    reported by its exit, not by the write that found it gone."""
    if child.stdin is None:
        return
    with contextlib.suppress(OSError):
        child.stdin.write(payload)
        child.stdin.close()


def _stop(child: subprocess.Popen[bytes]) -> None:
    """Kill *child*, and whatever it started, and reap it."""
    child.kill()
    _end_descendants(child)
    child.wait()


def _end_descendants(child: subprocess.Popen[bytes]) -> None:
    """Kill the processes *child* started, which its limits do not cover.

    On POSIX the child leads its own process group, so that is killed whole;
    one that has already emptied is no error. On Windows the child's Job
    Object is set to kill its members when the child's handle to it closes,
    which happens when the child ends, so there is nothing to do here.

    The order matters on POSIX: the group id is the child's pid, which is
    free for reuse once the child is reaped. `_stop` kills the group before
    reaping. After a normal exit `wait` has already reaped it, so a host that
    cycled through its whole pid space in between could have handed the id to
    another group leader; improbable, and accepted rather than polled for.
    """
    if sys.platform == "win32":
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGKILL)


def _death(name: str, returncode: int | None, stderr: bytes) -> Exception:
    """What a child that exited without reporting a result ran into."""
    text = stderr.decode("utf-8", "replace")
    if "MemoryError" in text:
        return _memory_refusal(name)
    if returncode == WATCHDOG_EXIT_CODE:
        return DocumentExtractionTimeout(
            f"Extracting {name} did not finish within its deadline and stopped itself."
        )
    cpu_signal = getattr(signal, "SIGXCPU", None)
    if cpu_signal is not None and returncode == -int(cpu_signal):
        return DocumentExtractionTimeout(
            f"Extracting {name} used up its {config.DOCUMENT_PARSE_CPU_SECONDS}s of CPU time."
        )
    # SIGKILL, with nothing on stderr: the kernel's out-of-memory killer, which
    # is how a container's memory limit is reached -- a cgroup, not an rlimit,
    # so the child never sees MemoryError. Nothing else here sends it: the
    # deadline and an interrupted caller both raise before this is reached.
    # A timeout rather than a final refusal: the limit is the whole container's
    # and the kernel kills the largest process in it, so the document is not
    # necessarily the cause, and a caller may retry (see that class).
    kill_signal = getattr(signal, "SIGKILL", None)
    if kill_signal is not None and returncode == -int(kill_signal):
        return DocumentExtractionTimeout(
            f"Extracting {name} was killed by the system, which a container's memory "
            "limit does when the container as a whole reaches it."
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
        # Pickling keeps neither the traceback nor the cause, so the parser's
        # stack goes with the exception as a note, where whoever logs it sees it.
        trace = traceback.format_exc()
        exc.add_note(f"Raised in the document-parsing child:\n{trace}")
        try:
            # Both ways: a class whose __init__ takes arguments it does not pass
            # on to Exception pickles, and then fails to unpickle in the parent.
            pickle.loads(pickle.dumps(exc))  # noqa: S301 - our own bytes
        except Exception:  # noqa: BLE001 - an exception that cannot cross the pipe
            return "unpicklable", f"{type(exc).__name__}: {exc}\n{trace}"
        return "raised", exc
