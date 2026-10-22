"""PDF, Word and Excel parsing in a child process with time and memory limits.

The fake connectors in ``tests/_fake_parsers.py`` misbehave in ways the
in-process budget cannot stop -- stuck inside one unit of work, allocating
without bound, spinning -- and these check that the child is stopped and the
caller gets the refusal it already handles.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from nlqueries import config
from nlqueries.document_connectors import (
    DocumentExtractionTimeout,
    DocumentTooComplexError,
    ExcelConnector,
    PdfConnector,
    WordConnector,
    _isolation,
    _limits,
)
from nlqueries.document_connectors._isolation import IsolatedIngest

from tests import _fake_parsers as fake

ROOT = Path(__file__).resolve().parents[1]
_POSIX = sys.platform != "win32"
posix_only = pytest.mark.skipif(not _POSIX, reason="Windows has no CPU-time limit")
rlimits_only = pytest.mark.skipif(not _POSIX, reason="resource limits are POSIX-only")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Isolation on, and the child able to import the fake connectors."""
    monkeypatch.setattr(config, "DOCUMENT_PARSE_ISOLATION", True)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    # A new interpreter, importing the parser, before its own clock starts.
    monkeypatch.setattr(_isolation, "CHILD_START_SECONDS", 10.0)


@pytest.fixture
def source(tmp_path) -> Path:
    path = tmp_path / "doc.bin"
    path.write_bytes(b"not read by the fakes")
    return path


def test_the_three_file_connectors_parse_in_a_child() -> None:
    for connector in (PdfConnector, WordConnector, ExcelConnector):
        assert issubclass(connector, IsolatedIngest)


def test_chunks_come_back_from_the_child(source) -> None:
    (chunk,) = fake.Returns().ingest(source, "s1")
    assert chunk.source_id == "s1"
    assert chunk.text == "parsed doc.bin"


def test_a_parser_stuck_inside_one_unit_is_killed_at_the_deadline(source, monkeypatch) -> None:
    monkeypatch.setattr(_limits, "MAX_SECONDS", 1.0)
    monkeypatch.setattr(_isolation, "CHILD_START_SECONDS", 4.0)
    started = time.monotonic()

    with pytest.raises(DocumentExtractionTimeout, match="did not finish within 5s"):
        fake.Hangs().ingest(source, "s1")

    assert time.monotonic() - started < 30


def test_a_parser_that_allocates_without_bound_is_refused(source, monkeypatch) -> None:
    """``RLIMIT_AS`` on POSIX, a Job Object on Windows."""
    monkeypatch.setattr(config, "DOCUMENT_PARSE_MEMORY_BYTES", 512 * 1024 * 1024)

    with pytest.raises(DocumentTooComplexError, match="512 MiB memory limit"):
        fake.Allocates().ingest(source, "s1")


@posix_only
def test_a_parser_that_spins_is_stopped_by_its_cpu_limit(source, monkeypatch) -> None:
    monkeypatch.setattr(config, "DOCUMENT_PARSE_CPU_SECONDS", 1)

    with pytest.raises(DocumentExtractionTimeout, match="1s of CPU time"):
        fake.Spins().ingest(source, "s1")


def test_lxml_running_out_of_memory_is_a_memory_refusal(source) -> None:
    pytest.importorskip("lxml")
    with pytest.raises(DocumentTooComplexError, match="memory limit"):
        fake.LxmlRunsOutOfMemory().ingest(source, "s1")


def test_a_malformed_document_is_not_mistaken_for_one(source) -> None:
    pytest.importorskip("lxml")
    with pytest.raises(RuntimeError, match="XMLSyntaxError: Document is empty") as caught:
        fake.LxmlRejectsTheXml().ingest(source, "s1")
    assert not isinstance(caught.value, DocumentTooComplexError)


def test_an_allocation_failure_is_found_through_the_exception_chain() -> None:
    try:
        try:
            raise MemoryError
        except MemoryError as exc:
            raise ValueError("the parser wrapped it") from exc
    except ValueError as wrapped:
        assert _isolation.ran_out_of_memory(wrapped)
    assert not _isolation.ran_out_of_memory(ValueError("unrelated"))


@rlimits_only
def test_a_lower_hard_limit_is_kept(monkeypatch) -> None:
    """A container's own ``ulimit`` wins over a larger setting."""
    from nlqueries.document_connectors import _child

    monkeypatch.setattr(_child.resource, "getrlimit", lambda which: (100, 100))
    assert _child._within_hard_limit(_child.resource.RLIMIT_AS, 2**31) == 100
    monkeypatch.setattr(
        _child.resource,
        "getrlimit",
        lambda which: (_child.resource.RLIM_INFINITY, _child.resource.RLIM_INFINITY),
    )
    assert _child._within_hard_limit(_child.resource.RLIMIT_AS, 2**31) == 2**31


def test_an_error_that_does_not_unpickle_is_still_reported(source) -> None:
    """It pickled in the child; the parent could not rebuild it."""
    with pytest.raises(RuntimeError, match="_NeedsArguments: /fonts/x.afm: broken glyph") as caught:
        fake.RaisesWhatDoesNotUnpickle().ingest(source, "s1")
    assert "_ingest_here" in str(caught.value)  # the child's stack came with it


def test_the_parsers_stack_comes_back_with_its_error(source) -> None:
    with pytest.raises(DocumentTooComplexError) as caught:
        fake.Refuses().ingest(source, "s1")
    (note,) = caught.value.__notes__
    assert note.startswith("Raised in the document-parsing child:")
    assert "_fake_parsers.py" in note and "_ingest_here" in note


def test_a_result_that_cannot_be_read_is_not_reported_as_a_death(source, monkeypatch) -> None:
    def unreadable(data):
        raise ValueError("bad pickle data")

    monkeypatch.setattr(_isolation.pickle, "loads", unreadable)
    with pytest.raises(RuntimeError, match="result could not be read .ValueError: bad pickle"):
        fake.Returns().ingest(source, "s1")


def test_a_result_cut_short_is_reported_by_how_the_child_ended(source, monkeypatch) -> None:
    """Killed mid-write, a child leaves part of a result: its exit is the news."""
    real_collected = _isolation._Output.collected

    def cut_short(self):
        _out, err = real_collected(self)
        return b"\x80\x05partial", err

    monkeypatch.setattr(_isolation._Output, "collected", cut_short)
    with pytest.raises(RuntimeError, match="exited with 3"):
        fake.ChattersThenDies().ingest(source, "s1")


def test_a_refusal_reaches_the_caller_as_itself(source) -> None:
    with pytest.raises(DocumentTooComplexError, match="over the 128 MiB limit"):
        fake.Refuses().ingest(source, "s1")


def test_an_error_that_cannot_cross_the_pipe_is_still_reported(source) -> None:
    with pytest.raises(RuntimeError, match="_Unpicklable: the parser broke"):
        fake.RaisesUnpicklable().ingest(source, "s1")


def test_what_a_parser_prints_does_not_corrupt_the_result(source) -> None:
    (chunk,) = fake.Prints().ingest(source, "s1")
    assert chunk.text == "after the chatter"


def test_what_reaches_the_descriptor_does_not_corrupt_the_result(source) -> None:
    (chunk,) = fake.WritesToTheDescriptor().ingest(source, "s1")
    assert chunk.text == "after the noise"


def test_a_caller_stopped_mid_parse_does_not_leave_the_child_running(source, monkeypatch) -> None:
    """A task's time limit or an interrupt reaches the parent, not the child."""
    started: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    class Interrupted(real_popen):  # type: ignore[type-arg, misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

        def wait(self, timeout=None):
            if timeout is not None:  # the wait for the parse, not the reaping
                time.sleep(1)  # the child is parsing by now
                raise KeyboardInterrupt
            return super().wait()

    monkeypatch.setattr(_isolation.subprocess, "Popen", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        fake.Hangs().ingest(source, "s1")

    (child,) = started
    assert child.poll() is not None  # killed and reaped, not left behind
    assert child.stdout is not None and child.stdout.closed


def test_only_the_end_of_the_childs_stderr_is_kept() -> None:
    """What a parser logs is unbounded; the parent holds a bounded tail of it."""
    import os
    import threading
    from types import SimpleNamespace

    read_end, write_end = os.pipe()

    def chatter() -> None:
        with os.fdopen(write_end, "wb") as pipe:
            for n in range(40_000):
                pipe.write(b"WARNING: unrecognised operator %d\n" % n)
            pipe.write(b"the last line\n")

    writer = threading.Thread(target=chatter)
    writer.start()
    with os.fdopen(read_end, "rb") as stderr:
        output = _isolation._Output(SimpleNamespace(stdin=None, stdout=None, stderr=stderr))
        writer.join()
        _out, err = output.collected()

    assert len(err) <= _isolation.STDERR_TAIL_BYTES
    assert err.endswith(b"the last line\n")


def test_a_process_left_holding_stderr_does_not_hold_the_caller(source) -> None:
    import os
    import signal

    started = time.monotonic()
    (chunk,) = fake.LeavesAGrandchild().ingest(source, "s1")
    try:
        # The drain waits are bounded. On POSIX the grandchild inherits
        # descriptor 2 and holds the pipe for its 60 seconds.
        assert time.monotonic() - started < 40
    finally:
        with contextlib.suppress(OSError):  # gone already, or not ours to stop
            os.kill(int(chunk.text), signal.SIGTERM)


def test_a_result_larger_than_the_pipe_comes_back_while_the_child_runs(source, monkeypatch) -> None:
    """The result is read as it is written. Read after the wait instead, the
    child would block on a full pipe until the deadline killed it."""
    monkeypatch.setattr(_limits, "MAX_SECONDS", 5.0)
    started = time.monotonic()

    chunks = fake.ReturnsMany().ingest(source, "s1")

    assert len(chunks) == 5000
    assert chunks[-1].text.startswith("chunk 4999 ")
    assert time.monotonic() - started < 15  # inside the 5 + 10 second deadline


@pytest.mark.skipif(not _POSIX, reason="Windows has no SIGKILL")
def test_a_child_killed_by_the_system_is_a_retryable_timeout(source) -> None:
    """A container's memory limit is reached by the kernel killing the largest
    process, so the document may not be the cause: a timeout, which callers may
    retry, and still a DocumentTooComplexError to those catching the parent."""
    with pytest.raises(DocumentExtractionTimeout, match="killed by the system"):
        fake.KilledBySystem().ingest(source, "s1")


def _gone(pid: int, within: float = 15.0) -> bool:
    from nlqueries.embeddings.embed_server import is_pid_alive

    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.25)
    return False


def test_what_the_parser_starts_ends_when_the_parse_does(source) -> None:
    """The limits are per process: a grandchild would keep its own budget."""
    import os
    import signal

    (chunk,) = fake.LeavesAGrandchild().ingest(source, "s1")
    pid = int(chunk.text)
    try:
        assert _gone(pid)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)


def test_what_the_parser_starts_ends_when_the_parse_is_stopped(source, monkeypatch) -> None:
    import os
    import signal

    monkeypatch.setattr(_limits, "MAX_SECONDS", 2.0)
    monkeypatch.setattr(_isolation, "CHILD_START_SECONDS", 8.0)

    with pytest.raises(DocumentExtractionTimeout):
        fake.StartsAGrandchildThenHangs().ingest(source, "s1")

    pid = int(source.with_suffix(".pid").read_text())
    try:
        assert _gone(pid)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)


def test_the_child_stops_itself_when_its_parent_does_not(source) -> None:
    """Run as a parent that died would leave it: nothing here kills it."""
    import os
    import pickle

    payload = {
        "connector": "tests._fake_parsers:Hangs",
        "source": str(source),
        "source_id": "s1",
        "limits": {},
        "memory_bytes": 1024 * 1024 * 1024,
        "cpu_seconds": 120,
        "deadline_seconds": 3.0,
    }
    started = time.monotonic()
    ran = subprocess.run(
        [sys.executable, "-P", "-m", "nlqueries.document_connectors._child"],
        input=pickle.dumps(payload),
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=60,
        check=False,
    )
    assert ran.returncode == _isolation.WATCHDOG_EXIT_CODE
    assert time.monotonic() - started < 30


def test_a_child_its_watchdog_stopped_is_a_timeout() -> None:
    error = _isolation._death("doc.pdf", _isolation.WATCHDOG_EXIT_CODE, b"")
    assert isinstance(error, DocumentExtractionTimeout)


def test_a_child_that_dies_is_reported_by_its_last_line(source) -> None:
    with pytest.raises(RuntimeError, match="exited with 3.*the last thing it said"):
        fake.ChattersThenDies().ingest(source, "s1")


def test_a_document_refused_by_its_size_starts_no_child(source, monkeypatch) -> None:
    """The checks that read only a size or a zip directory run in the parent first."""
    monkeypatch.setattr(_limits, "PDF_BYTES", 4)

    def no_child(*args, **kwargs):
        raise AssertionError("a child process was started")

    monkeypatch.setattr(_isolation.subprocess, "Popen", no_child)
    with pytest.raises(DocumentTooComplexError, match="over the"):
        PdfConnector().ingest(source, "s1")


def test_a_document_refused_by_its_zip_directory_starts_no_child(tmp_path, monkeypatch) -> None:
    import zipfile

    path = tmp_path / "big.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 200_000)
    monkeypatch.setattr(_limits, "MAX_EXPANDED_BYTES", 1000)

    def no_child(*args, **kwargs):
        raise AssertionError("a child process was started")

    monkeypatch.setattr(_isolation.subprocess, "Popen", no_child)
    for connector in (WordConnector(), ExcelConnector()):
        with pytest.raises(DocumentTooComplexError):
            connector.ingest(path, "s1")


@pytest.mark.parametrize(
    ("value", "isolated"),
    [(None, True), ("false", False), ("0", False), ("off", False), ("true", True), ("flase", True)],
)
def test_isolation_is_on_unless_turned_off_by_name(tmp_path, value, isolated) -> None:
    """Read in a fresh interpreter, since `config` reads it at import. A value
    that is not one of the off spellings leaves it on: it is a safety control."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "NLQ_DOCUMENT_PARSE_ISOLATION"}
    if value is not None:
        env["NLQ_DOCUMENT_PARSE_ISOLATION"] = value
    env["PYTHONPATH"] = str(ROOT)
    shown = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nlqueries import config; print(config.DOCUMENT_PARSE_ISOLATION)",
        ],
        cwd=tmp_path,  # no .env from the checkout
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert shown == str(isolated)


def test_the_child_runs_under_the_parents_limits(source, monkeypatch) -> None:
    monkeypatch.setattr(_limits, "MAX_ROWS", 7)
    monkeypatch.setattr(_limits, "MAX_SECONDS", 33.0)

    (chunk,) = fake.ReportsLimits().ingest(source, "s1")

    assert chunk.text == "7 33.0"


def test_with_isolation_off_it_parses_here(source, monkeypatch) -> None:
    monkeypatch.setattr(config, "DOCUMENT_PARSE_ISOLATION", False)

    def no_child(*args, **kwargs):
        raise AssertionError("a child process was started")

    with patch.object(_isolation.subprocess, "Popen", no_child):
        (chunk,) = fake.Returns().ingest(source, "s1")
    assert chunk.text == "parsed doc.bin"


def test_a_real_document_parses_the_same_in_the_child(tmp_path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "policy.docx"
    document = docx.Document()
    document.add_paragraph("Refunds are accepted within 30 days of delivery.")
    document.add_paragraph("Store credit is offered after that.")
    document.save(str(path))

    in_child = WordConnector().ingest(path, "policy")
    here = WordConnector()._ingest_here(path, "policy")

    assert [c.text for c in in_child] == [c.text for c in here]
    assert "Refunds are accepted" in in_child[0].text


def test_the_expansion_cap_default_is_lower() -> None:
    assert config.MAX_DOCUMENT_EXPANDED_BYTES <= 128 * 1024 * 1024 or (
        "NLQ_MAX_DOCUMENT_EXPANDED_BYTES" in __import__("os").environ
    )
