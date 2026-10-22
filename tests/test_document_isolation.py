"""PDF, Word and Excel parsing in a child process with time and memory limits.

The fake connectors in ``tests/_fake_parsers.py`` misbehave in ways the
in-process budget cannot stop -- stuck inside one unit of work, allocating
without bound, spinning -- and these check that the child is stopped and the
caller gets the refusal it already handles.
"""

from __future__ import annotations

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

        def communicate(self, input=None, timeout=None):
            if input is not None:
                self.stdin.write(input)  # the child starts parsing
                self.stdin.close()
                time.sleep(1)
                raise KeyboardInterrupt
            return super().communicate(timeout=timeout)

    monkeypatch.setattr(_isolation.subprocess, "Popen", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        fake.Hangs().ingest(source, "s1")

    (child,) = started
    assert child.poll() is not None  # killed and reaped, not left behind
    assert child.stdout is not None and child.stdout.closed


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
