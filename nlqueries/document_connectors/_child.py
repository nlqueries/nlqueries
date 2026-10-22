"""
nlqueries.document_connectors._child
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The child process :mod:`._isolation` starts to parse one document.

Reads its instructions from stdin, applies the limits, parses, and writes the
outcome to stdout. Anything a parser writes to stdout goes to stderr instead, so
it cannot corrupt the result.
"""

from __future__ import annotations

import contextlib
import os
import pickle  # noqa: S403 - both ends are this package
import signal
import sys
import threading

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - the ctypes layout of JOBOBJECT_BASIC_LIMIT_INFORMATION
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - IO_COUNTERS
            (name, ctypes.c_uint64)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _EXTENDED_LIMIT_INFORMATION = 9  # JobObjectExtendedLimitInformation
    _LIMIT_PROCESS_MEMORY = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    # The job's members are killed when its last handle closes: this
    # process holds the only one, so whatever the parser started ends with it.
    _LIMIT_KILL_ON_JOB_CLOSE = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    def _apply_resource_limits(memory_bytes: int, cpu_seconds: int) -> None:
        """A Job Object memory limit; Windows has no CPU-time limit to match.

        The process puts itself in a job of its own, which Windows 8 and later
        allow inside whatever job it already runs in. An allocation past the
        limit fails, which Python reports as ``MemoryError``, as ``RLIMIT_AS``
        does on POSIX. The wall-clock deadline in the parent is the time limit.
        If the job cannot be set up, the error ends the child and the parse is
        reported as failed, rather than running without the limit.
        """
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        # The handle stays open for the life of the process, which is the job's.
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _LIMIT_PROCESS_MEMORY | _LIMIT_KILL_ON_JOB_CLOSE
        limits.ProcessMemoryLimit = memory_bytes
        if not kernel32.SetInformationJobObject(
            job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())

else:
    import resource

    def _within_hard_limit(which: int, wanted: int) -> int:
        """*wanted*, or the hard limit if that is lower, which an unprivileged
        process cannot raise: a container started with a smaller ``ulimit``
        keeps it, rather than every parse failing to set its own."""
        _soft, hard = resource.getrlimit(which)
        return wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)

    def _apply_resource_limits(memory_bytes: int, cpu_seconds: int) -> None:
        """``RLIMIT_AS`` and ``RLIMIT_CPU``."""
        memory = _within_hard_limit(resource.RLIMIT_AS, memory_bytes)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        # SIGXCPU at the soft limit; the hard limit, a little later, kills outright.
        soft = _within_hard_limit(resource.RLIMIT_CPU, cpu_seconds)
        resource.setrlimit(
            resource.RLIMIT_CPU, (soft, _within_hard_limit(resource.RLIMIT_CPU, soft + 5))
        )


def _arm_watchdog(seconds: float | None) -> None:
    """End this process at *seconds*, whatever the parser is doing.

    The parent kills the child at the same deadline, so this matters only when
    the parent cannot: killed itself, or stopped by a hard time limit that gave
    it no chance. Without it the parse would run on, holding the file and up
    to its memory limit, and on Windows with nothing bounding its time.

    What the parser started goes too. On Windows the Job Object sees to that
    however this process ends. On POSIX the parent would kill the process
    group this process leads, and the parent is the one thing not there, so
    the watchdog kills the group itself -- this process with it, so the exit
    is SIGKILL rather than the watchdog's code, which reads as a timeout all
    the same. Only when it leads the group, as the parent starts it: run any
    other way, the group is someone else's.
    """
    if not seconds:
        return
    from nlqueries.document_connectors._isolation import WATCHDOG_EXIT_CODE  # noqa: PLC0415

    def fire() -> None:
        if sys.platform != "win32" and os.getpgrp() == os.getpid():
            with contextlib.suppress(OSError):
                os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(WATCHDOG_EXIT_CODE)

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()


def main() -> int:
    # The result goes out on a private copy of descriptor 1, and descriptor 1
    # itself is pointed at stderr. Rebinding `sys.stdout` alone would leave the
    # descriptor to the parser's C libraries, `sys.__stdout__` and any process
    # the parser starts, and one stray byte there makes the result unreadable.
    # The copy is not inherited (PEP 446), so a grandchild cannot reach it.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    payload = pickle.loads(sys.stdin.buffer.read())  # noqa: S301 - written by our parent

    from nlqueries.document_connectors import _isolation, _limits  # noqa: PLC0415

    for name, value in payload["limits"].items():
        setattr(_limits, name, value)
    _apply_resource_limits(payload["memory_bytes"], payload["cpu_seconds"])
    _arm_watchdog(payload.get("deadline_seconds"))

    result = _isolation.child_result(payload["connector"], payload["source"], payload["source_id"])
    out.write(pickle.dumps(result))
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
