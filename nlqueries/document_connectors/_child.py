"""
nlqueries.document_connectors._child
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The child process :mod:`._isolation` starts to parse one document.

Reads its instructions from stdin, applies the limits, parses, and writes the
outcome to stdout. Anything a parser prints goes to stderr, so it cannot corrupt
the result.
"""

from __future__ import annotations

import pickle  # noqa: S403 - both ends are this package
import sys

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
        limits.BasicLimitInformation.LimitFlags = _LIMIT_PROCESS_MEMORY
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


def main() -> int:
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    payload = pickle.loads(sys.stdin.buffer.read())  # noqa: S301 - written by our parent

    from nlqueries.document_connectors import _isolation, _limits  # noqa: PLC0415

    for name, value in payload["limits"].items():
        setattr(_limits, name, value)
    _apply_resource_limits(payload["memory_bytes"], payload["cpu_seconds"])

    result = _isolation.child_result(payload["connector"], payload["source"], payload["source_id"])
    out.write(pickle.dumps(result))
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
