# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.state_files
~~~~~~~~~~~~~~~~~~~~~
Permissions for the files NLQueries writes under the state directory.

Session and feedback files hold questions, generated SQL and corrections, which
disclose the schema and the shape of the data. They were created with the
process umask, which on most systems makes them readable by every account on the
host (SEC-20).

Knowledge bases, query capsules and the connector registry are written through
:func:`write_private`, which also makes the write atomic: a reader sees either
the previous file or the new one, never a half-written one.

On Windows the POSIX permission bits are not enforced, and ``Path.chmod`` sets
only the read-only flag. These functions are therefore a no-op there in
practice; access control on that platform is the directory ACL, which is outside
what this can set portably.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path

#: Owner may read, write and enter. Nobody else.
_DIR_MODE = stat.S_IRWXU

#: Owner may read and write. Nobody else.
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def private_dir(path: Path) -> Path:
    """Create *path* if absent, restrict it to its owner, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    # A directory whose mode cannot be set is still usable; the caller is
    # writing state, not enforcing policy.
    with contextlib.suppress(OSError):
        path.chmod(_DIR_MODE)
    return path


def restrict(path: Path) -> Path:
    """Restrict an existing file to its owner, and return it.

    Applied after writing rather than before: a file created by ``open`` takes
    the process umask, and there is no portable way to pass a mode to it.
    """
    with contextlib.suppress(OSError):
        if path.exists():
            path.chmod(_FILE_MODE)
    return path


def write_private(path: Path, text: str) -> Path:
    """Write *text* to *path* readable by its owner only, atomically, and return it.

    The text goes to a temporary file in the same directory, created at mode
    0600 so it is never readable by others, not even briefly. It is flushed to
    disk and then renamed over *path*. The rename replaces whatever is at
    *path*: an interrupted write leaves the previous file as it was, and a
    symlink there is replaced by the new file rather than written through.

    A missing parent directory is created with :func:`private_dir`. One that
    already exists is left as it is: *path* may be an output the operator chose,
    and its directory is theirs, not state this owns.
    """
    if not path.parent.is_dir():
        private_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return path
