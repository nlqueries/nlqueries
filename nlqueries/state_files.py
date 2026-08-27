# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.state_files
~~~~~~~~~~~~~~~~~~~~~
Permissions for the files NLQueries writes under the state directory.

Session and feedback files hold questions, generated SQL and corrections, which
disclose the schema and the shape of the data. They were created with the
process umask, which on most systems makes them readable by every account on the
host (SEC-20).

On Windows the POSIX permission bits are not enforced, and ``Path.chmod`` sets
only the read-only flag. These functions are therefore a no-op there in
practice; access control on that platform is the directory ACL, which is outside
what this can set portably.
"""

from __future__ import annotations

import contextlib
import stat
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
