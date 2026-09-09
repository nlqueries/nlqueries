#!/usr/bin/env python3
"""Check that the external links in our Markdown still resolve.

The README is the front page of this repository *and* what PyPI renders on the
project page, so a link that rots is broken in the two places people are most
likely to meet this project — and nothing would have said so. It is also the
file most likely to point at a site whose structure someone else controls.

Deliberately narrow:

* Only ``https://`` links. Relative links are checked by the reader's own
  filesystem, and a broken one is obvious in review; a broken external link is
  invisible until someone clicks it.
* HEAD, falling back to GET, because plenty of servers do not implement HEAD.
* Retries, because a link checker that fails on one dropped connection gets
  switched off within a month, and a check nobody trusts is worse than none.
* 401 and 403 count as reachable. They mean a server answered about a page that
  exists; a login wall is not a dead link.

Exit code 1 lists every failure rather than the first, so one run tells you
everything to fix.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Any `](https://…)` target, plus bare angle-bracket autolinks.
#
# Deliberately not anchored on a preceding `[text]`: a badge is
# `[![alt](image)](target)`, and a pattern requiring `[` with no `]` inside
# consumes `[![alt](image)` and matches only the *image*, leaving the target
# unchecked while the run still prints a count and says everything resolves.
# Every badge at the top of the README reaches its destination through exactly
# that shape.
_LINK = re.compile(r"\]\((https://[^)\s]+)\)|<(https://[^>\s]+)>")

_TIMEOUT_S = 20
_ATTEMPTS = 3
_BACKOFF_S = 2.0

#: A server answering "you may not see this" has confirmed the page exists.
_REACHABLE = {200, 201, 202, 203, 204, 301, 302, 303, 307, 308, 401, 403}

#: Hosts that rate-limit or block automated requests hard enough that a failure
#: here says nothing about the link. Skipped rather than silently passed, and
#: reported, so the list cannot quietly grow to cover a real problem.
#:
#: Compared against the parsed host, not searched for in the whole URL: as a
#: substring test an entry like "github.com" would also skip
#: "example.com/?ref=github.com", and one short entry could silently exempt
#: half the file.
_SKIP_HOSTS = frozenset({"img.shields.io", "visitor-badge.laobi.icu"})

_UA = "nlqueries-link-check/1.0 (+https://github.com/nlqueries/nlqueries)"


def find_links(paths: list[pathlib.Path]) -> dict[str, list[str]]:
    """Map each URL to the files it appears in."""
    found: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _LINK.finditer(text):
            url = (match.group(1) or match.group(2)).rstrip(".,;:")
            found.setdefault(url, []).append(str(path))
    return found


def check(url: str) -> tuple[str, int | None, str]:
    """Return ``(url, status, note)``. ``status`` is None when nothing answered."""
    last = ""
    for attempt in range(_ATTEMPTS):
        for method in ("HEAD", "GET"):
            request = urllib.request.Request(url, method=method, headers={"User-Agent": _UA})
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                    return url, int(response.status), ""
            except urllib.error.HTTPError as exc:
                if exc.code in _REACHABLE:
                    return url, exc.code, ""
                # 405 means HEAD is unsupported, not that the page is missing.
                if exc.code != 405:
                    last = f"HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001 — any transport failure
                last = f"{type(exc).__name__}: {exc}"
        if attempt < _ATTEMPTS - 1:
            time.sleep(_BACKOFF_S * (attempt + 1))
    return url, None, last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["README.md", "docs"])
    args = parser.parse_args()

    files: list[pathlib.Path] = []
    missing: list[str] = []
    for raw in args.paths:
        path = pathlib.Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.is_file():
            files.append(path)
        else:
            missing.append(raw)

    # Refusing rather than skipping. A renamed directory, a mistyped name, or a
    # run from the wrong working directory would otherwise produce
    # "Checking 0 external links across 0 files. All links resolve." and exit 0
    # — a green tick over nothing, which is the exact failure this script exists
    # to prevent, wearing the costume of a pass.
    if missing:
        print(f"No such path: {', '.join(missing)}", file=sys.stderr)
        return 1

    links = find_links(files)
    skipped = [u for u in links if urllib.parse.urlparse(u).hostname in _SKIP_HOSTS]
    to_check = sorted(u for u in links if u not in skipped)

    # Checked against what will actually be requested, not against what was
    # found. Guarding the found set left a file whose every URL is skipped
    # printing "Checking 0 external links" and then "All links resolve" -- the
    # same green tick over nothing, one filter further along.
    if not to_check:
        print(
            f"No external links to check across {len(files)} file(s) "
            f"({len(skipped)} skipped). That is almost certainly wrong, so this "
            "is a failure rather than a pass.",
            file=sys.stderr,
        )
        return 1

    print(f"Checking {len(to_check)} external links across {len(files)} files.")
    for url in sorted(skipped):
        print(f"  skipped (rate-limits automated requests): {url}")

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for url, status, note in pool.map(check, to_check):
            if status is None or status not in _REACHABLE:
                failures.append((url, note or f"HTTP {status}"))

    if not failures:
        print("All links resolve.")
        return 0

    print(f"\n{len(failures)} link(s) did not resolve:", file=sys.stderr)
    for url, note in sorted(failures):
        where = ", ".join(sorted(set(links[url])))
        print(f"  {url}\n      {note}\n      in: {where}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
