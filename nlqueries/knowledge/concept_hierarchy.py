# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.knowledge.concept_hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional concept subsumption for the knowledge base (CG-2.1). Two flavours, both
purely structural (no LLM, no enterprise imports) so they are reusable from the
enterprise tier's enforcement path (CG-2.2):

1. **Glossary terms** carry an optional single ``parent:`` string naming another
   term — ``ActiveCustomer is-a Customer``. One parent only (no multiple
   inheritance in v1). Cycles are rejected; a ``parent:`` that names a term that
   does not exist is *not* an error — the term is treated as a root and reported
   as an orphan so a typo never makes a KB unloadable.

2. **Signal tags** are dotted strings — ``pii.email is-a pii`` — so their
   hierarchy is the dotted path itself. A guard or rule on ``pii`` subsumes every
   ``pii.*`` descendant via segment-boundary ancestry (``pii`` covers
   ``pii.email`` but not ``piix.email``).

Both are opt-in and backward compatible: a flat glossary and flat tags behave
exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TAG_SEPARATOR = "."


class HierarchyError(ValueError):
    """A glossary hierarchy is malformed — currently only raised for cycles."""


# ---------------------------------------------------------------------------
# Signal tags — dotted-path subsumption
# ---------------------------------------------------------------------------


def tag_ancestors(tag: str) -> list[str]:
    """Ancestor tags of a dotted tag, nearest first.

    ``"pii.contact.email"`` -> ``["pii.contact", "pii"]``; a top-level tag
    (``"pii"``) or the empty string has no ancestors.
    """
    parts = [p for p in tag.split(TAG_SEPARATOR) if p]
    if len(parts) <= 1:
        return []
    return [TAG_SEPARATOR.join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def tag_covers(ancestor: str, tag: str) -> bool:
    """True when *ancestor* subsumes *tag* on segment boundaries.

    ``tag_covers("pii", "pii.email")`` is ``True``; ``tag_covers("pi", ...)`` and
    ``tag_covers("piix", ...)`` are ``False``. A tag always covers itself.
    """
    return ancestor == tag or ancestor in tag_ancestors(tag)


def expand_tag_with_ancestors(tag: str) -> list[str]:
    """The tag followed by its ancestors — the full is-a chain for matching."""
    return [tag, *tag_ancestors(tag)]


# ---------------------------------------------------------------------------
# Glossary terms — explicit single-parent hierarchy
# ---------------------------------------------------------------------------


def _term_of(entry: Any) -> str | None:
    """The term name of a glossary entry (dict with ``term`` or a bare string)."""
    if isinstance(entry, dict):
        term = str(entry.get("term", "")).strip()
        return term or None
    if isinstance(entry, str):
        return entry.strip() or None
    return None


def _declared_parent(entry: Any) -> str | None:
    """The raw ``parent:`` of a glossary entry, if any (bare strings have none)."""
    if isinstance(entry, dict):
        parent = entry.get("parent")
        if isinstance(parent, str) and parent.strip():
            return parent.strip()
    return None


@dataclass(frozen=True)
class GlossaryHierarchy:
    """Resolved parent/ancestor view over a KB glossary.

    ``parents`` maps every known term to its *declared* parent string (which may
    be dangling). The public methods only ever return **known** terms, so a
    dangling parent surfaces as a root here and an orphan in :meth:`orphans`.
    """

    parents: dict[str, str | None]

    def _known(self, term: str) -> bool:
        return term in self.parents

    def parent(self, term: str) -> str | None:
        """The term's parent, or ``None`` if it is a root or its parent is dangling."""
        declared = self.parents.get(term)
        return declared if declared is not None and self._known(declared) else None

    def ancestors(self, term: str) -> list[str]:
        """Known ancestors, nearest first; cycle-safe even on an unvalidated graph."""
        chain: list[str] = []
        seen: set[str] = {term}
        current = self.parent(term)
        while current is not None and current not in seen:
            chain.append(current)
            seen.add(current)
            current = self.parent(current)
        return chain

    def orphans(self) -> list[str]:
        """Terms whose declared parent is not itself a known term (dangling refs)."""
        return sorted(
            term
            for term, declared in self.parents.items()
            if declared is not None and not self._known(declared)
        )

    def roots(self) -> list[str]:
        """Terms with no (effective) parent."""
        return sorted(term for term in self.parents if self.parent(term) is None)

    def max_depth(self) -> int:
        """Longest ancestor chain across all terms (0 for a flat glossary)."""
        return max((len(self.ancestors(term)) for term in self.parents), default=0)


def _detect_cycle(parents: dict[str, str | None]) -> list[str] | None:
    """Return a cycle path (over known-term edges) if one exists, else ``None``."""
    color: dict[str, int] = {}  # 0 = visiting, 1 = done

    def walk(term: str, path: list[str]) -> list[str] | None:
        color[term] = 0
        declared = parents.get(term)
        # Only known-term edges can form a cycle; a dangling parent is a leaf.
        if declared is not None and declared in parents:
            state = color.get(declared)
            if state == 0:  # back-edge → cycle
                idx = path.index(declared) if declared in path else 0
                return [*path[idx:], term, declared]
            if state is None:
                found = walk(declared, [*path, term])
                if found:
                    return found
        color[term] = 1
        return None

    for node in parents:
        if color.get(node) is None:
            cycle = walk(node, [])
            if cycle:
                return cycle
    return None


def build_glossary_hierarchy(
    glossary: list[Any] | None, *, validate: bool = True
) -> GlossaryHierarchy:
    """Build a :class:`GlossaryHierarchy` from a KB glossary list.

    When *validate* is true (the default, used at build/save and stats time), a
    cycle raises :class:`HierarchyError`. Runtime callers pass ``validate=False``
    so a malformed KB is rendered flat rather than crashing a live query.
    """
    parents: dict[str, str | None] = {}
    for entry in glossary or []:
        term = _term_of(entry)
        if term is None:
            continue
        # First declaration wins; a repeated term keeps its first parent.
        parents.setdefault(term, _declared_parent(entry))

    if validate:
        cycle = _detect_cycle(parents)
        if cycle is not None:
            raise HierarchyError("glossary hierarchy has a cycle: " + " -> ".join(cycle))

    return GlossaryHierarchy(parents=parents)
