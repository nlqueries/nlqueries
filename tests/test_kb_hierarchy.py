"""
Tests for nlqueries.knowledge.concept_hierarchy (CG-2.1) and its wiring into the
KB generator, kb-stats, and prompt assembly: optional glossary ``parent:`` links
plus dotted signal-tag subsumption.
"""

from __future__ import annotations

import pytest
from nlqueries.connectors.base import SchemaSpec, TableSpec
from nlqueries.knowledge.concept_hierarchy import (
    HierarchyError,
    build_glossary_hierarchy,
    expand_tag_with_ancestors,
    tag_ancestors,
    tag_covers,
)

# ---------------------------------------------------------------------------
# Signal tags — dotted-path subsumption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("pii", []),
        ("pii.email", ["pii"]),
        ("pii.contact.email", ["pii.contact", "pii"]),
        ("", []),
        ("a.b.c.d", ["a.b.c", "a.b", "a"]),
    ],
)
def test_tag_ancestors(tag: str, expected: list[str]) -> None:
    assert tag_ancestors(tag) == expected


def test_tag_covers_is_segment_bounded() -> None:
    assert tag_covers("pii", "pii.email")
    assert tag_covers("pii.email", "pii.email")  # covers itself
    assert tag_covers("pii.contact", "pii.contact.email")
    # Not a segment-boundary prefix — must NOT match.
    assert not tag_covers("pi", "pii.email")
    assert not tag_covers("piix", "pii.email")
    assert not tag_covers("pii.email", "pii")  # descendant does not cover ancestor


def test_expand_tag_with_ancestors() -> None:
    assert expand_tag_with_ancestors("pii.email") == ["pii.email", "pii"]
    assert expand_tag_with_ancestors("pii") == ["pii"]


# ---------------------------------------------------------------------------
# Glossary hierarchy — happy paths
# ---------------------------------------------------------------------------

_GLOSSARY = [
    {"term": "Customer", "definition": "Has placed at least one order."},
    {"term": "ActiveCustomer", "definition": "Ordered recently.", "parent": "Customer"},
    {"term": "VipCustomer", "definition": "Big spender.", "parent": "ActiveCustomer"},
    "PlainString",  # bare-string entry, tolerated, no parent
]


def test_parent_and_ancestors() -> None:
    h = build_glossary_hierarchy(_GLOSSARY)
    assert h.parent("ActiveCustomer") == "Customer"
    assert h.parent("Customer") is None
    assert h.ancestors("VipCustomer") == ["ActiveCustomer", "Customer"]
    assert h.ancestors("Customer") == []


def test_roots_and_depth() -> None:
    h = build_glossary_hierarchy(_GLOSSARY)
    assert set(h.roots()) == {"Customer", "PlainString"}
    assert h.max_depth() == 2  # VipCustomer -> ActiveCustomer -> Customer


def test_descendants_breadth_first_depth_capped() -> None:
    glossary = [
        {"term": "A"},
        {"term": "B", "parent": "A"},
        {"term": "C", "parent": "B"},
        {"term": "D", "parent": "C"},
        {"term": "E", "parent": "A"},
    ]
    h = build_glossary_hierarchy(glossary)
    assert set(h.descendants("A", max_depth=1)) == {"B", "E"}
    assert set(h.descendants("A", max_depth=2)) == {"B", "E", "C"}
    assert set(h.descendants("A", max_depth=3)) == {"B", "E", "C", "D"}
    assert h.descendants("D") == []  # a leaf has none


def test_descendants_cycle_safe() -> None:
    h = build_glossary_hierarchy(
        [{"term": "A", "parent": "B"}, {"term": "B", "parent": "A"}], validate=False
    )
    # Must terminate (no infinite loop) despite the cycle.
    assert set(h.descendants("A")) == {"B"}


def test_bare_string_entry_has_no_parent() -> None:
    h = build_glossary_hierarchy(_GLOSSARY)
    assert h.parent("PlainString") is None
    assert "PlainString" in h.parents


def test_empty_glossary_is_flat() -> None:
    h = build_glossary_hierarchy([])
    assert h.max_depth() == 0
    assert h.roots() == []
    assert h.orphans() == []


# ---------------------------------------------------------------------------
# Dangling parent — treated as root + reported as orphan (never an error)
# ---------------------------------------------------------------------------


def test_dangling_parent_is_orphan_not_error() -> None:
    glossary = [{"term": "Widget", "definition": "x", "parent": "DoesNotExist"}]
    h = build_glossary_hierarchy(glossary)  # validate=True, must NOT raise
    assert h.parent("Widget") is None  # dangling → treated as root
    assert h.ancestors("Widget") == []
    assert h.orphans() == ["Widget"]
    assert "Widget" in h.roots()


# ---------------------------------------------------------------------------
# Cycles — rejected when validating, survivable when not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "glossary",
    [
        [{"term": "A", "parent": "A"}],  # self-cycle
        [{"term": "A", "parent": "B"}, {"term": "B", "parent": "A"}],  # 2-cycle
        [
            {"term": "A", "parent": "B"},
            {"term": "B", "parent": "C"},
            {"term": "C", "parent": "A"},
        ],  # 3-cycle
    ],
)
def test_cycle_raises_when_validating(glossary: list) -> None:
    with pytest.raises(HierarchyError, match="cycle"):
        build_glossary_hierarchy(glossary, validate=True)


def test_cycle_survivable_without_validation() -> None:
    glossary = [{"term": "A", "parent": "B"}, {"term": "B", "parent": "A"}]
    h = build_glossary_hierarchy(glossary, validate=False)  # must not raise
    # Ancestor walk is cycle-safe (terminates, no infinite loop).
    assert h.ancestors("A") == ["B"]
    assert h.ancestors("B") == ["A"]


def test_cycle_only_over_known_edges() -> None:
    # A -> missing is not a cycle even though 'missing' is absent.
    h = build_glossary_hierarchy([{"term": "A", "parent": "missing"}], validate=True)
    assert h.orphans() == ["A"]


# ---------------------------------------------------------------------------
# kb_generator preserves business_context across regeneration
# ---------------------------------------------------------------------------


def _schema() -> SchemaSpec:
    return SchemaSpec(
        database="shop",
        tables=[
            TableSpec(
                name="orders",
                schema="public",
                row_count=10,
                columns=[],
                description="",
            )
        ],
        extracted_at="2026-07-23T00:00:00Z",
    )


def test_generator_preserves_glossary_hierarchy() -> None:
    from nlqueries.knowledge.kb_generator import generate_knowledge_base

    existing = {
        "business_context": {
            "glossary": [
                {"term": "Customer", "definition": "c"},
                {"term": "ActiveCustomer", "definition": "a", "parent": "Customer"},
            ],
            "rules": ["Exclude cancelled orders."],
        }
    }
    kb = generate_knowledge_base(_schema(), [], "shop", existing_kb=existing)
    glossary = kb["business_context"]["glossary"]
    assert {e["term"] for e in glossary} == {"Customer", "ActiveCustomer"}
    active = next(e for e in glossary if e["term"] == "ActiveCustomer")
    assert active["parent"] == "Customer"
    assert kb["business_context"]["rules"] == ["Exclude cancelled orders."]


def test_generator_flat_when_no_existing_kb() -> None:
    from nlqueries.knowledge.kb_generator import generate_knowledge_base

    kb = generate_knowledge_base(_schema(), [], "shop")
    assert kb["business_context"] == {"glossary": [], "rules": []}


def test_generator_rejects_cyclic_hierarchy() -> None:
    from nlqueries.knowledge.kb_generator import generate_knowledge_base

    existing = {
        "business_context": {
            "glossary": [{"term": "A", "parent": "B"}, {"term": "B", "parent": "A"}],
            "rules": [],
        }
    }
    with pytest.raises(HierarchyError):
        generate_knowledge_base(_schema(), [], "shop", existing_kb=existing)


# ---------------------------------------------------------------------------
# Round-trip: save -> load -> save is byte-stable with parent present
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_byte_stable(tmp_path) -> None:
    import yaml
    from nlqueries.knowledge.kb_generator import save_knowledge_base

    kb = {
        "kb_version": 2,
        "db_name": "shop",
        "schema": {"tables": [], "foreign_keys": []},
        "business_context": {
            "glossary": [
                {"term": "Customer", "definition": "c"},
                {"term": "ActiveCustomer", "definition": "a", "parent": "Customer"},
            ],
            "rules": [],
        },
        "query_capsules": [],
    }
    p1 = tmp_path / "a.yaml"
    save_knowledge_base(kb, str(p1))
    first = p1.read_text(encoding="utf-8")
    assert "parent: Customer" in first

    reloaded = yaml.safe_load(first)
    p2 = tmp_path / "b.yaml"
    save_knowledge_base(reloaded, str(p2))
    assert p2.read_text(encoding="utf-8") == first


# ---------------------------------------------------------------------------
# kb-stats reports hierarchy info
# ---------------------------------------------------------------------------


def test_kb_stats_reports_hierarchy(tmp_path) -> None:
    import yaml
    from nlqueries.knowledge.kb_stats import compute_kb_stats

    kb = {
        "kb_version": 2,
        "schema": {"tables": []},
        "business_context": {
            "glossary": [
                {"term": "Customer", "definition": "c"},
                {"term": "ActiveCustomer", "definition": "a", "parent": "Customer"},
                {"term": "Vip", "definition": "v", "parent": "ActiveCustomer"},
                {"term": "Broken", "definition": "b", "parent": "Nope"},  # orphan
            ],
        },
        "query_capsules": [],
    }
    p = tmp_path / "shop.yaml"
    p.write_text(yaml.safe_dump(kb), encoding="utf-8")
    stats = compute_kb_stats("shop", p)
    assert stats.glossary_terms == 4
    assert stats.glossary_with_parent == 3
    assert stats.hierarchy_depth == 2  # Vip -> ActiveCustomer -> Customer
    assert stats.glossary_orphans == 1  # Broken
    assert stats.glossary_has_cycle is False


def test_kb_stats_flags_cycle(tmp_path) -> None:
    import yaml
    from nlqueries.knowledge.kb_stats import compute_kb_stats

    kb = {
        "schema": {"tables": []},
        "business_context": {
            "glossary": [{"term": "A", "parent": "B"}, {"term": "B", "parent": "A"}],
        },
        "query_capsules": [],
    }
    p = tmp_path / "shop.yaml"
    p.write_text(yaml.safe_dump(kb), encoding="utf-8")
    stats = compute_kb_stats("shop", p)
    assert stats.glossary_has_cycle is True


# ---------------------------------------------------------------------------
# prompt_assembly renders parent context on injected terms
# ---------------------------------------------------------------------------


def test_prompt_assembly_shows_parent_context() -> None:
    from nlqueries.orchestrator.prompt_assembly import _build_business_context_section

    kb = {
        "business_context": {
            "glossary": [
                {"term": "Customer", "definition": "Placed an order."},
                {"term": "ActiveCustomer", "definition": "Ordered recently.", "parent": "Customer"},
                {"term": "Broken", "definition": "x", "parent": "Ghost"},  # dangling
            ],
        }
    }
    out = _build_business_context_section(kb)
    assert "- Customer: Placed an order." in out
    assert "- ActiveCustomer (a kind of Customer): Ordered recently." in out
    # A dangling parent renders flat, no "(a kind of ...)".
    assert "- Broken: x" in out
    assert "a kind of Ghost" not in out
