"""
nlqueries.orchestrator.prompt_assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Builds the (system_prompt, user_prompt) pair from a YAML knowledge base.

Public API
----------
``assemble_prompt(question, knowledge_base, top_k_capsules, *, collection)``
    Select relevant schema tables and query capsules from *knowledge_base*,
    embed them as grounding context, and return the two-tuple of prompts
    suitable for passing directly to ``LLMClient.stream()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlqueries.orchestrator.document_retrieval import DocumentRetrievalResult


def assemble_prompt(
    question: str,
    knowledge_base: dict[str, Any],
    top_k_capsules: int = 5,
    *,
    collection: str | None = None,
) -> tuple[str, str]:
    """Assemble a (system_prompt, user_prompt) pair for *question*.

    Selects relevant schema tables and query capsules from *knowledge_base*
    and embeds them as grounding context in the system prompt.  When
    *collection* is given, ``search_schema`` / ``search`` from the Qdrant
    store are called to rank the most relevant entries; otherwise the raw
    KB dict is used directly (first *top_k_capsules* capsules, all tables).

    Args:
        question:       The user's natural-language question.
        knowledge_base: Parsed YAML KB dict as returned by
                        ``nlqueries.knowledge.kb_generator.generate_knowledge_base``.
        top_k_capsules: Number of example query capsules to embed in the
                        system prompt (default 5).
        collection:     Optional Qdrant collection name.  When provided,
                        semantic search is used to rank context snippets.

    Returns:
        ``(system_prompt, user_prompt)`` — both are plain strings.
    """
    schema_section = _build_schema_section(question, knowledge_base, collection)
    capsules_section = _build_capsules_section(question, knowledge_base, top_k_capsules, collection)
    business_section = _build_business_context_section(knowledge_base)

    system_parts: list[str] = [
        "You are a SQL generation assistant for a natural-language query interface.",
        "Use the schema and example queries below to translate the user's question into valid SQL.",
        "",
    ]
    if schema_section:
        system_parts.append(schema_section)
    if business_section:
        system_parts.append(business_section)
    if capsules_section:
        system_parts.append(capsules_section)
    system_parts += [
        "## Instructions",
        "- Generate only a single SELECT SQL statement.",
        "- Use only tables and columns present in the schema above.",
        "- Respond with SQL only — no explanations or markdown fences.",
    ]

    return "\n".join(system_parts), question


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_schema_section(
    question: str,
    knowledge_base: dict[str, Any],
    collection: str | None,
) -> str:
    """Return a formatted Markdown section describing the relevant schema tables."""
    tables: list[dict[str, Any]] = knowledge_base.get("schema", {}).get("tables", [])
    if not tables:
        return ""

    relevant_tables = tables
    if collection:
        try:
            from nlqueries.embeddings.qdrant_store import search_schema

            hits = search_schema(collection, question, top_k=10)
            hit_names = {h.get("table_name") for h in hits if h.get("table_name")}
            if hit_names:
                relevant_tables = [t for t in tables if t.get("name") in hit_names] or tables
        except Exception:  # noqa: BLE001 — Qdrant unreachable, fall back to full KB
            pass

    lines: list[str] = ["## Database Schema", ""]
    for table in relevant_tables:
        name = table.get("name", "")
        desc = table.get("description", "")
        lines.append(f"### Table: {name}")
        if desc:
            lines.append(f"Description: {desc}")
        row_count = table.get("row_count")
        if row_count is not None:
            lines.append(f"Rows: {row_count:,}")
        lines.append("Columns:")
        for col in table.get("columns", []):
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            col_desc = col.get("description", "")
            if col_desc:
                lines.append(f"  - {col_name} ({col_type}): {col_desc}")
            else:
                lines.append(f"  - {col_name} ({col_type})")
        lines.append("")
    return "\n".join(lines)


def _build_capsules_section(
    question: str,
    knowledge_base: dict[str, Any],
    top_k: int,
    collection: str | None,
) -> str:
    """Return a formatted Markdown section with example query capsules."""
    raw_capsules: list[dict[str, Any]] = knowledge_base.get("query_capsules", [])
    if not raw_capsules:
        return ""

    selected: list[dict[str, Any]] = raw_capsules[:top_k]
    if collection:
        try:
            from nlqueries.embeddings.qdrant_store import search

            qdrant_hits = search(collection, question, top_k=top_k)
            if qdrant_hits:
                selected = [
                    {
                        "intent": c.intent or c.auto_description,
                        "template": c.template_sql,
                    }
                    for c in qdrant_hits
                ]
        except Exception:  # noqa: BLE001 — Qdrant unreachable, fall back to raw KB
            pass

    lines: list[str] = [
        "## Example Queries",
        "",
        "The following queries represent common patterns in this database:",
        "",
    ]
    for i, cap in enumerate(selected, 1):
        intent = cap.get("intent", "")
        template = cap.get("template", "")
        if not intent and not template:
            continue
        lines.append(f"{i}. Intent: {intent}")
        if template:
            lines.append(f"   SQL Template: {template}")
        lines.append("")
    return "\n".join(lines)


def _build_business_context_section(knowledge_base: dict[str, Any]) -> str:
    """Return a formatted Markdown section for glossary and business rules."""
    biz: dict[str, Any] = knowledge_base.get("business_context", {})
    glossary: list[Any] = biz.get("glossary", [])
    rules: list[Any] = biz.get("rules", [])
    if not glossary and not rules:
        return ""

    lines: list[str] = ["## Business Context", ""]
    if glossary:
        lines.append("### Glossary")
        for entry in glossary:
            if isinstance(entry, dict):
                lines.append(f"- {entry.get('term', '')}: {entry.get('definition', '')}")
            else:
                lines.append(f"- {entry}")
        lines.append("")
    if rules:
        lines.append("### Business Rules")
        for rule in rules:
            lines.append(f"- {rule}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document Q&A prompt assembly (Task 11.1)
# ---------------------------------------------------------------------------


def assemble_document_prompt(
    question: str,
    retrieval_result: DocumentRetrievalResult,
) -> tuple[str, str]:
    """Build ``(system_prompt, user_prompt)`` for document Q&A.

    The system prompt instructs the LLM to answer using only the provided
    document chunks and to cite sources inline as ``[source_name, page N]``
    (or ``[source_name]`` when no page number is available).

    The user prompt contains the original question followed by the formatted
    chunk context, with each chunk labelled by its citation index.

    Args:
        question:         The user's natural-language question.
        retrieval_result: Result from ``retrieve_for_question``, containing
                          chunks and citations sorted by relevance.

    Returns:
        ``(system_prompt, user_prompt)`` — both are plain strings.
    """
    system_prompt = (
        "You are a document question-answering assistant.\n"
        "Answer the user's question using ONLY the document excerpts provided below.\n"
        "You MUST cite sources inline using the format [source_name, page N] when a page "
        "number is available, or [source_name] when it is not.\n"
        "If the answer cannot be found in the provided excerpts, say so explicitly — "
        "do not fabricate information.\n"
        "Do not reference any knowledge outside the provided document context."
    )

    context_lines: list[str] = ["## Document Context", ""]
    for i, (chunk, citation) in enumerate(
        zip(retrieval_result.chunks, retrieval_result.citations, strict=True),
        start=1,
    ):
        if citation.page_number is not None:
            label = f"[{citation.source_name}, page {citation.page_number}]"
        else:
            label = f"[{citation.source_name}]"
        context_lines.append(f"### Excerpt {i} — {label}")
        context_lines.append(chunk.text)
        context_lines.append("")

    user_prompt = f"{question}\n\n" + "\n".join(context_lines)
    return system_prompt, user_prompt
