"""
nlqueries.orchestrator.document_orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Document Q&A orchestrator — answers natural-language questions from ingested
document chunks instead of generating SQL.

Public API
----------
``DocumentOrchestrator``
    Call ``handle_question(question, collection, source_id, top_k)`` to get an
    async token stream ending in a structured JSON citations chunk.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from nlqueries.llm import get_llm_client
from nlqueries.orchestrator.document_retrieval import retrieve_for_question
from nlqueries.orchestrator.prompt_assembly import assemble_document_prompt


class DocumentOrchestrator:
    """Answer natural-language questions from ingested document chunks.

    Each invocation of ``handle_question`` independently:

    1. Retrieves the most relevant chunks from a Qdrant collection.
    2. Assembles a grounded LLM prompt from those chunks.
    3. Streams the LLM response token by token.
    4. Yields a final structured JSON citations chunk.

    Usage::

        orch = DocumentOrchestrator()
        async for token in orch.handle_question(
            question="What is the refund policy?",
            collection="doc_my-policy-doc-uuid_chunks",
            source_id="my-policy-doc-uuid",
        ):
            print(token, end="", flush=True)
    """

    async def handle_question(
        self,
        question: str,
        collection: str,
        source_id: str | None = None,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """Retrieve document chunks and stream a grounded LLM answer.

        Yields LLM response tokens one at a time, then yields a single
        structured JSON string as the final chunk::

            {"type": "citations", "citations": [
                {"source_name": "...", "page_number": 1, "excerpt": "..."},
                ...
            ]}

        Args:
            question:   The natural-language question from the user.
            collection: Qdrant collection name (convention:
                        ``doc_{source_id}_chunks``).
            source_id:  When set, restricts retrieval to this source only.
            top_k:      Maximum number of chunks to retrieve (default 5).

        Yields:
            String tokens from the LLM response, then a final JSON citations
            chunk.
        """
        retrieval_result = retrieve_for_question(
            question,
            collection,
            top_k=top_k,
            source_id_filter=source_id,
        )

        system_prompt, user_prompt = assemble_document_prompt(question, retrieval_result)

        llm = get_llm_client()
        for token in llm.stream(system_prompt, user_prompt):
            yield token

        citations_payload = [
            {
                "source_name": c.source_name,
                "page_number": c.page_number,
                "excerpt": c.excerpt,
            }
            for c in retrieval_result.citations
        ]
        yield json.dumps({"type": "citations", "citations": citations_payload})
