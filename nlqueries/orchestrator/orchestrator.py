"""
nlqueries.orchestrator.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single-agent orchestrator: loads a knowledge base, assembles the LLM prompt,
and streams response tokens back to the caller.

Public API
----------
``Orchestrator``
    Call ``handle_question(question, agent_id)`` to get an async token stream.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

import yaml

from nlqueries import config
from nlqueries.llm import get_llm_client
from nlqueries.orchestrator.prompt_assembly import assemble_prompt


class Orchestrator:
    """Single-agent orchestrator for NLQueries.

    Each invocation of ``handle_question`` independently loads the agent's
    YAML knowledge base, assembles an LLM prompt via ``assemble_prompt``,
    and streams response tokens from the configured ``LLMClient``.

    Multi-agent routing is out of scope for v1 and will be added in Phase 2
    (Sprint 12).
    """

    async def handle_question(
        self,
        question: str,
        agent_id: str,
    ) -> AsyncGenerator[str, None]:
        """Translate *question* into a token stream from the LLM.

        Loads the YAML knowledge base for *agent_id*, assembles a grounding
        prompt, calls ``LLMClient.stream()``, and yields each response token
        in the order it arrives.

        Args:
            question: The natural-language question from the user.
            agent_id: Identifier of the agent whose knowledge base to use.
                      Must match a file under ``config.KB_PATH``.

        Yields:
            String tokens from the LLM response, in arrival order.

        Raises:
            FileNotFoundError: When no knowledge base file exists for
                               *agent_id*.
        """
        kb = self._load_knowledge_base(agent_id)
        collection = f"agent_{agent_id}_schema"
        system_prompt, user_prompt = assemble_prompt(
            question,
            kb,
            top_k_capsules=5,
            collection=collection,
        )
        llm = get_llm_client()
        for token in llm.stream(system_prompt, user_prompt):
            yield token

    def _load_knowledge_base(self, agent_id: str) -> dict[str, Any]:
        """Load and return the YAML knowledge base for *agent_id*.

        Sanitises *agent_id* to a safe filename stem (colons and slashes
        become underscores) then reads
        ``{config.KB_PATH}/{safe_agent_id}.yaml``.

        Raises:
            FileNotFoundError: When the knowledge base file does not exist.
        """
        safe_id = re.sub(r"[^\w.-]", "_", agent_id)
        kb_file = config.KB_PATH / f"{safe_id}.yaml"
        if not kb_file.exists():
            raise FileNotFoundError(
                f"No knowledge base found for agent '{agent_id}'. "
                f"Run 'nlqueries export-kb {agent_id}' first."
            )
        return yaml.safe_load(kb_file.read_text(encoding="utf-8")) or {}
