from __future__ import annotations

from typing import Any


def build_grounded_prompt(query: str, rows: list[dict[str, Any]]) -> str:
    """Build a prompt that forces the LLM to answer only from retrieved rows.

    This layer packages the question and evidence into a constrained prompt
    that teaches the model to stay inside dataset-backed facts.

    Why it matters in RAG:
    Even with good retrieval, weak prompt grounding can still produce
    overconfident or unsupported answers.

    What to implement next:
    - Format evidence rows clearly.
    - Add refusal instructions when evidence is insufficient.
    - Tune for structured, concise answers.
    """
    # TODO: Compose a grounding prompt with retrieved row evidence only.
    return (
        "You are answering questions about a rice dataset. "
        "Use only the retrieved evidence rows. "
        f"Question: {query} | Evidence rows: {rows}"
    )
