"""Prompt grounding service — builds constrained LLM prompts from retrieved evidence.

Why this layer exists:
    Even with perfect retrieval, a poorly written prompt lets the LLM
    hallucinate, speculate, or answer from its training data instead of
    the provided rows.  The grounding prompt is the final guardrail —
    it explicitly tells the model what evidence it has, what it may say,
    and when it must refuse.

Design decisions:
    * **System + user message split.**  The system message sets behavioural
      rules (grounding, refusal, tone) once.  The user message carries the
      evidence and question.  This leverages how most chat-completion APIs
      weight system instructions.
    * **Numbered evidence rows.**  Each row is formatted as ``[1] field: value
      | field: value`` so the LLM can cite specific rows by number and the
      output stays auditable.
    * **Explicit refusal instruction.**  If the evidence does not contain
      enough information, the model must say so rather than guess.  This is
      the core anti-hallucination mechanism.
    * **Structured answer request.**  The prompt asks for concise, factual
      answers — no filler, no opinions, no external knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a data analyst assistant. You answer questions about an Indian rice \
export/import (EXIM) dataset strictly using the EVIDENCE ROWS provided below.

Rules:
1. Use ONLY the evidence rows to answer. Do NOT use outside knowledge.
2. If the evidence rows do not contain enough information to answer the \
question, say: "The provided data does not contain enough information to \
answer this question."
3. When citing numbers (quantities, values, rates), use the exact figures \
from the evidence rows.
4. Keep answers concise, factual, and structured. Use bullet points or \
short paragraphs where appropriate.
5. If the user asks for aggregations (totals, averages, counts) that require \
data beyond the retrieved rows, note that your answer is based on the top \
matching rows, not the full dataset.
6. Never invent exporter names, countries, quantities, or any other data \
point not present in the evidence.\
"""


def _format_evidence_row(index: int, row: dict[str, Any]) -> str:
    """Format a single retrieved row as a numbered evidence block.

    Uses the ``content`` field (the semantic text from the chunking layer)
    and appends the similarity score so the LLM can gauge relevance.
    """
    content = row.get("content", "")
    score = row.get("score")
    score_str = f" (relevance: {score:.2f})" if score is not None else ""
    row_id = row.get("row_id", f"row-{index}")
    return f"[{index + 1}] {row_id}{score_str}\n{content}"


def build_grounded_messages(
    query: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build chat-completion messages that constrain the LLM to retrieved evidence.

    Returns a list of ``{"role": ..., "content": ...}`` dicts suitable for
    any OpenAI-compatible chat completion API.

    Args:
        query: The user's natural-language question.
        rows: Retrieved document dicts from ``VectorStoreService.search()``,
              each containing ``row_id``, ``content``, ``metadata``, ``score``.

    Returns:
        A two-element list: system message (grounding rules) and user message
        (evidence + question).
    """
    evidence_blocks = [_format_evidence_row(i, row) for i, row in enumerate(rows)]
    evidence_section = "\n\n".join(evidence_blocks)

    user_content = f"""\
EVIDENCE ROWS ({len(rows)} rows retrieved):
---
{evidence_section}
---

QUESTION: {query}

Answer based strictly on the evidence rows above.\
"""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    total_len = sum(len(m["content"]) for m in messages)
    logger.info(
        "Grounded prompt built — %d evidence rows, query: '%s', total length: %d chars",
        len(rows),
        query[:80],
        total_len,
    )

    return messages


def build_grounded_prompt(
    query: str,
    rows: list[dict[str, Any]],
) -> str:
    """Legacy single-string wrapper around ``build_grounded_messages``.

    Kept for backward compatibility with code that expects a flat string.
    """
    messages = build_grounded_messages(query, rows)
    return "\n\n".join(m["content"] for m in messages)
