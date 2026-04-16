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

_SYSTEM_PROMPT_TEMPLATE = """\
You are a data analyst assistant. You answer questions about an Indian rice \
export/import (EXIM) dataset.

HOW THIS SYSTEM WORKS:
The full dataset has {total_rows:,} rows. A search system retrieved the \
{evidence_count} rows below as the closest matches to the user's question. \
You can only see these {evidence_count} rows, not the other {remaining:,}.

YOU MUST HANDLE THREE SITUATIONS DIFFERENTLY:

SITUATION A — The user asks about specific shipments, exporters, dates, or \
transactions AND your evidence rows contain matching data.
→ Answer confidently and in detail from the evidence. No disclaimers needed. \
This is what you are built for.

SITUATION B — The user asks about specific shipments, dates, or transactions \
BUT your evidence rows do NOT contain a match (wrong dates, wrong countries, \
wrong products).
→ Say: "I searched the dataset but didn't find a record matching [what they \
asked for]. The closest records I found are about [briefly describe what the \
rows actually contain]. Try rephrasing your question or asking about a \
different date/exporter/product."
Do NOT blame the 5-row limit. The issue is that the search didn't find a match.

SITUATION C — The user asks for rankings, totals, counts, "most", "least", \
"top", "how many", "compare", or any aggregation across the whole dataset.
→ Say: "This question needs a count/ranking/total across all {total_rows:,} \
rows in the dataset, but I can only search for the most relevant rows — I \
can't scan and aggregate the full dataset. To answer this accurately, you \
would need a direct database query or analytics tool."

SITUATION D — The user asks a random question which has nothing to do with this rice dataset \
tell him that you are a data analyst assistant built to answer questions about an Indian rice export/import (EXIM) dataset, and that you don't have the knowledge to answer his question.
dont hallucinate, or make up things, or dont even tell the answer even if you know it.
example question: If the user say hi, or greet you, or ask about the weather, or ask about sports, or ask about politics, or ask about movies, or ask about music, or ask about anything that is not related to the Indian rice export/import (EXIM) dataset, you should respond with "I am a data analyst assistant built to answer questions about an Indian rice export/import (EXIM) dataset. I don't have the knowledge to answer your question."

OTHER RULES:
- Use ONLY the evidence rows. No outside knowledge.
- Cite exact figures from the rows.
- Keep answers concise, factual, and structured.
- Never invent data not in the evidence.\
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
    total_rows: int = 0,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat-completion messages that constrain the LLM to retrieved evidence.

    Args:
        query: The user's natural-language question.
        rows: Retrieved document dicts from ``VectorStoreService.search()``.
        total_rows: Total number of rows in the full dataset so the LLM
                    understands how small its evidence window is.
        history: Previous conversation turns as ``[{role, content}, ...]``.

    Returns:
        A list of messages: system, prior conversation turns, then the
        current user message with evidence.
    """
    evidence_count = len(rows)
    remaining = max(0, total_rows - evidence_count)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        total_rows=total_rows,
        evidence_count=evidence_count,
        remaining=remaining,
    )

    evidence_blocks = [_format_evidence_row(i, row) for i, row in enumerate(rows)]
    evidence_section = "\n\n".join(evidence_blocks)

    user_content = f"""\
EVIDENCE ROWS ({evidence_count} retrieved from {total_rows:,} total):
---
{evidence_section}
---

QUESTION: {query}

Decide which situation applies and respond accordingly. Do NOT mention \
"situation A/B/C" in your response — just answer naturally.\
"""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Include conversation history so the LLM can resolve references
    # like "the ones", "that exporter", "show me more", etc.
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_content})

    total_len = sum(len(m["content"]) for m in messages)
    logger.info(
        "Grounded prompt built — %d/%d evidence rows, %d history turns, query: '%s', length: %d chars",
        evidence_count,
        total_rows,
        len(history) if history else 0,
        query[:80],
        total_len,
    )

    return messages
