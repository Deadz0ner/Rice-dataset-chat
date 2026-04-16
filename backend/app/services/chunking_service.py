"""Chunking service — converts cleaned DataFrame rows into semantic text documents.

Why this layer exists in a RAG pipeline:
    Embedding models work on text, not tabular data.  A raw dataframe row is a
    bag of typed cells — an embedding model has no idea what ``100.0`` in a
    ``quantity`` column means unless we tell it.  By serializing each row into a
    human-readable sentence like ``Quantity: 100.0 MTS | Foreign Country: IRAN``
    we give the embedding model the *semantic labels* it needs to place the
    document in the right region of vector space.

    Poor chunking is the #1 silent killer of retrieval quality — if the text
    chunk is noisy or missing key facts, no amount of better embeddings or
    smarter prompts will recover them.

Design decisions:
    * **One row = one chunk.**  The dataset is already tabular with independent
      export records so there is no benefit to grouping rows.
    * **Tiered importance.**  Address fields (``exporter_address1``,
      ``exporter_address2``, ``exporter_city``) are stored in ``metadata`` only,
      NOT embedded in the ``content`` text.  Raw address strings like
      ``"VILLAGE CHABBA TARN TARAN ROAD, 143001"`` dilute the embedding vector
      for the vast majority of business queries.  Location-specific queries
      (``"exporters from Karnal"``) are handled at the retrieval layer via
      metadata filtering, not via vector similarity against address noise.
    * **Skip empty values.**  Including ``Exporter City:`` with no value adds
      noise tokens that dilute the embedding.
    * **Human-readable labels.**  ``product_description`` becomes
      ``Product Description`` so the embedding model (trained on natural text)
      can leverage the semantics of the label itself.
    * **Pipe-delimited serialization.**  A simple ``|`` separator keeps chunks
      parseable by both humans and models without introducing ambiguous
      punctuation.
    * **Framework-agnostic.**  No LangChain / LlamaIndex dependency — just
      pandas and plain dicts so the rest of the pipeline stays swappable.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column → human-readable label map (embedded in content)
# ---------------------------------------------------------------------------
# Keys are the snake_case column names produced by DatasetService.
# Values are the labels that appear in the serialized chunk text.
# Ordering here controls the serialization order — most semantically
# important fields first so the embedding model front-loads them.
_COLUMN_LABELS: dict[str, str] = {
    "product_description": "Product Description",
    "exporter": "Exporter",
    "buyer": "Buyer",
    "foreign_country": "Foreign Country",
    "foreign_port": "Foreign Port",
    "indian_port": "Indian Port",
    "quantity": "Quantity",
    "unit": "Unit",
    "rate_fc": "Rate FC",
    "value_fc": "Value FC",
    "fob": "FOB",
    "rate": "Rate",
    "currency": "Currency",
    "mode": "Mode",
    "date": "Date",
    "month": "Month",
    "year": "Year",
    "iec": "IEC",
    "bill_number": "Bill Number",
}

# ---------------------------------------------------------------------------
# Metadata-only columns (stored but NOT embedded)
# ---------------------------------------------------------------------------
# Address fields are kept in metadata for citation and metadata-filtering at
# the retrieval layer.  They are deliberately excluded from the content string
# because raw address text dilutes embedding vectors for business queries.
_METADATA_ONLY_COLUMNS: set[str] = {
    "exporter_address1",
    "exporter_address2",
    "exporter_city",
}


def _is_empty(value: object) -> bool:
    """Return True if the value carries no useful information."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "nan"


def _serialize_row(row: pd.Series) -> str:
    """Convert a single DataFrame row into a pipe-delimited labeled string.

    Only columns present in ``_COLUMN_LABELS`` are serialized, and only when
    the value is non-empty.  This keeps chunks tight and noise-free.

    Example output::

        Product Description: 1121 EXTRA LONG BASMATI SELLA RICE |
        Exporter: SUPPLE TEK INDUSTRIES PRIVATE LIMITED |
        Foreign Country: IRAN | Quantity: 100.0 | Unit: MTS
    """
    parts: list[str] = []
    for col, label in _COLUMN_LABELS.items():
        if col not in row.index:
            continue
        value = row[col]
        if _is_empty(value):
            continue
        parts.append(f"{label}: {value}")
    return " | ".join(parts)


def chunk_excel_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a cleaned DataFrame into a list of retrieval-ready documents.

    Each document is a dict with:
        * ``row_id``  — stable identifier carried over from DatasetService.
        * ``content`` — the semantic text chunk that will be embedded.
        * ``metadata`` — the original column values for citation / filtering.

    Args:
        df: A pandas DataFrame already cleaned by ``DatasetService``.

    Returns:
        A list of document dicts ready for the embedding layer.
    """
    documents: list[dict[str, Any]] = []
    skipped = 0

    logger.info("Chunking %d dataframe rows into text documents", len(df))

    for _, row in df.iterrows():
        text = _serialize_row(row)
        if not text:
            skipped += 1
            continue

        row_id = str(row.get("row_id", f"row-{len(documents)}"))

        metadata: dict[str, Any] = {}
        for col in row.index:
            if col == "row_id":
                continue
            val = row[col]
            if not _is_empty(val):
                metadata[col] = val

        documents.append({
            "row_id": row_id,
            "content": text,
            "metadata": metadata,
        })

    avg_len = sum(len(d["content"]) for d in documents) / len(documents) if documents else 0
    metadata_only_count = sum(
        1 for d in documents
        if any(k in d["metadata"] for k in _METADATA_ONLY_COLUMNS)
    )
    logger.info(
        "Chunking complete: %d documents produced, %d empty rows skipped, "
        "avg content length %.0f chars, %d documents carry address metadata",
        len(documents),
        skipped,
        avg_len,
        metadata_only_count,
    )
    if documents:
        logger.debug("Sample chunk [%s]: %s", documents[0]["row_id"], documents[0]["content"][:200])

    return documents
