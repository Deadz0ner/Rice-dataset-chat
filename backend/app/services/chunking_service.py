from __future__ import annotations

from typing import Any

import pandas as pd


def chunk_excel_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe rows into retrieval-ready documents.

    This layer transforms tabular rows into text-rich chunks that can later
    be embedded and searched semantically.

    Why it matters in RAG:
    Row-to-document conversion decides what evidence the retriever can find.
    Poor chunking often leads to missed facts or noisy retrieval.

    What to implement next:
    - Choose row-wise vs grouped-row chunking.
    - Serialize columns into concise text.
    - Preserve stable row identifiers and metadata.

    Expected input:
    - A pandas DataFrame containing the rice dataset.

    Expected output:
    - A list like:
      [
        {
          "row_id": "row-1",
          "content": "Variety: Basmati | Region: Punjab | Yield: 4.2",
          "metadata": {"Variety": "Basmati", "Region": "Punjab"}
        }
      ]
    """
    # TODO: Turn dataframe rows into semantically meaningful documents.
    return []
