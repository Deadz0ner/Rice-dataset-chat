from __future__ import annotations

from typing import Any


class VectorStoreService:
    """Abstract vector storage facade for FAISS or Chroma.

    This layer owns vector indexing and similarity search while keeping the
    rest of the app independent from the chosen vector database.

    Why it matters in RAG:
    The vector store determines how efficiently and accurately the app can
    find relevant rows from the dataset.

    What to implement next:
    - Instantiate FAISS or Chroma clients.
    - Persist indexed documents.
    - Return top-k matches with metadata and scores.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        """Store vectors plus source documents for later retrieval.

        Example input:
        - documents: [{"row_id": "row-1", "content": "...", "metadata": {...}}]
        - vectors: [[0.1, 0.2, 0.3]]
        """
        # TODO: Build the actual FAISS or Chroma indexing flow.
        return None

    def retrieve_relevant_rows(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant dataset rows for a user query.

        Example output:
        - [{"row_id": "row-9", "content": "...", "metadata": {...}, "score": 0.91}]
        """
        # TODO: Search the index and apply any filtering thresholds.
        return []
