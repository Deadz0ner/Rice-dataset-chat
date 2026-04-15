from __future__ import annotations


class EmbeddingService:
    """Create vector embeddings for dataset documents and queries.

    This layer maps text into numerical vectors so semantically similar
    questions and rows can be matched in vector search.

    Why it matters in RAG:
    Embeddings are the foundation for retrieval quality. Better embeddings
    usually mean more relevant dataset rows are surfaced.

    What to implement next:
    - Pick an embedding provider.
    - Batch document embedding calls.
    - Normalize vectors consistently for indexing and search.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider

    def create_embeddings(self, documents: list[str]) -> list[list[float]]:
        """Generate embeddings for dataset chunks.

        Example input:
        - ["Variety: IR64 | Moisture: 12", "Variety: Basmati | Moisture: 10"]

        Example output:
        - [[0.12, 0.44, ...], [0.30, 0.08, ...]]
        """
        # TODO: Call your embedding model here and return document vectors.
        return []

    def embed_query(self, query: str) -> list[float]:
        """Generate a single embedding vector for a user question."""
        # TODO: Use the same embedding model/settings as document indexing.
        return []
