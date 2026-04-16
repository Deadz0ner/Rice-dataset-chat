"""Embedding service — converts semantic text chunks into dense numerical vectors.

How text becomes a vector:
    A sentence-transformer model reads a text string (e.g. ``"Product Description:
    BASMATI RICE | Foreign Country: IRAN | Quantity: 100.0"``) and compresses its
    meaning into a fixed-length list of floats (e.g. 384 dimensions for
    all-MiniLM-L6-v2).  Texts that are *semantically similar* end up as vectors
    that are *geometrically close* in that 384-dimensional space — so a user
    question about "basmati exports to Iran" lands near chunks that mention
    basmati and Iran, even if the exact words differ.

Why consistent ordering matters:
    The vector store (FAISS, next layer) maps each vector to a document by
    *positional index* — vector[0] belongs to document[0], vector[1] to
    document[1], etc.  If the embedding layer reordered, shuffled, or skipped
    documents, the vector store would return the wrong rows for a query.
    This service guarantees that output vectors are in the exact same order
    as the input document list.

Why this layer is separate from vector indexing:
    Embedding and indexing are independent concerns.  Keeping them separate
    means you can swap the embedding model (MiniLM → OpenAI → Cohere) without
    touching the FAISS code, or swap the vector store (FAISS → Chroma → Pinecone)
    without re-implementing embedding logic.  It also makes each layer
    independently testable.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Local embedding service backed by sentence-transformers.

    The model is loaded lazily on first use (``load_model``) so application
    startup stays fast when embeddings aren't needed yet (e.g. health checks).
    """

    def __init__(self, provider: str, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self.provider = provider
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def load_model(self) -> SentenceTransformer:
        """Load the sentence-transformer model into memory.

        Subsequent calls return the already-loaded model (singleton pattern).
        The first call downloads the model if it isn't cached locally.
        Uses local_files_only when the model is already cached to avoid
        slow HuggingFace network round-trips on every startup.
        """
        if self._model is not None:
            return self._model

        logger.info("Loading embedding model '%s'...", self.model_name)
        start = time.perf_counter()
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:
            logger.info("Model not cached locally — downloading from HuggingFace")
            self._model = SentenceTransformer(self.model_name)
        elapsed = time.perf_counter() - start
        logger.info(
            "Embedding model loaded in %.2fs — dimension: %d",
            elapsed,
            self._model.get_embedding_dimension(),
        )
        return self._model

    def create_embeddings(self, documents: list[str]) -> np.ndarray:
        """Generate embeddings for a list of document content strings.

        Args:
            documents: Ordered list of text chunks (one per dataset row).

        Returns:
            A numpy array of shape ``(len(documents), embedding_dim)`` where
            ``embedding_dim`` is 384 for all-MiniLM-L6-v2.  Row order is
            preserved — ``result[i]`` is the embedding for ``documents[i]``.
        """
        if not documents:
            logger.warning("create_embeddings called with empty document list")
            return np.array([])

        model = self.load_model()

        logger.info("Embedding %d documents...", len(documents))
        start = time.perf_counter()
        vectors = model.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "Embedded %d documents in %.2fs — output shape: %s",
            len(documents),
            elapsed,
            vectors.shape,
        )
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a single embedding vector for a user question.

        Uses the same model and normalization as ``create_embeddings`` so that
        query vectors and document vectors live in the same vector space.

        Args:
            query: The user's natural-language question.

        Returns:
            A 1-D numpy array of shape ``(embedding_dim,)``.
        """
        model = self.load_model()

        logger.info("Embedding query: '%s'", query[:100])
        start = time.perf_counter()
        vector = model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.perf_counter() - start
        logger.info("Query embedded in %.4fs — dimension: %d", elapsed, vector.shape[0])

        return vector
