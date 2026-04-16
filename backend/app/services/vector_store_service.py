"""Vector store service — FAISS-backed index for similarity search over document embeddings.

Why FAISS:
    FAISS (Facebook AI Similarity Search) is the standard library for
    dense-vector nearest-neighbour search.  It runs entirely in-process
    with no external server, supports millions of vectors on a single
    machine, and integrates natively with numpy — a natural fit for a
    local RAG pipeline.

Why ``IndexFlatIP`` (inner product):
    Our ``EmbeddingService`` L2-normalizes every vector before returning it.
    For unit-length vectors, the inner product equals cosine similarity::

        cos(a, b)  =  (a · b) / (||a|| · ||b||)  =  a · b     when ||a|| = ||b|| = 1

    ``IndexFlatIP`` performs an *exact* brute-force inner-product search
    with no quantization or approximation, which is ideal for our dataset
    size (~15k vectors × 384 dims).  Scores range from −1 to 1 and can be
    read directly as cosine similarity.

Why positional mapping matters:
    FAISS stores raw float arrays — it knows nothing about ``row_id``,
    ``content``, or ``metadata``.  When a search returns "the vector at
    position 42 scored 0.71", this service must map position 42 back to
    the original chunk document so the pipeline can cite the correct
    dataset row.  The ``_documents`` list preserved at index-build time
    provides that mapping: ``_documents[42]`` is the chunk dict that
    produced the vector at FAISS index position 42.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreService:
    """FAISS-backed vector store for document retrieval.

    Lifecycle:
        1. ``build_index(documents, embeddings)`` — create the FAISS index
           and store the parallel document list for positional lookup.
        2. ``search(query_vector, top_k)`` — find the closest vectors and
           return the corresponding chunk documents with scores.

    The pipeline also uses the legacy names ``index_documents`` and
    ``retrieve_relevant_rows`` which delegate to the above.
    """

    def __init__(self, backend: str = "faiss") -> None:
        self.backend = backend
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Create a FAISS inner-product index from pre-computed embeddings.

        Args:
            documents: Ordered chunk dicts (``{row_id, content, metadata}``).
                       Position ``i`` must correspond to ``embeddings[i]``.
            embeddings: Numpy array of shape ``(N, dim)``, L2-normalized.
        """
        if embeddings.ndim != 2 or len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Document count ({len(documents)}) must match "
                f"embedding row count ({embeddings.shape[0] if embeddings.ndim == 2 else '?'})"
            )

        dim = embeddings.shape[1]

        logger.info(
            "Building FAISS IndexFlatIP — %d vectors × %d dimensions",
            len(documents),
            dim,
        )
        start = time.perf_counter()

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        elapsed = time.perf_counter() - start

        self._index = index
        self._documents = documents

        logger.info(
            "FAISS index built in %.3fs — total vectors: %d",
            elapsed,
            self._index.ntotal,
        )

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Pipeline-compatible alias for ``build_index``."""
        self.build_index(documents, embeddings)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the top-k documents most similar to a query vector.

        Args:
            query_vector: 1-D numpy array of shape ``(dim,)``, L2-normalized.
            top_k: Number of results to return.

        Returns:
            A list of dicts, each containing the original chunk fields
            (``row_id``, ``content``, ``metadata``) plus a ``score`` key
            (cosine similarity, higher is better).  Ordered by descending
            score.
        """
        if self._index is None or not self._documents:
            logger.warning("search called before index was built")
            return []

        # FAISS expects a 2-D query matrix of shape (n_queries, dim)
        query_matrix = query_vector.reshape(1, -1).astype(np.float32)

        start = time.perf_counter()
        scores, indices = self._index.search(query_matrix, top_k)
        elapsed = time.perf_counter() - start

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for unfilled slots when index has fewer
                # vectors than top_k
                continue
            doc = self._documents[idx]
            results.append({
                "row_id": doc["row_id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "score": float(score),
            })

        logger.info(
            "FAISS search completed in %.4fs — top-%d results, best score: %.4f",
            elapsed,
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results

    def retrieve_relevant_rows(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Pipeline-compatible alias for ``search``.

        The ``query`` string argument is accepted for interface compatibility
        with the existing ``rag_pipeline.py`` call site but is not used —
        the actual search runs on ``query_vector``.
        """
        return self.search(query_vector, top_k=top_k)
