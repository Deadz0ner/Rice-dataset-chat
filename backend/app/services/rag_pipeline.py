from __future__ import annotations

import logging
import threading

from app.schemas.chat import ChatRequest, ChatResponse, SourceRow
from app.services.chunking_service import chunk_excel_rows
from app.services.dataset_service import DatasetService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.prompt_service import build_grounded_messages
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class RAGPipelineService:
    def __init__(self, dataset_service: DatasetService, vector_store_backend: str, llm_provider: str) -> None:
        self.dataset_service = dataset_service
        self.embedding_service = EmbeddingService(provider=llm_provider)
        self.vector_store = VectorStoreService(backend=vector_store_backend)
        self.llm_service = LLMService(provider=llm_provider)
        self._indexed = False
        self._index_lock = threading.Lock()

    def _ensure_index(self) -> bool:
        """Build the vector index from the current dataset if not already built.

        Returns True if the index is ready, False if no data is available.
        Thread-safe: a lock prevents concurrent builds from the background
        warm-up and an incoming query.
        """
        if self._indexed:
            return True

        with self._index_lock:
            if self._indexed:
                return True

            dataframe = self.dataset_service.get_dataframe()
            if dataframe is None:
                return False

            documents = chunk_excel_rows(dataframe)
            if not documents:
                return False

            vectors = self.embedding_service.create_embeddings(
                [doc["content"] for doc in documents]
            )
            self.vector_store.build_index(documents, vectors)
            self._indexed = True
            logger.info("RAG index built — %d documents indexed", len(documents))
            return True

    def warm_up(self) -> None:
        """Pre-build the index in the background so the first query is fast."""
        logger.info("Background warm-up started")
        try:
            ready = self._ensure_index()
            if ready:
                logger.info("Background warm-up complete — index is ready")
            else:
                logger.info("Background warm-up skipped — no dataset loaded yet")
        except Exception:
            logger.exception("Background warm-up failed — first query will retry")

    def invalidate_index(self) -> None:
        """Call after a new dataset is uploaded to force re-indexing."""
        with self._index_lock:
            self._indexed = False

    def answer_query(self, payload: ChatRequest) -> ChatResponse:
        if not self._ensure_index():
            return ChatResponse(
                answer="No dataset is loaded yet. Please upload an Excel file first.",
                grounded=False,
            )

        query_vector = self.embedding_service.embed_query(payload.message)
        retrieved_rows = self.vector_store.search(query_vector, top_k=5)

        if not retrieved_rows:
            return ChatResponse(
                answer="Not found in provided dataset.",
                grounded=False,
            )

        messages = build_grounded_messages(payload.message, retrieved_rows)
        answer = self.llm_service.generate_grounded_response(messages)

        sources = [
            SourceRow(
                row_id=row["row_id"],
                preview=row["content"][:200],
                score=row.get("score"),
                metadata=row.get("metadata", {}),
            )
            for row in retrieved_rows
        ]

        return ChatResponse(answer=answer, grounded=True, sources=sources)
