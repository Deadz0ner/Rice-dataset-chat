from __future__ import annotations

import logging

from app.models.document import RetrievedRow
from app.schemas.chat import ChatRequest, ChatResponse, SourceRow
from app.services.chunking_service import chunk_excel_rows
from app.services.dataset_service import DatasetService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.prompt_service import build_grounded_prompt
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class RAGPipelineService:
    def __init__(self, dataset_service: DatasetService, vector_store_backend: str, llm_provider: str) -> None:
        self.dataset_service = dataset_service
        self.embedding_service = EmbeddingService(provider=llm_provider)
        self.vector_store = VectorStoreService(backend=vector_store_backend)
        self.llm_service = LLMService(provider=llm_provider)

    def answer_query(self, payload: ChatRequest) -> ChatResponse:
        dataframe = self.dataset_service.get_dataframe()
        if dataframe is None:
            return ChatResponse(
                answer="No dataset is loaded yet.",
                grounded=False,
                note="Upload an Excel file before asking questions.",
            )

        documents = chunk_excel_rows(dataframe)
        if not documents:
            logger.info("Chunking placeholder returned no documents.")
            return ChatResponse(
                answer=(
                    "The dataset is loaded, but the retrieval pipeline is still a scaffold. "
                    "Implement chunking, embeddings, indexing, and retrieval to answer questions."
                ),
                grounded=False,
                note="RAG placeholder logic has not been implemented yet.",
            )

        document_vectors = self.embedding_service.create_embeddings(
            [document["content"] for document in documents]
        )
        self.vector_store.index_documents(documents, document_vectors)

        query_vector = self.embedding_service.embed_query(payload.message)
        retrieved_rows = self.vector_store.retrieve_relevant_rows(payload.message, query_vector)

        if not retrieved_rows:
            return ChatResponse(
                answer="I could not find enough grounded evidence in the dataset to answer that question.",
                grounded=False,
                note="Implement retrieval thresholds and fallback rules in the RAG pipeline.",
            )

        prompt = build_grounded_prompt(payload.message, retrieved_rows)
        answer = self.llm_service.generate_grounded_response(prompt)
        sources = [
            SourceRow(
                row_id=row["row_id"],
                preview=row["content"],
                score=row.get("score"),
                metadata=row.get("metadata", {}),
            )
            for row in retrieved_rows
        ]
        return ChatResponse(answer=answer, grounded=True, sources=sources)

    @staticmethod
    def coerce_rows(rows: list[dict[str, object]]) -> list[RetrievedRow]:
        return [
            RetrievedRow(
                row_id=str(row["row_id"]),
                content=str(row["content"]),
                metadata=dict(row.get("metadata", {})),
                score=float(row["score"]) if row.get("score") is not None else None,
            )
            for row in rows
        ]
