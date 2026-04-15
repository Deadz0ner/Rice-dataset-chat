from fastapi import Request

from app.services.dataset_service import DatasetService
from app.services.rag_pipeline import RAGPipelineService


def get_dataset_service(request: Request) -> DatasetService:
    return request.app.state.dataset_service


def get_rag_pipeline(request: Request) -> RAGPipelineService:
    return request.app.state.rag_pipeline
