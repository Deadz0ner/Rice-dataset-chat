from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_pipeline
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_pipeline import RAGPipelineService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    rag_pipeline: RAGPipelineService = Depends(get_rag_pipeline),
) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message must not be empty.")

    return rag_pipeline.answer_query(payload)
