from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, datasets, health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.dataset_service import DatasetService
from app.services.rag_pipeline import RAGPipelineService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    dataset_service = DatasetService(settings=settings)
    rag_pipeline = RAGPipelineService(
        dataset_service=dataset_service,
        vector_store_backend=settings.vector_backend,
        llm_provider=settings.llm_provider,
    )

    app.state.settings = settings
    app.state.dataset_service = dataset_service
    app.state.rag_pipeline = rag_pipeline

    dataset_service.ensure_data_dir()
    if settings.default_dataset_path:
        dataset_service.preload_default_dataset(settings.default_dataset_path)

    yield


def create_application() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(datasets.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    return app


app = create_application()
