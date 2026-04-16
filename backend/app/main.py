import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, datasets, health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.dataset_service import DatasetService
from app.services.rag_pipeline import RAGPipelineService

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


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
    else:
        dataset_service.auto_detect_dataset()

    threading.Thread(target=rag_pipeline.warm_up, daemon=True).start()

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

    # Serve the built React frontend in production
    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file = STATIC_DIR / full_path
            if file.is_file():
                return FileResponse(file)
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_application()
