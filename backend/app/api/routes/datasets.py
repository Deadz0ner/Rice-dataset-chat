from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_dataset_service
from app.schemas.dataset import DatasetLoadResponse, DatasetSummaryResponse
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetLoadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetLoadResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File name is required.")

    return await dataset_service.save_and_load_upload(file)


@router.get("/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary(
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetSummaryResponse:
    summary = dataset_service.get_summary()
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No dataset is currently loaded.")
    return summary
