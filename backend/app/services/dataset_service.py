from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from app.core.config import Settings
from app.schemas.dataset import DatasetLoadResponse, DatasetSummaryResponse

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._dataframe: pd.DataFrame | None = None
        self._file_name: str | None = None

    def ensure_data_dir(self) -> None:
        Path(self.settings.data_dir).mkdir(parents=True, exist_ok=True)

    def preload_default_dataset(self, dataset_path: str) -> None:
        path = Path(dataset_path)
        if not path.exists():
            logger.warning("Default dataset path does not exist: %s", dataset_path)
            return
        self.load_dataframe(path)

    async def save_and_load_upload(self, file: UploadFile) -> DatasetLoadResponse:
        self.ensure_data_dir()
        destination = Path(self.settings.data_dir) / file.filename
        content = await file.read()
        destination.write_bytes(content)
        self.load_dataframe(destination)
        return DatasetLoadResponse(
            message="Dataset uploaded and loaded successfully.",
            summary=self.get_summary_or_raise(),
        )

    def load_dataframe(self, file_path: Path) -> None:
        logger.info("Loading dataset from %s", file_path)
        dataframe = pd.read_excel(file_path)
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        self._dataframe = dataframe
        self._file_name = file_path.name
        logger.info("Dataset loaded with %s rows and %s columns", len(dataframe), len(dataframe.columns))

    def get_dataframe(self) -> pd.DataFrame | None:
        return self._dataframe

    def get_summary(self) -> DatasetSummaryResponse | None:
        if self._dataframe is None or self._file_name is None:
            return None

        sample_rows = (
            self._dataframe.head(3)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        )
        return DatasetSummaryResponse(
            file_name=self._file_name,
            row_count=len(self._dataframe),
            column_count=len(self._dataframe.columns),
            columns=[str(column) for column in self._dataframe.columns.tolist()],
            sample_rows=sample_rows,
        )

    def get_summary_or_raise(self) -> DatasetSummaryResponse:
        summary = self.get_summary()
        if summary is None:
            raise ValueError("No dataset is currently loaded.")
        return summary
