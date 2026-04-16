from __future__ import annotations

import logging
import re
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

    def auto_detect_dataset(self) -> None:
        """Load the first Excel file found in data_dir, if any."""
        data_dir = Path(self.settings.data_dir)
        if not data_dir.is_dir():
            return
        for ext in ("*.xlsx", "*.xls"):
            files = sorted(data_dir.glob(ext))
            if files:
                logger.info("Auto-detected dataset: %s", files[0])
                self.load_dataframe(files[0])
                return

    def _clear_data_dir(self) -> None:
        """Remove all Excel files from data_dir."""
        data_dir = Path(self.settings.data_dir)
        if not data_dir.is_dir():
            return
        for f in data_dir.iterdir():
            if f.suffix.lower() in (".xlsx", ".xls"):
                f.unlink()
                logger.info("Removed old dataset: %s", f.name)

    async def save_and_load_upload(self, file: UploadFile) -> DatasetLoadResponse:
        self.ensure_data_dir()
        self._clear_data_dir()
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
        dataframe = self._normalize_and_clean(dataframe)
        self._dataframe = dataframe
        self._file_name = file_path.name
        logger.info(
            "Dataset loaded with %s rows and %s columns",
            len(dataframe),
            len(dataframe.columns),
        )

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """'Product Description' → 'product_description', 'Value_FC' → 'value_fc'."""
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    def _normalize_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Normalize column names to snake_case
        df.columns = [self._normalize_column_name(c) for c in df.columns]

        # 2. Drop the serial-number column ('s') — it's just a spreadsheet index
        if "s" in df.columns:
            df = df.drop(columns=["s"])

        # 3. Strip whitespace from string columns and collapse internal whitespace
        for col in df.select_dtypes(include="object").columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"_x000D_", "", regex=False)  # Excel carriage-return artefact
            )

        # 4. Fill missing string values with empty string, numeric with 0
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].replace("nan", "").fillna("")
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(0)

        # 5. Add a stable row_id that survives re-loads
        df = df.reset_index(drop=True)
        df.insert(0, "row_id", ["row-" + str(i) for i in range(len(df))])

        # 6. Deduplicate identical rows (keep first, preserve row_id)
        subset = [c for c in df.columns if c != "row_id"]
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        df["row_id"] = ["row-" + str(i) for i in range(len(df))]
        after = len(df)
        if before != after:
            logger.info("Dropped %d duplicate rows", before - after)

        return df

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
