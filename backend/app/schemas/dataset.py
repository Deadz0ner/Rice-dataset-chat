from pydantic import BaseModel, Field


class DatasetSummaryResponse(BaseModel):
    file_name: str
    row_count: int
    column_count: int
    columns: list[str]
    sample_rows: list[dict[str, str]] = Field(default_factory=list)


class DatasetLoadResponse(BaseModel):
    message: str
    summary: DatasetSummaryResponse
