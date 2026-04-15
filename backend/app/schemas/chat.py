from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question in natural language.")


class SourceRow(BaseModel):
    row_id: str
    preview: str
    score: float | None = None
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = Field(
        ...,
        description="Signals whether the answer is based on retrieved dataset evidence.",
    )
    sources: list[SourceRow] = Field(default_factory=list)
    note: str | None = Field(
        default=None,
        description="Optional explanation when the system cannot answer from available data.",
    )
