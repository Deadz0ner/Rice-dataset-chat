from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedRow:
    row_id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None
