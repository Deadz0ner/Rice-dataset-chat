from __future__ import annotations

import pandas as pd


def stringify_dataframe_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)
