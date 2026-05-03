"""二手房表结构化画像（供 Agent 与规划节点使用）。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_dataset_profile(df: pd.DataFrame, sample_per_col: int = 3) -> dict[str, Any]:
    if df.empty:
        return {"row_count": 0, "columns": []}
    rows: list[dict[str, Any]] = []
    n = len(df)
    for col in df.columns:
        ser = df[col]
        nn = int(ser.notna().sum())
        ratio = round(nn / n, 4) if n else 0.0
        samples = [str(x)[:120] for x in ser.dropna().head(sample_per_col).tolist()]
        pipe_hint: float | None = None
        slash_hint: float | None = None
        if ser.dtype == object or pd.api.types.is_string_dtype(ser):
            s_str = ser.dropna().astype(str)
            if len(s_str) > 0:
                pipe_hint = float(s_str.str.contains(r"\|", regex=True, na=False).mean())
                slash_hint = float(s_str.str.contains(r"/", regex=True, na=False).mean())
        rows.append(
            {
                "name": str(col),
                "dtype": str(ser.dtype),
                "non_null_ratio": ratio,
                "sample_values": samples,
                "pipe_like_ratio": pipe_hint,
                "slash_like_ratio": slash_hint,
            }
        )
    return {"row_count": n, "columns": rows}
