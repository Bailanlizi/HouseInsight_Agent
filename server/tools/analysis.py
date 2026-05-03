from __future__ import annotations

from typing import Any

import pandas as pd


def analyze_second_hand_listings(df: pd.DataFrame) -> dict[str, Any]:
    """二手房固定维度分析（非通用）。"""
    out: dict[str, Any] = {"row_count": int(len(df)), "columns": list(df.columns)}
    if df.empty:
        return out

    if "district" in df.columns:
        g = df.groupby("district", dropna=False)
        out["district_summary"] = (
            g.agg(listings=("district", "count"), avg_unit_price=("unit_price", "mean"))
            .reset_index()
            .fillna("")
            .to_dict(orient="records")
        )

    if "unit_price" in df.columns:
        s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
        if len(s):
            out["unit_price_quantiles"] = {
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.5)),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }

    if "area_m2" in df.columns:
        s = pd.to_numeric(df["area_m2"], errors="coerce").dropna()
        if len(s):
            bins = [0, 60, 90, 120, 150, 10_000]
            labels = ["<=60", "60-90", "90-120", "120-150", ">150"]
            cats = pd.cut(s, bins=bins, labels=labels, right=True)
            vc = cats.value_counts().reindex(labels).fillna(0).astype(int)
            out["area_buckets"] = {str(k): int(v) for k, v in vc.items()}

    if {"build_year", "unit_price"}.issubset(df.columns):
        sub = df[["build_year", "unit_price"]].copy()
        sub["build_year"] = pd.to_numeric(sub["build_year"], errors="coerce")
        sub["unit_price"] = pd.to_numeric(sub["unit_price"], errors="coerce")
        sub = sub.dropna()
        if len(sub) >= 5:
            sub["decade"] = (sub["build_year"] // 10 * 10).astype(int)
            gg = sub.groupby("decade")["unit_price"].mean().reset_index()
            out["decade_avg_unit_price"] = gg.rename(
                columns={"decade": "decade_start", "unit_price": "avg_unit_price"}
            ).to_dict(orient="records")

    return out
