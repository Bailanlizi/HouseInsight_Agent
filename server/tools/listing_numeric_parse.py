"""从「153万」「17190元/平米」「89平米」等文本中解析数值，供清洗收尾与分析前统一加固。"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_UNIT_PRICE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万)?\s*(?:元)?\s*(?:/)?\s*(?:㎡|平米|m²|m2|M2)",
    re.IGNORECASE,
)
_AREA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:㎡|平米|m²|m2|M2)?", re.IGNORECASE)
_TOTAL_WAN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*万", re.IGNORECASE)
_SIMPLE_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _scalar_float(x: Any) -> float:
    if pd.isna(x):
        return float("nan")
    if isinstance(x, bool):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return float("nan")
    m = _SIMPLE_FLOAT_RE.search(s.replace(",", ""))
    return float(m.group(1)) if m else float("nan")


def parse_unit_price_yuan(val: Any) -> float:
    """单价统一为元/㎡。"""
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v = float(val)
        if 50 < v < 800:
            return v * 10_000
        return v
    s = str(val).strip().replace(",", "")
    if not s:
        return float("nan")
    m = _UNIT_PRICE_RE.search(s)
    if m:
        num = float(m.group(1))
        if m.group(2) or ("万" in s and "元" in s and "/" in s):
            return num * 10_000
        return num
    if "万" in s and ("㎡" in s or "平米" in s or "m²" in s.lower()):
        m2 = _SIMPLE_FLOAT_RE.search(s)
        if m2:
            return float(m2.group(1)) * 10_000
    return _scalar_float(s)


def parse_total_price_wan(val: Any) -> float:
    """总价统一为万元。"""
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v = float(val)
        if v > 50_000:
            return v / 10_000
        return v
    s = str(val).strip().replace(",", "")
    if not s:
        return float("nan")
    m = _TOTAL_WAN_RE.search(s)
    if m:
        return float(m.group(1))
    if "万" in s:
        m2 = _SIMPLE_FLOAT_RE.search(s)
        if m2:
            return float(m2.group(1))
    v = _scalar_float(s)
    if not pd.isna(v) and v > 50_000:
        return v / 10_000
    return v


def parse_area_m2(val: Any) -> float:
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).strip().replace(",", "")
    if not s:
        return float("nan")
    m = _AREA_RE.search(s)
    if m:
        return float(m.group(1))
    return _scalar_float(s)


_STAGING_PROMOTE: tuple[tuple[str, str], ...] = (
    ("area_m2_str", "area_m2"),
    ("area_m2_text", "area_m2"),
    ("layout_str", "layout"),
    ("orientation_str", "orientation"),
    ("floor_text", "floor"),
)


def promote_staging_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, tgt in _STAGING_PROMOTE:
        if src not in out.columns:
            continue
        if tgt not in out.columns:
            out[tgt] = out[src]
            continue
        tcol = out[tgt]
        mask = tcol.isna() | (tcol.astype(str).str.strip() == "") | (tcol.astype(str).str.strip().str.lower() == "nan")
        out.loc[mask, tgt] = out.loc[mask, src]
    return out


def parse_object_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsers: dict[str, Any] = {
        "unit_price": parse_unit_price_yuan,
        "total_price": parse_total_price_wan,
        "area_m2": parse_area_m2,
    }
    for col, parser in parsers.items():
        if col not in out.columns:
            continue
        ser = out[col]
        if ser.dtype == object or pd.api.types.is_string_dtype(ser):
            out[col] = ser.map(parser)
    return out


def finalize_listing_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """拆分后的过渡列合并 + 文本中的万/元/㎡解析 + 数值化 + 缺失单价推算。"""
    from server.tools.cleaning import coerce_numeric_columns, fill_unit_price_from_total_and_area

    work = promote_staging_columns(df)
    work = parse_object_numeric_columns(work)
    work = coerce_numeric_columns(work)
    work = fill_unit_price_from_total_and_area(work)
    return work
