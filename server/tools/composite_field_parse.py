"""从 house_info_raw、follow_info_raw 等复合列确定性抽取结构化字段（不依赖 LLM）。"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from server.core.house_schema import STANDARD_COLUMN_KEYS
from server.tools.cleaning_housing import coerce_followers_series

_LAYOUT_RE = re.compile(r"\d+\s*室\s*\d+\s*厅|\d+室\d+厅")
_AREA_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:平米|㎡|m²|m2|M2)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
_ORIENTS = (
    "南北通透",
    "东南",
    "西南",
    "东北",
    "西北",
    "南北",
    "东西",
    "朝南",
    "朝北",
    "朝东",
    "朝西",
    "南",
    "北",
    "东",
    "西",
)
_DECO_KEYS = (
    (("精装", "精装修"), "精装"),
    (("简装", "普通装修"), "简装"),
    (("毛坯", "清水"), "毛坯"),
    (("豪装", "豪华"), "豪装"),
)
_BUILDING_TYPES = ("塔板结合", "塔板", "板楼", "塔楼", "钢混", "砖混", "别墅", "洋房")


def _parse_house_info_cell(val: Any) -> dict[str, str]:
    acc: dict[str, str] = {}
    if pd.isna(val):
        return acc
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return acc
    parts = [x.strip() for x in s.split("|") if x.strip()]
    for p in parts:
        if "layout_str" not in acc and _LAYOUT_RE.search(p):
            m = _LAYOUT_RE.search(p)
            assert m is not None
            acc["layout_str"] = re.sub(r"\s+", "", m.group(0))
            continue
        if "area_m2_str" not in acc and _AREA_NUM_RE.search(p):
            am = _AREA_NUM_RE.search(p)
            assert am is not None
            acc["area_m2_str"] = f"{am.group(1)}平米"
            continue
        if "orientation_str" not in acc:
            hit = False
            for o in _ORIENTS:
                if o in p and len(p) <= 16:
                    acc["orientation_str"] = o
                    hit = True
                    break
            if hit:
                continue
        if "decoration_str" not in acc:
            hit = False
            for keys, label in _DECO_KEYS:
                if any(k in p for k in keys):
                    acc["decoration_str"] = label
                    hit = True
                    break
            if hit:
                continue
        if "build_year" not in acc:
            ym = _YEAR_RE.search(p)
            if ym and ("年" in p or "建" in p or "建成" in p):
                acc["build_year"] = ym.group(1)
                continue
            if ym and len(p) <= 8:
                acc["build_year"] = ym.group(1)
                continue
        if "building_type_str" not in acc:
            hit = False
            for bt in _BUILDING_TYPES:
                if bt in p:
                    acc["building_type_str"] = bt
                    hit = True
                    break
            if hit:
                continue
        if "floor_text" not in acc and ("层" in p or "楼" in p or "共" in p):
            acc["floor_text"] = p
    return acc


def _follow_time_text(val: Any) -> Any:
    if pd.isna(val):
        return pd.NA
    s = str(val).strip().replace("／", "/")
    if "/" in s:
        tail = s.split("/", 1)[1].strip()
        return tail if tail else pd.NA
    if "|" in s:
        tail = s.split("|", 1)[1].strip()
        return tail if tail else pd.NA
    return pd.NA


def _is_blank_series_col(ser: pd.Series) -> pd.Series:
    if ser.dtype == object or pd.api.types.is_string_dtype(ser):
        t = ser.astype(str).str.strip()
        return ser.isna() | (t == "") | (t.str.lower() == "nan")
    return ser.isna()


def expand_composite_listing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 house_info_raw（| 分隔）与 follow_info_raw（关注/时间）抽取标准列；仅填补目标列中的空位，不覆盖已有值。
    """
    out = df.copy()
    if "house_info_raw" in out.columns:
        parsed = out["house_info_raw"].map(_parse_house_info_cell)
        keys_seen: set[str] = set()
        for row in parsed:
            if isinstance(row, dict):
                keys_seen.update(row.keys())
        for key in keys_seen:
            if key not in STANDARD_COLUMN_KEYS:
                continue
            series = parsed.map(lambda d, k=key: d.get(k, pd.NA) if isinstance(d, dict) else pd.NA)
            if key not in out.columns:
                out[key] = series
            else:
                mask = _is_blank_series_col(out[key])
                out.loc[mask, key] = series[mask]

    if "follow_info_raw" in out.columns:
        fol = coerce_followers_series(out["follow_info_raw"])
        time_txt = out["follow_info_raw"].map(_follow_time_text)
        if "followers" not in out.columns:
            out["followers"] = fol
        else:
            m = out["followers"].isna()
            if out["followers"].dtype == object or pd.api.types.is_string_dtype(out["followers"]):
                m = m | _is_blank_series_col(out["followers"])
            out.loc[m, "followers"] = fol[m]
        for tcol in ("publish_time_raw", "listing_time"):
            if tcol not in STANDARD_COLUMN_KEYS:
                continue
            if tcol not in out.columns:
                out[tcol] = time_txt
            else:
                m2 = _is_blank_series_col(out[tcol])
                out.loc[m2, tcol] = time_txt[m2]
    return out
