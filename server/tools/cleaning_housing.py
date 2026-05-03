"""二手房领域清洗原语 + LangChain 工具（会话闭包）。"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from server.core.house_schema import STANDARD_COLUMN_KEYS
from server.core.session_store import SessionStore
from server.tools.dataset_profile import build_dataset_profile


def split_delimited_column(
    df: pd.DataFrame,
    source_column: str,
    delimiter: str,
    output_columns: list[str],
) -> pd.DataFrame:
    """按分隔符拆分列，输出列数须与 split 段数一致（不足填 NA，多余截断）。"""
    out = df.copy()
    if source_column not in out.columns:
        raise ValueError(f"列不存在: {source_column}")
    bad = [c for c in output_columns if c not in STANDARD_COLUMN_KEYS]
    if bad:
        raise ValueError(f"输出列名必须是标准字段键之一，无效: {bad}")

    def _split_cell(val: Any) -> list[str]:
        if pd.isna(val):
            return [pd.NA] * len(output_columns)
        parts = str(val).split(delimiter)
        parts = [p.strip() if isinstance(p, str) else p for p in parts]
        while len(parts) < len(output_columns):
            parts.append(pd.NA)
        return parts[: len(output_columns)]

    mat = out[source_column].map(_split_cell)
    arr = pd.DataFrame(mat.tolist(), index=out.index, columns=output_columns)
    for c in output_columns:
        out[c] = arr[c]
    return out


def split_slash_part(
    df: pd.DataFrame,
    source_column: str,
    part_index: int,
    output_column: str,
) -> pd.DataFrame:
    """按 / 取第 part_index 段（0 基），写入 output_column（须为标准键）。"""
    if output_column not in STANDARD_COLUMN_KEYS:
        raise ValueError(f"output_column 须为标准字段键: {output_column}")
    out = df.copy()
    if source_column not in out.columns:
        raise ValueError(f"列不存在: {source_column}")

    def _take(val: Any) -> Any:
        if pd.isna(val):
            return pd.NA
        parts = [p.strip() for p in str(val).split("/") if p.strip()]
        if part_index < 0 or part_index >= len(parts):
            return pd.NA
        return parts[part_index]

    out[output_column] = out[source_column].map(_take)
    return out


_FLOOR_CN_RE = re.compile(r"(低层|中层|高层|低区|中区|高区|一楼顶楼)")


def derive_floor_band_column(df: pd.DataFrame, source_column: str = "floor") -> pd.DataFrame:
    """从楼层文本生成 floor_band：低层/中层/高层/未知。"""
    out = df.copy()
    if source_column not in out.columns:
        return out

    def _band(val: Any) -> str:
        if pd.isna(val):
            return "未知"
        s = str(val).strip()
        if not s:
            return "未知"
        if _FLOOR_CN_RE.search(s):
            m = _FLOOR_CN_RE.search(s)
            assert m is not None
            w = m.group(1)
            if "低" in w:
                return "低层"
            if "中" in w:
                return "中层"
            if "高" in w:
                return "高层"
        slash = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
        if slash:
            cur, tot = int(slash.group(1)), int(slash.group(2))
            if tot <= 0:
                return "未知"
            r = cur / tot
            if r <= 1 / 3:
                return "低层"
            if r <= 2 / 3:
                return "中层"
            return "高层"
        return "未知"

    out["floor_band"] = out[source_column].map(_band)
    return out


_DECORATION_MAP = [
    (["毛坯", "清水"], "毛坯"),
    (["简装", "普通装修"], "简装"),
    (["精装", "精装修"], "精装"),
    (["豪装", "豪华"], "豪装"),
]


def normalize_decoration_column(df: pd.DataFrame, source_column: str, output_column: str = "decoration") -> pd.DataFrame:
    if output_column not in STANDARD_COLUMN_KEYS:
        raise ValueError("output_column 须为标准键")
    out = df.copy()
    if source_column not in out.columns:
        raise ValueError(f"列不存在: {source_column}")

    def _norm(val: Any) -> str:
        if pd.isna(val):
            return "未知"
        s = str(val).strip()
        if not s:
            return "未知"
        for keys, label in _DECORATION_MAP:
            if any(k in s for k in keys):
                return label
        return s[:20]

    out[output_column] = out[source_column].map(_norm)
    return out


_FOLLOWERS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万|千|w|W)?")


def coerce_followers_series(ser: pd.Series) -> pd.Series:
    def _one(val: Any) -> float:
        if pd.isna(val):
            return float("nan")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        s = str(val).strip()
        m = _FOLLOWERS_RE.search(s)
        if not m:
            return float("nan")
        num = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("万", "w"):
            num *= 10_000
        elif unit == "千":
            num *= 1000
        return num

    return ser.map(_one)


def coerce_followers_column(df: pd.DataFrame, source_column: str, output_column: str = "followers") -> pd.DataFrame:
    if output_column not in STANDARD_COLUMN_KEYS:
        raise ValueError("output_column 须为标准键")
    out = df.copy()
    if source_column not in out.columns:
        raise ValueError(f"列不存在: {source_column}")
    out[output_column] = coerce_followers_series(out[source_column])
    return out


def apply_standard_rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """mapping: 当前列名 -> 标准键（值须在 STANDARD_COLUMN_KEYS）。"""
    out = df.copy()
    for old, new in mapping.items():
        if old not in out.columns:
            raise ValueError(f"列不存在: {old}")
        if new not in STANDARD_COLUMN_KEYS:
            raise ValueError(f"目标不是标准字段键: {new}")
    return out.rename(columns=mapping)


def make_housing_cleaning_tools(store: SessionStore, session_id: str):
    def _trace(msg: str) -> None:
        try:
            st = store.require(session_id)
            st.cleaning_trace.append(msg)
        except Exception:
            pass

    @tool
    def get_dataset_profile() -> str:
        """返回当前清洗表的 JSON 画像（列名、缺失率、样例值、| 与 / 出现比例启发）。请先调用以决定拆分与映射策略。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return json.dumps({"error": "清洗表为空"}, ensure_ascii=False)
        prof = build_dataset_profile(st.df_clean)
        _trace("get_dataset_profile")
        return json.dumps(prof, ensure_ascii=False)

    @tool
    def split_delimited_column_tool(source_column: str, delimiter: str, output_columns_json: str) -> str:
        """
        将 source_column 按 delimiter（如 |）拆成多列。
        output_columns_json 为标准字段键的 JSON 数组，如 ["district","layout"]。
        """
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        cols = json.loads(output_columns_json)
        if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
            return "错误：output_columns_json 须为字符串数组。"
        st.df_clean = split_delimited_column(st.df_clean, source_column, delimiter, cols)
        _trace(f"split_delimited_column {source_column} -> {cols}")
        return f"已拆分 {source_column}，新建列: {cols}"

    @tool
    def split_slash_field_tool(source_column: str, part_index: int, output_column: str) -> str:
        """从 source_column 按 / 取第 part_index 段（从 0 开始），写入 output_column（标准键）。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        st.df_clean = split_slash_part(st.df_clean, source_column, int(part_index), output_column)
        _trace(f"split_slash {source_column}[{part_index}] -> {output_column}")
        return f"已写入 {output_column}"

    @tool
    def derive_floor_band_tool(source_column: str = "floor") -> str:
        """根据楼层列（如 18/33 或含 低/中/高）生成 floor_band。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        if source_column not in st.df_clean.columns:
            return f"错误：无列 {source_column}"
        st.df_clean = derive_floor_band_column(st.df_clean, source_column)
        _trace("derive_floor_band")
        return "已生成 floor_band"

    @tool
    def normalize_decoration_tool(source_column: str, output_column: str = "decoration") -> str:
        """将装修描述规范为 毛坯/简装/精装/豪装/未知 等。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        st.df_clean = normalize_decoration_column(st.df_clean, source_column, output_column)
        _trace("normalize_decoration")
        return f"已规范化装修 -> {output_column}"

    @tool
    def coerce_followers_tool(source_column: str, output_column: str = "followers") -> str:
        """关注人数文本（含 万/千）转数值。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        st.df_clean = coerce_followers_column(st.df_clean, source_column, output_column)
        _trace("coerce_followers")
        return f"已数值化 -> {output_column}"

    @tool
    def apply_column_rename_tool(mapping_json: str) -> str:
        """
        列重命名到标准键。mapping_json 形如 {"原始列名":"district","挂牌价":"total_price"}，
        键须为当前表列名，值须为标准字段键。
        """
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        mapping = json.loads(mapping_json)
        if not isinstance(mapping, dict):
            return "错误：须为 JSON 对象。"
        st.df_clean = apply_standard_rename(st.df_clean, {str(k): str(v) for k, v in mapping.items()})
        _trace(f"rename {mapping}")
        return f"已重命名 {len(mapping)} 列"

    return [
        get_dataset_profile,
        split_delimited_column_tool,
        split_slash_field_tool,
        derive_floor_band_tool,
        normalize_decoration_tool,
        coerce_followers_tool,
        apply_column_rename_tool,
    ]
