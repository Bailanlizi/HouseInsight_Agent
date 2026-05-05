from __future__ import annotations

import pandas as pd

from server.core.session_store import SessionStore

# 去重子集：至少两列且各列非空比例达标，避免「仅按总价」等单列把数千行压成一行。
# 含原文/标题类列，避免同城同价多套房源被误并为一条。
_PREFERRED_DEDUP_KEYS: tuple[str, ...] = (
    "listing_id",
    "community",
    "layout",
    "area_m2",
    "house_info_raw",
    "listing_title",
    "description_raw",
    "location_raw",
    "follow_info_raw",
    "total_price",
    "unit_price",
    "district",
)


def resolve_listing_dedup_subset(df: pd.DataFrame, min_non_null_ratio: float = 0.05) -> list[str] | None:
    """确定去重列：存在 ingest_file 时与 listing_id 组合，避免多区县 Excel 合并后编号撞车被删。"""
    has_ingest = "ingest_file" in df.columns and bool(df["ingest_file"].notna().any())

    if "listing_id" in df.columns:
        ratio = float(df["listing_id"].notna().mean())
        if ratio >= min_non_null_ratio:
            if has_ingest:
                return ["ingest_file", "listing_id"]
            nu = int(df["listing_id"].nunique(dropna=True))
            n = len(df)
            if n > 0 and nu >= max(1, int(0.92 * n)):
                return ["listing_id"]
    keys = [c for c in _PREFERRED_DEDUP_KEYS if c in df.columns]
    keys = [c for c in keys if float(df[c].notna().mean()) >= min_non_null_ratio]
    if has_ingest:
        keys = ["ingest_file"] + [k for k in keys if k != "ingest_file"]
    if len(keys) >= 2:
        return keys
    return None


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["area_m2", "total_price", "unit_price", "build_year", "followers"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fill_unit_price_from_total_and_area(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not {"total_price", "area_m2"}.issubset(out.columns):
        return out
    # 总价万元 -> 元；单价元/㎡
    total_yuan = out["total_price"] * 10_000
    area = out["area_m2"]
    derived = total_yuan / area.replace(0, pd.NA)
    if "unit_price" not in out.columns:
        out["unit_price"] = derived
    else:
        mask = out["unit_price"].isna()
        out.loc[mask, "unit_price"] = derived[mask]
    return out


def drop_exact_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if subset:
        use = [c for c in subset if c in out.columns]
        if use:
            return out.drop_duplicates(subset=use)
    return out.drop_duplicates()


def remove_price_outliers_iqr(df: pd.DataFrame, col: str = "unit_price", k: float = 1.5) -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col].dropna()
    if len(s) < 10:
        return df
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return df.loc[(df[col].isna()) | ((df[col] >= low) & (df[col] <= high))].copy()


def apply_default_cleaning_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """清洗 + 规则文本特征 + finalize，与 LangGraph 线上一致。"""
    from server.pipeline.listing_etl import run_listing_etl
    from server.pipeline.rule_feature_engineering import apply_rule_text_features

    work, note_etl = run_listing_etl(df)
    work, note_feat = apply_rule_text_features(work)
    return work, "; ".join(x for x in (note_etl, note_feat) if x)


def make_cleaning_tools(store: SessionStore, session_id: str):
    """闭包工具：供 create_agent 调用，读写 SessionStore 中的 df_clean。"""
    from langchain_core.tools import tool

    @tool
    def reload_raw_dataframe() -> str:
        """从会话中的原始合并表重新载入清洗起点（覆盖当前清洗结果）。"""
        st = store.require(session_id)
        if st.df_raw is None:
            return "错误：原始数据不存在，请先运行 ingest。"
        st.df_clean = st.df_raw.copy()
        return f"已重置清洗表，行数={len(st.df_clean)}"

    @tool
    def coerce_house_numeric_columns() -> str:
        """将面积、总价、单价、建筑年代等列转为数值（无法转换的为 NaN）。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        from server.tools.listing_numeric_parse import finalize_listing_dataframe

        st.df_clean = finalize_listing_dataframe(st.df_clean)
        return "已解析「万/元/㎡」文本并数值化、填补单价（如有总价与面积）"

    @tool
    def fill_unit_price_missing() -> str:
        """用总价(万元)与建筑面积推算缺失的单价(元/㎡)。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        st.df_clean = fill_unit_price_from_total_and_area(st.df_clean)
        return "单价缺失填充完成"

    @tool
    def drop_duplicate_listings() -> str:
        """按多列组合去除重复挂牌（至少两列有效；单列不构成子集，退化为仅删完全相同行）。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        subset = resolve_listing_dedup_subset(st.df_clean)
        before = len(st.df_clean)
        st.df_clean = drop_exact_duplicates(st.df_clean, subset=subset)
        return f"去重: {before} -> {len(st.df_clean)}"

    @tool
    def remove_unit_price_outliers_iqr() -> str:
        """对单价使用 IQR 规则剔除极端异常值（保守）。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        before = len(st.df_clean)
        st.df_clean = remove_price_outliers_iqr(st.df_clean)
        return f"IQR 过滤: {before} -> {len(st.df_clean)}"

    @tool
    def run_full_default_clean() -> str:
        """一键执行默认清洗流水线（数值化→单价推算→去重→IQR）。"""
        st = store.require(session_id)
        if st.df_clean is None:
            return "错误：清洗表为空。"
        st.df_clean, note = apply_default_cleaning_pipeline(st.df_clean)
        st.cleaning_notes = note
        return note

    return [
        reload_raw_dataframe,
        coerce_house_numeric_columns,
        fill_unit_price_missing,
        drop_duplicate_listings,
        remove_unit_price_outliers_iqr,
        run_full_default_clean,
    ]
