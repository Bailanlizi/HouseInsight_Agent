"""二手房挂牌表一键确定性 ETL（L1 数值 + L2 复合列规则），不调用 LLM。"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _stderr_handler = logging.StreamHandler()
    _stderr_handler.setFormatter(logging.Formatter("%(levelname)s [listing_etl] %(message)s"))
    logger.addHandler(_stderr_handler)
logger.propagate = False


from server.tools.cleaning import (
    coerce_numeric_columns,
    drop_exact_duplicates,
    fill_unit_price_from_total_and_area,
    remove_price_outliers_iqr,
    resolve_listing_dedup_subset,
)
from server.tools.cleaning_housing import derive_floor_band_column
from server.tools.composite_field_parse import expand_composite_listing_columns
from server.tools.listing_numeric_parse import parse_object_numeric_columns, promote_staging_columns


def run_listing_etl(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    确定性清洗：复合列展开 → 过渡列晋升 → 文本数值解析 → 数值化 → 单价推算
    → 去重 → IQR → 楼层档（不含规则文本标签；标签见 pipeline.rule_feature_engineering）。
    """
    notes: list[str] = []
    before = len(df)
    logger.info(f"[input] rows: {before}")

    stage_counts: list[tuple[str, int]] = [("input", before)]

    work = expand_composite_listing_columns(df.copy())
    stage_counts.append(("expand", len(work)))
    logger.info(f"[expand] rows: {len(work)}")

    work = promote_staging_columns(work)
    stage_counts.append(("promote", len(work)))
    logger.info(f"[promote] rows: {len(work)}")

    work = parse_object_numeric_columns(work)
    stage_counts.append(("parse_object_numeric", len(work)))
    logger.info(f"[parse_object_numeric] rows: {len(work)}")

    work = coerce_numeric_columns(work)
    stage_counts.append(("coerce", len(work)))
    logger.info(f"[coerce] rows: {len(work)}")

    work = fill_unit_price_from_total_and_area(work)
    stage_counts.append(("fill_unit_price", len(work)))
    logger.info(f"[fill_unit_price] rows: {len(work)}")

    subset = resolve_listing_dedup_subset(work)
    work = drop_exact_duplicates(work, subset=subset)
    stage_counts.append(("dedup", len(work)))
    logger.info(f"[dedup] rows: {len(work)} subset={subset}")
    notes.append(f"去重后行数: {len(work)} (原始 {before})")

    work = remove_price_outliers_iqr(work)
    stage_counts.append(("iqr", len(work)))
    logger.info(f"[iqr] rows: {len(work)}")
    notes.append(f"IQR 单价异常过滤后行数: {len(work)}")

    if "floor" in work.columns:
        work = derive_floor_band_column(work, "floor")
        stage_counts.append(("floor_band", len(work)))
        logger.info(f"[floor_band] rows: {len(work)}")

    max_drop = 0
    max_drop_stage = ""
    for i in range(1, len(stage_counts)):
        prev_n, cur_n = stage_counts[i - 1][1], stage_counts[i][1]
        drop = prev_n - cur_n
        if drop > max_drop:
            max_drop = drop
            max_drop_stage = stage_counts[i][0]
    if max_drop > 0:
        logger.info(f"[row_drop_max] step={max_drop_stage} rows_lost={max_drop}")
        notes.append(f"行数下降最多: {max_drop_stage}（减少 {max_drop} 行）")

    return work, "; ".join(notes)
