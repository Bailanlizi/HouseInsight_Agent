"""规则文本特征（无 LLM）：与确定性清洗分离，供 LangGraph 独立节点或工具链调用。"""

from __future__ import annotations

import pandas as pd

from server.tools.listing_numeric_parse import finalize_listing_dataframe
from server.tools.text_label_features import apply_text_label_features


def apply_rule_text_features(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    写入 tag_* / layout_normalized / build_year 文本回填，再经 finalize 统一数值化与单价推算。
    调用方应传入 df 的副本；返回新 DataFrame 与一行说明便于写入 cleaning_notes。
    """
    if df.empty:
        return df, "规则特征：表为空，已跳过"
    work = apply_text_label_features(df.copy())
    work = finalize_listing_dataframe(work)
    note = "规则文本特征已写入（含 finalize：过渡列晋升、数值化、单价推算）"
    return work, note
