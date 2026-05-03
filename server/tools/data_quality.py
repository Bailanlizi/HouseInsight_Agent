"""清洗后数据质量评估（规则为主）+ 可选 LLM 生成重试建议（多 Agent 协作中的「质检/教练」）。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from server.core.config import Settings


def _ratio_non_null(ser: pd.Series) -> float:
    if len(ser) == 0:
        return 0.0
    return float(ser.notna().mean())


def _unit_price_valid_ratio(df: pd.DataFrame) -> float:
    if "unit_price" not in df.columns or len(df) == 0:
        return 0.0
    s = pd.to_numeric(df["unit_price"], errors="coerce")
    ok = s.notna() & (s > 100) & (s < 500_000)
    return float(ok.mean())


def _geo_ratio(df: pd.DataFrame) -> float:
    """district 或 community 至少其一有值的比例。"""
    if len(df) == 0:
        return 0.0
    d = df["district"] if "district" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    c = df["community"] if "community" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    either = d.notna() | c.notna()
    return float(either.mean())


def assess_clean_quality(
    df_raw: pd.DataFrame | None,
    df_clean: pd.DataFrame | None,
    settings: Settings,
) -> dict[str, Any]:
    """
    返回 dict：passed, metrics, failures, hints_zh（给清洗 Agent 的中文要点）。
    """
    failures: list[str] = []
    hints: list[str] = []

    n_raw = int(len(df_raw)) if df_raw is not None else 0
    n_clean = int(len(df_clean)) if df_clean is not None else 0

    metrics: dict[str, Any] = {
        "raw_rows": n_raw,
        "clean_rows": n_clean,
        "clean_columns": list(df_clean.columns) if df_clean is not None else [],
    }

    if df_clean is None or df_clean.empty:
        failures.append("clean_empty")
        hints.append("清洗结果为空：请从原始表重新载入（reload_raw_dataframe），检查拆分与列映射是否正确。")
        return _finalize(False, metrics, failures, hints)

    retention = (n_clean / n_raw) if n_raw > 0 else 1.0
    metrics["row_retention_ratio"] = round(retention, 4)

    min_rows = settings.quality_min_rows
    min_ret = settings.quality_min_retention_ratio
    min_up = settings.quality_min_unit_price_coverage
    min_geo = settings.quality_min_geo_coverage

    if n_raw >= 50 and retention < min_ret:
        failures.append("row_retention_low")
        hints.append(
            f"清洗后行数仅占原始的 {retention:.1%}，疑似误去重或过滤过猛；"
            "避免仅用单列去重，优先保留 listing_id / house_info_raw / description_raw 等区分列。"
        )

    if n_clean < min_rows and n_raw >= min_rows:
        failures.append("too_few_rows")
        hints.append(f"清洗后有效行数 {n_clean} 低于阈值 {min_rows}，难以做统计分析；请减轻去重或异常值过滤。")

    up_r = _unit_price_valid_ratio(df_clean)
    metrics["unit_price_valid_ratio"] = round(up_r, 4)

    tp_ok = _ratio_non_null(df_clean["total_price"]) if "total_price" in df_clean.columns else 0.0
    ar_ok = _ratio_non_null(df_clean["area_m2"]) if "area_m2" in df_clean.columns else 0.0
    metrics["total_price_non_null_ratio"] = round(tp_ok, 4)
    metrics["area_m2_non_null_ratio"] = round(ar_ok, 4)

    price_ok = up_r >= min_up or (tp_ok >= 0.25 and ar_ok >= 0.25)
    if not price_ok:
        failures.append("price_fields_weak")
        hints.append(
            "单价有效占比过低，且总价+建筑面积不足以推算单价；"
            "请数值化 total_price/unit_price/area_m2，或用总面积与总价填补缺失单价。"
        )

    geo_r = _geo_ratio(df_clean)
    metrics["geo_non_null_ratio"] = round(geo_r, 4)
    if geo_r < min_geo:
        failures.append("geo_sparse")
        hints.append("城区或小区信息缺失过多；请从复合列拆分或 apply_column_rename 映射到 district/community。")

    passed = len(failures) == 0
    return _finalize(passed, metrics, failures, hints)


def _finalize(passed: bool, metrics: dict[str, Any], failures: list[str], hints: list[str]) -> dict[str, Any]:
    metrics["failure_codes"] = failures.copy()
    return {
        "passed": passed,
        "metrics": metrics,
        "failures": failures,
        "hints_zh": hints,
    }


def _bullet_block(lines: list[str]) -> str:
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        out.append(s if s.lstrip().startswith("-") else f"- {s}")
    return "\n".join(out)


def coach_clean_retry_hints(settings: Settings, report: dict[str, Any]) -> str:
    """质检未通过时：规则要点 + 可选 LLM 补充（教练 Agent）。"""
    hints = list(report.get("hints_zh") or [])
    rule_block = _bullet_block(hints)
    if report.get("passed"):
        return rule_block
    if not settings.dashscope_api_key:
        return rule_block
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.houseinsight_llm_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0.1,
        )
        payload = {
            "passed": report.get("passed"),
            "failures": report.get("failures"),
            "metrics": report.get("metrics"),
            "rule_hints": hints,
        }
        msg = (
            "你是二手房数据清洗教练。上轮清洗未通过质检。"
            "根据下列 JSON，用 2～5 条简短中文列出下一步应用工具的具体建议。"
            "不要编造表中不存在的列名；不要输出 markdown。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        resp = llm.invoke([HumanMessage(content=msg)])
        extra = getattr(resp, "content", str(resp)).strip()
        if not extra:
            return rule_block
        return rule_block + "\n\n【模型补充建议】\n" + _bullet_block(extra.split("\n"))
    except Exception:
        return rule_block
