"""清洗后数据质量评估（规则为主）+ 可选 LLM 生成重试建议（多 Agent 协作中的「质检/教练」）。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from server.core.config import Settings

# 行数相对原始表丢失超过该比例时，阻塞并触发回到 clean 重试
_ROW_RETENTION_BLOCK_THRESHOLD = 0.5

# 规则标签：近地铁 True 占比低于该值时仅警告
_TAG_NEAR_SUBWAY_WARN_THRESHOLD = 0.3


def _ratio_non_null(ser: pd.Series) -> float:
    if len(ser) == 0:
        return 0.0
    return float(ser.notna().mean())


def _unit_price_valid_ratio(df: pd.DataFrame) -> float:
    if "unit_price" not in df.columns or len(df) == 0:
        return 0.0
    s = pd.to_numeric(df["unit_price"], errors="coerce")
    # 成都等：单价多在几千～几万/㎡；过低可能是误标为「万/㎡」的小数，交由前置解析修正
    ok = s.notna() & (s > 30) & (s < 600_000)
    return float(ok.mean())


def _meaningful_str_mask(ser: pd.Series) -> pd.Series:
    s = ser.fillna("").astype(str).str.strip()
    return s.ne("") & ~s.str.lower().isin(("nan", "none", "null", "待定"))


def _geo_ratio(df: pd.DataFrame) -> float:
    """district 或 community 至少其一有有效文本的比例。"""
    if len(df) == 0:
        return 0.0
    d = df["district"] if "district" in df.columns else pd.Series([""] * len(df), index=df.index)
    c = df["community"] if "community" in df.columns else pd.Series([""] * len(df), index=df.index)
    either = _meaningful_str_mask(d) | _meaningful_str_mask(c)
    return float(either.mean())


def assess_clean_quality(
    df_raw: pd.DataFrame | None,
    df_clean: pd.DataFrame | None,
    settings: Settings,
) -> dict[str, Any]:
    """
    返回 dict：passed、failures（仅阻塞项）、hints_zh（阻塞说明）、warnings_zh / warning_codes（不阻塞）、metrics。
    阻塞：clean_empty；或 n_raw>=50 且行保留率 < 0.5（丢失过半）。其余为警告，不触发重洗。
    """
    failures: list[str] = []
    hints: list[str] = []
    warnings: list[str] = []
    warning_codes: list[str] = []

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
        return _finalize(False, metrics, failures, hints, warnings, warning_codes)

    retention = (n_clean / n_raw) if n_raw > 0 else 1.0
    metrics["row_retention_ratio"] = round(retention, 4)

    min_rows = settings.quality_min_rows
    min_ret = settings.quality_min_retention_ratio
    min_up = settings.quality_min_unit_price_coverage
    min_geo = settings.quality_min_geo_coverage

    if n_raw >= 50 and retention < _ROW_RETENTION_BLOCK_THRESHOLD:
        failures.append("row_retention_critical")
        hints.append(
            f"清洗后行数仅占原始的 {retention:.1%}（低于 50% 阈值），疑似误去重或过滤过猛；"
            "将触发重试清洗。请检查去重子集与 IQR 规则。"
        )
    elif n_raw >= 50 and retention < min_ret:
        warning_codes.append("row_retention_low")
        warnings.append(
            f"清洗后行数仅占原始的 {retention:.1%}，低于配置的保留率期望 {min_ret:.1%}，但不触发自动重洗；"
            "若分析结果异常可手动检查去重与过滤。"
        )

    if n_clean < min_rows and n_raw >= min_rows:
        warning_codes.append("too_few_rows")
        warnings.append(
            f"清洗后有效行数 {n_clean} 低于建议阈值 {min_rows}；不触发自动重洗，统计可能波动较大。"
        )

    up_r = _unit_price_valid_ratio(df_clean)
    metrics["unit_price_valid_ratio"] = round(up_r, 4)

    tp_ok = _ratio_non_null(df_clean["total_price"]) if "total_price" in df_clean.columns else 0.0
    ar_ok = _ratio_non_null(df_clean["area_m2"]) if "area_m2" in df_clean.columns else 0.0
    metrics["total_price_non_null_ratio"] = round(tp_ok, 4)
    metrics["area_m2_non_null_ratio"] = round(ar_ok, 4)

    derivable = tp_ok >= 0.2 and ar_ok >= 0.2
    price_ok = up_r >= min_up or (tp_ok >= 0.25 and ar_ok >= 0.25) or derivable
    if not price_ok:
        warning_codes.append("price_fields_weak")
        warnings.append(
            "单价有效占比偏低且总价+建面不足以可靠推算单价；不触发自动重洗。"
            "可检查「153万」「元/㎡」类文本是否映射到 total_price/unit_price/area_m2。"
        )
    elif up_r < 0.5:
        warning_codes.append("unit_price_sparse")
        warnings.append(
            f"单价列有效占比约 {up_r:.1%}，超过半数行缺失或无效；建议检查解析规则（仅警告）。"
        )

    geo_r = _geo_ratio(df_clean)
    metrics["geo_non_null_ratio"] = round(geo_r, 4)
    if geo_r < min_geo:
        warning_codes.append("geo_sparse")
        warnings.append(
            "城区或小区信息有效占比低于配置阈值；不触发自动重洗。"
            "可从复合列拆分或列映射到 district/community。"
        )

    if "tag_near_subway" in df_clean.columns and len(df_clean) > 0:
        tr = float(df_clean["tag_near_subway"].astype(bool).mean())
        metrics["tag_near_subway_true_ratio"] = round(tr, 4)
        if tr < _TAG_NEAR_SUBWAY_WARN_THRESHOLD:
            warning_codes.append("tag_near_subway_sparse")
            warnings.append(
                f"近地铁规则标签为 True 的占比约 {tr:.1%}，低于 {_TAG_NEAR_SUBWAY_WARN_THRESHOLD:.0%} 提示阈值；"
                "多为描述未写地铁或规则未覆盖，不阻塞流水线。"
            )

    if "description_raw" in df_clean.columns and len(df_clean) > 0:
        dr = float(_meaningful_str_mask(df_clean["description_raw"]).mean())
        metrics["description_raw_non_empty_ratio"] = round(dr, 4)

    passed = len(failures) == 0
    return _finalize(passed, metrics, failures, hints, warnings, warning_codes)


def _finalize(
    passed: bool,
    metrics: dict[str, Any],
    failures: list[str],
    hints: list[str],
    warnings: list[str],
    warning_codes: list[str],
) -> dict[str, Any]:
    metrics["failure_codes"] = failures.copy()
    metrics["warning_codes"] = warning_codes.copy()
    return {
        "passed": passed,
        "metrics": metrics,
        "failures": failures,
        "hints_zh": hints,
        "warnings_zh": warnings,
        "warning_codes": warning_codes,
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
    """质检未通过时：规则要点 + 可选 LLM 补充（教练 Agent）。仅针对阻塞项 failures。"""
    hints = list(report.get("hints_zh") or [])
    warns = list(report.get("warnings_zh") or [])
    rule_block = _bullet_block(hints)
    if report.get("passed"):
        if warns:
            return ("【质检警告（不阻塞）】\n" + _bullet_block(warns)).strip()
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
            "warnings": warns,
        }
        msg = (
            "你是二手房数据清洗教练。上轮清洗未通过质检（仅行保留率过低等阻塞项）。"
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
