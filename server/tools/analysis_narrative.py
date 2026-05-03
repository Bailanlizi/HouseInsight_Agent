"""基于结构化分析结果生成「分析师观点」（独立 LLM 调用，与规划/清洗 Agent 分工）。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from server.core.config import Settings


def enrich_analysis_markdown(settings: Settings, df: pd.DataFrame, analysis: dict[str, Any]) -> str:
    base = (analysis.get("analysis_summary_markdown") or "").strip()
    if not settings.dashscope_api_key or df.empty:
        return base or "（暂无图表摘要）"

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        snippet: dict[str, Any] = {
            "rows": len(df),
            "columns": list(df.columns),
            "district_summary_sample": (analysis.get("district_summary") or [])[:8],
            "tasks_completed": [
                k
                for k, v in (analysis.get("task_results") or {}).items()
                if isinstance(v, dict) and v.get("ok")
            ],
        }
        if "unit_price" in df.columns:
            s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
            if len(s) > 5:
                snippet["unit_price_summary"] = {
                    "median": float(s.median()),
                    "p25": float(s.quantile(0.25)),
                    "p75": float(s.quantile(0.75)),
                }

        model = settings.houseinsight_plan_model or settings.houseinsight_llm_model
        llm = ChatOpenAI(
            model=model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0.35,
        )
        prompt = (
            "你是资深二手房市场分析师。根据下列数据摘要，撰写「分析师观点」。\n"
            "要求：4～6 条短句，有对比、有判断（如区域价差、主力户型/面积段、单价区间）；\n"
            "不要编造摘要中不存在的事实；字段不足时如实说明；可用「-」列表，不要 markdown 标题。\n\n"
            + json.dumps(snippet, ensure_ascii=False)
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        voice = getattr(resp, "content", str(resp)).strip()
        if not voice:
            return base
        return (base + "\n\n### 分析师观点\n\n" + voice).strip()
    except Exception:
        return base
