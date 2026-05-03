"""二手房分析任务白名单：LLM 规划 JSON + 确定性执行。"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from server.core.config import Settings
from server.tools.dataset_profile import build_dataset_profile


class AnalysisTaskType(str, Enum):
    district_price_rank = "district_price_rank"
    layout_price_box = "layout_price_box"
    area_band_price = "area_band_price"
    decoration_price_compare = "decoration_price_compare"
    floor_band_price = "floor_band_price"
    building_age_price_trend = "building_age_price_trend"
    community_followers_rank = "community_followers_rank"
    total_area_scatter = "total_area_scatter"
    price_outlier_flag = "price_outlier_flag"
    unit_price_histogram = "unit_price_histogram"
    area_band_share_pie = "area_band_share_pie"


class PlannedTask(BaseModel):
    type: AnalysisTaskType
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            return AnalysisTaskType(v)
        return v


class AnalysisPlanResponse(BaseModel):
    tasks: list[PlannedTask] = Field(default_factory=list, max_length=8)


_MIN_N = 5


def _need_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def execute_task(df: pd.DataFrame, task: PlannedTask) -> dict[str, Any]:
    t = task.type
    out: dict[str, Any] = {"task": t.value, "ok": False}

    try:
        if t == AnalysisTaskType.district_price_rank:
            if not _need_cols(df, ["district", "unit_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 district+unit_price 且样本足够"
                return out
            sub = df[["district", "unit_price"]].copy()
            sub["unit_price"] = _num(sub["unit_price"])
            sub = sub.dropna()
            agg = (
                sub.groupby("district", dropna=False)["unit_price"]
                .mean()
                .reset_index()
                .sort_values("unit_price", ascending=False)
            )
            agg["unit_price"] = pd.to_numeric(agg["unit_price"], errors="coerce")
            agg = agg.dropna(subset=["unit_price"])
            if agg.empty or len(agg) < 1:
                out["reason"] = "城区单价有效样本不足"
                return out
            out["ok"] = True
            out["chart_kind"] = "bar"
            out["bar_horizontal"] = True
            out["title"] = "各城区二手房均价排行（元/㎡）"
            out["records"] = agg.to_dict(orient="records")
            out["x"] = "district"
            out["y"] = "unit_price"

        elif t == AnalysisTaskType.layout_price_box:
            if not _need_cols(df, ["layout", "unit_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 layout+unit_price"
                return out
            sub = df[["layout", "unit_price"]].copy()
            sub["unit_price"] = _num(sub["unit_price"])
            sub = sub.dropna()
            top = sub["layout"].astype(str).value_counts().head(12).index
            sub = sub[sub["layout"].astype(str).isin(top)]
            out["ok"] = True
            out["chart_kind"] = "box"
            out["title"] = "户型与单价分布（箱线）"
            out["records"] = sub.rename(columns={"layout": "category"}).to_dict(orient="records")

        elif t == AnalysisTaskType.area_band_price:
            if not _need_cols(df, ["area_m2", "unit_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 area_m2+unit_price"
                return out
            sub = df[["area_m2", "unit_price"]].copy()
            sub["area_m2"] = _num(sub["area_m2"])
            sub["unit_price"] = _num(sub["unit_price"])
            sub = sub.dropna()
            bins = [0, 60, 90, 120, 150, 10_000]
            labels = ["<=60", "60-90", "90-120", "120-150", ">150"]
            sub["band"] = pd.cut(sub["area_m2"], bins=bins, labels=labels, right=True)
            gg = sub.groupby("band", observed=True)["unit_price"].mean().reset_index()
            out["ok"] = True
            out["chart_kind"] = "bar"
            out["title"] = "面积段均价（元/㎡）"
            out["records"] = gg.astype({"band": str}).to_dict(orient="records")
            out["x"] = "band"
            out["y"] = "unit_price"

        elif t == AnalysisTaskType.area_band_share_pie:
            if "area_m2" not in df.columns or len(df) < _MIN_N:
                out["reason"] = "需要 area_m2"
                return out
            sub = df[["area_m2"]].copy()
            sub["area_m2"] = _num(sub["area_m2"])
            sub = sub.dropna()
            if len(sub) < _MIN_N:
                out["reason"] = "面积有效样本不足"
                return out
            bins = [0, 60, 90, 120, 150, 10_000]
            labels = ["<=60", "60-90", "90-120", "120-150", ">150"]
            sub["band"] = pd.cut(sub["area_m2"], bins=bins, labels=labels, right=True)
            vc = sub.groupby("band", observed=True).size().reset_index(name="count")
            out["ok"] = True
            out["chart_kind"] = "pie"
            out["title"] = "面积段套数占比"
            out["records"] = vc.astype({"band": str}).to_dict(orient="records")
            out["pie_names"] = "band"
            out["pie_values"] = "count"

        elif t == AnalysisTaskType.decoration_price_compare:
            if not _need_cols(df, ["decoration", "unit_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 decoration+unit_price"
                return out
            sub = df[["decoration", "unit_price"]].copy()
            sub["unit_price"] = _num(sub["unit_price"])
            sub = sub.dropna()
            gg = sub.groupby(sub["decoration"].astype(str), dropna=False)["unit_price"].mean().reset_index()
            gg.columns = ["decoration", "unit_price"]
            out["ok"] = True
            out["chart_kind"] = "bar"
            out["title"] = "装修类型与均价"
            out["records"] = gg.to_dict(orient="records")
            out["x"] = "decoration"
            out["y"] = "unit_price"

        elif t == AnalysisTaskType.floor_band_price:
            if not _need_cols(df, ["floor_band", "unit_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 floor_band+unit_price"
                return out
            sub = df[["floor_band", "unit_price"]].copy()
            sub["unit_price"] = _num(sub["unit_price"])
            sub = sub.dropna()
            gg = sub.groupby(sub["floor_band"].astype(str), dropna=False)["unit_price"].mean().reset_index()
            gg.columns = ["floor_band", "unit_price"]
            out["ok"] = True
            out["chart_kind"] = "bar"
            out["title"] = "楼层档与均价"
            out["records"] = gg.to_dict(orient="records")
            out["x"] = "floor_band"
            out["y"] = "unit_price"

        elif t == AnalysisTaskType.building_age_price_trend:
            if not _need_cols(df, ["build_year", "unit_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 build_year+unit_price"
                return out
            sub = df[["build_year", "unit_price"]].copy()
            sub["build_year"] = _num(sub["build_year"])
            sub["unit_price"] = _num(sub["unit_price"])
            sub = sub.dropna()
            sub["decade"] = (sub["build_year"] // 10 * 10).astype(int)
            gg = sub.groupby("decade")["unit_price"].mean().reset_index()
            out["ok"] = True
            out["chart_kind"] = "bar"
            out["title"] = "建成年代（十年）与均价"
            out["records"] = gg.rename(columns={"decade": "decade_start"}).to_dict(orient="records")
            out["x"] = "decade_start"
            out["y"] = "unit_price"

        elif t == AnalysisTaskType.community_followers_rank:
            if not _need_cols(df, ["community", "followers"]) or len(df) < _MIN_N:
                out["reason"] = "需要 community+followers"
                return out
            sub = df[["community", "followers"]].copy()
            sub["followers"] = _num(sub["followers"])
            sub = sub.dropna()
            gg = sub.groupby(sub["community"].astype(str), dropna=False)["followers"].sum().reset_index()
            gg = gg.sort_values("followers", ascending=False).head(15)
            out["ok"] = True
            out["chart_kind"] = "bar"
            out["title"] = "小区关注热度 Top15"
            out["records"] = gg.to_dict(orient="records")
            out["x"] = "community"
            out["y"] = "followers"

        elif t == AnalysisTaskType.total_area_scatter:
            if not _need_cols(df, ["area_m2", "total_price"]) or len(df) < _MIN_N:
                out["reason"] = "需要 area_m2+total_price"
                return out
            sub = df[["area_m2", "total_price"]].copy()
            sub["area_m2"] = _num(sub["area_m2"])
            sub["total_price"] = _num(sub["total_price"])
            sub = sub.dropna()
            out["ok"] = True
            out["chart_kind"] = "scatter"
            out["title"] = "建筑面积 vs 总价（万元）"
            out["records"] = sub.to_dict(orient="records")
            out["x"] = "area_m2"
            out["y"] = "total_price"

        elif t == AnalysisTaskType.price_outlier_flag:
            if "unit_price" not in df.columns or len(df) < _MIN_N:
                out["reason"] = "需要 unit_price"
                return out
            s = _num(df["unit_price"]).dropna()
            if len(s) < _MIN_N:
                out["reason"] = "单价有效样本不足"
                return out
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df["unit_price"].pipe(_num) < low) | (df["unit_price"].pipe(_num) > high)
            cnt = int(mask.sum())
            out["ok"] = True
            out["chart_kind"] = "text"
            out["title"] = "单价异常房源（IQR）"
            out["summary"] = f"有效样本 {len(s)}，识别疑似异常挂牌约 {cnt} 条（仅供参考）。"

        elif t == AnalysisTaskType.unit_price_histogram:
            if "unit_price" not in df.columns or len(df) < _MIN_N:
                out["reason"] = "需要 unit_price"
                return out
            s = _num(df["unit_price"]).dropna()
            if len(s) < _MIN_N:
                return out
            arr = s.astype(float).to_numpy()
            nb = min(40, max(10, len(arr) // 5))
            counts, edges = np.histogram(arr, bins=nb)
            mids = (edges[:-1] + edges[1:]) / 2.0
            out["ok"] = True
            out["chart_kind"] = "histogram"
            out["title"] = "单价分布"
            out["records"] = [{"bin": float(m), "count": int(c)} for m, c in zip(mids, counts)]

    except Exception as e:
        out["error"] = str(e)

    return out


def fallback_plan(df: pd.DataFrame) -> list[PlannedTask]:
    tasks: list[PlannedTask] = []
    if _need_cols(df, ["district", "unit_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.district_price_rank))
    if _need_cols(df, ["layout", "unit_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.layout_price_box))
    if _need_cols(df, ["area_m2", "unit_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.area_band_price))
    if "area_m2" in df.columns:
        tasks.append(PlannedTask(type=AnalysisTaskType.area_band_share_pie))
    if _need_cols(df, ["decoration", "unit_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.decoration_price_compare))
    if _need_cols(df, ["floor_band", "unit_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.floor_band_price))
    if _need_cols(df, ["build_year", "unit_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.building_age_price_trend))
    if _need_cols(df, ["community", "followers"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.community_followers_rank))
    if _need_cols(df, ["area_m2", "total_price"]):
        tasks.append(PlannedTask(type=AnalysisTaskType.total_area_scatter))
    if "unit_price" in df.columns:
        tasks.append(PlannedTask(type=AnalysisTaskType.unit_price_histogram))
        tasks.append(PlannedTask(type=AnalysisTaskType.price_outlier_flag))
    return tasks[:8]


_PLAN_PROMPT = """你是二手房数据分析负责人。根据「数据画像」与「列名」，选出最值得做的分析任务（最多 8 个）。
只输出合法 JSON，不要 markdown，格式：
{"tasks":[{"type":"任务类型枚举"}, ...]}

可选 type（字符串必须完全一致）：
district_price_rank, layout_price_box, area_band_price, area_band_share_pie, decoration_price_compare,
floor_band_price, building_age_price_trend, community_followers_rank,
total_area_scatter, price_outlier_flag, unit_price_histogram

规则：
- 缺少必填列时不要选该任务（例如 decoration_price_compare 需要 decoration 与 unit_price）。
- 样本行数过少时少选复杂任务。
- 优先：区域均价（district_price_rank，对应横向柱状图更易读）、户型与单价、面积段均价与面积段占比饼图。
- 若有 decoration/floor_band/followers 则各选一项。
"""


def plan_analysis_with_llm(df: pd.DataFrame, settings: Settings) -> tuple[list[PlannedTask], str]:
    """返回 (tasks, raw_json_or_error_note)。"""
    from langchain_openai import ChatOpenAI

    if not settings.dashscope_api_key:
        tasks = fallback_plan(df)
        return tasks, ""

    plan_model = settings.houseinsight_plan_model or settings.houseinsight_llm_model
    profile = build_dataset_profile(df, sample_per_col=2)
    payload = {
        "profile": profile,
        "columns": list(df.columns),
        "row_count": len(df),
    }
    llm = ChatOpenAI(
        model=plan_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=0.1,
    )
    msg = _PLAN_PROMPT + "\n数据:\n" + json.dumps(payload, ensure_ascii=False)
    resp = llm.invoke([HumanMessage(content=msg)])
    text = getattr(resp, "content", str(resp)).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
        parsed = AnalysisPlanResponse.model_validate(data)
        tasks = parsed.tasks[:8]
        if not tasks:
            fb = fallback_plan(df)
            return fb, f"plan_empty_fallback:{text[:500]}"
        return tasks, text
    except Exception as e:
        return fallback_plan(df), f"plan_parse_fallback:{e}"


def run_planned_analysis(df: pd.DataFrame, tasks: list[PlannedTask]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for i, task in enumerate(tasks):
        key = f"{task.type.value}_{i}"
        results[key] = execute_task(df, task)
    narrative_parts: list[str] = []
    for key, r in results.items():
        if not r.get("ok"):
            continue
        if r.get("chart_kind") == "text":
            narrative_parts.append("- " + (r.get("summary") or r.get("title") or key))
        elif r.get("title"):
            narrative_parts.append(f"- {r['title']}（图表 {key}）。")
    summary_md = "\n".join(narrative_parts) if narrative_parts else "（当前数据下可展示的分析项较少，请检查列是否齐全。）"
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "task_results": results,
        "analysis_summary_markdown": summary_md,
    }
