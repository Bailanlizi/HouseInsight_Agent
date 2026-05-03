from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _bar_from_records(
    records: list[dict[str, Any]],
    x: str,
    y: str,
    title: str,
    include_plotlyjs: str | bool,
    *,
    horizontal: bool = False,
    x_title: str | None = None,
    y_title: str | None = None,
) -> str:
    sub = pd.DataFrame(records)
    if y in sub.columns:
        sub[y] = pd.to_numeric(sub[y], errors="coerce")
        sub = sub.dropna(subset=[y])
    if horizontal:
        fig = px.bar(sub, x=y, y=x, orientation="h", title=title)
        fig.update_layout(xaxis_title=x_title or y, yaxis_title=y_title or x)
    else:
        fig = px.bar(sub, x=x, y=y, title=title)
        fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
    fig.update_layout(template="plotly_white", margin=dict(l=80, r=40, t=50, b=80))
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def figures_from_task_results(df: pd.DataFrame, task_results: dict[str, Any]) -> dict[str, str]:
    """按分析任务结果动态生成图表；首个含 CDN plotly.js。"""
    figures: dict[str, str] = {}
    first_js: str | bool = "cdn"

    for name, spec in task_results.items():
        if not isinstance(spec, dict) or not spec.get("ok"):
            continue
        kind = spec.get("chart_kind")
        title = str(spec.get("title") or name)

        try:
            if kind == "bar" and spec.get("records") and spec.get("x") and spec.get("y"):
                hz = bool(spec.get("bar_horizontal"))
                html = _bar_from_records(
                    spec["records"],
                    spec["x"],
                    spec["y"],
                    title,
                    first_js,
                    horizontal=hz,
                    x_title="单价（元/㎡）" if hz else None,
                    y_title="城区" if hz and spec.get("x") == "district" else None,
                )
                figures[name] = html
                first_js = False
            elif kind == "pie" and spec.get("records"):
                sub = pd.DataFrame(spec["records"])
                nc = spec.get("pie_names") or "band"
                vc = spec.get("pie_values") or "count"
                if nc in sub.columns and vc in sub.columns:
                    fig = px.pie(sub, names=nc, values=vc, title=title, hole=0.28)
                    fig.update_layout(template="plotly_white", margin=dict(t=50, b=40))
                    figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                    first_js = False
            elif kind == "box" and spec.get("records"):
                sub = pd.DataFrame(spec["records"])
                if "category" in sub.columns and "unit_price" in sub.columns:
                    fig = px.box(sub, x="category", y="unit_price", title=title)
                    figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                    first_js = False
            elif kind == "scatter" and spec.get("records") and spec.get("x") and spec.get("y"):
                sub = pd.DataFrame(spec["records"])
                fig = px.scatter(sub, x=spec["x"], y=spec["y"], title=title)
                figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                first_js = False
            elif kind == "histogram" and spec.get("records"):
                sub = pd.DataFrame(spec["records"])
                if {"bin", "count"}.issubset(sub.columns):
                    fig = px.bar(sub, x="bin", y="count", title=title)
                    fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="套数")
                    figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                    first_js = False
            elif kind == "text":
                msg = str(spec.get("summary") or spec.get("title") or "")
                figures[name] = empty_message_figure(msg or title, include_plotlyjs=first_js)
                first_js = False
        except Exception:
            continue

    if not figures and not df.empty:
        return _figures_legacy_fallback(df, cdn=True)
    return figures


def _figures_legacy_fallback(df: pd.DataFrame, *, cdn: bool) -> dict[str, str]:
    """无任务结果时的最小兜底图表。"""
    figures: dict[str, str] = {}
    js: str | bool = "cdn" if cdn else False
    if "district" in df.columns and "unit_price" in df.columns:
        t = df[["district", "unit_price"]].copy()
        t["unit_price"] = pd.to_numeric(t["unit_price"], errors="coerce")
        agg = t.groupby("district", dropna=False)["unit_price"].mean().reset_index().sort_values(
            "unit_price", ascending=False
        )
        agg = agg.dropna(subset=["unit_price"])
        fig = px.bar(agg, x="unit_price", y="district", orientation="h", title="各城区均价（元/㎡）")
        fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="城区", template="plotly_white")
        figures["district_avg_unit_price"] = fig.to_html(full_html=False, include_plotlyjs=js)
        js = False
    if "unit_price" in df.columns:
        s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
        if len(s) >= 5:
            fig = px.histogram(s, nbins=40, title="单价分布（元/㎡）")
            fig.update_layout(xaxis_title="单价", yaxis_title="套数")
            figures["unit_price_hist"] = fig.to_html(full_html=False, include_plotlyjs=js)
            js = False
    return figures


def figures_from_analysis(df: pd.DataFrame, analysis: dict[str, Any]) -> dict[str, str]:
    """兼容：若 analysis 含 task_results 则按任务出图，否则沿用固定图表集。"""
    if df.empty:
        return {}

    tr = analysis.get("task_results")
    if isinstance(tr, dict) and tr:
        return figures_from_task_results(df, tr)

    figures: dict[str, str] = {}

    if "district" in df.columns and "unit_price" in df.columns:
        t = df[["district", "unit_price"]].copy()
        t["unit_price"] = pd.to_numeric(t["unit_price"], errors="coerce")
        agg = t.groupby("district", dropna=False)["unit_price"].mean().reset_index().sort_values(
            "unit_price", ascending=False
        )
        agg = agg.dropna(subset=["unit_price"])
        fig = px.bar(agg, x="unit_price", y="district", orientation="h", title="各城区均价（元/㎡）")
        fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="城区", template="plotly_white")
        figures["district_avg_unit_price"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    if "unit_price" in df.columns:
        s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
        if len(s) >= 5:
            fig = px.histogram(s, nbins=40, title="单价分布（元/㎡）")
            fig.update_layout(xaxis_title="单价", yaxis_title="套数")
            figures["unit_price_hist"] = fig.to_html(full_html=False, include_plotlyjs=False)

    if {"area_m2", "unit_price"}.issubset(df.columns):
        sub = df[["area_m2", "unit_price"]].copy()
        sub["area_m2"] = pd.to_numeric(sub["area_m2"], errors="coerce")
        sub["unit_price"] = pd.to_numeric(sub["unit_price"], errors="coerce")
        sub = sub.dropna()
        if len(sub) >= 5:
            fig = px.scatter(sub, x="area_m2", y="unit_price", title="建筑面积 vs 单价")
            fig.update_layout(xaxis_title="建筑面积㎡", yaxis_title="单价元/㎡")
            figures["area_vs_unit_price"] = fig.to_html(full_html=False, include_plotlyjs=False)

    if "layout" in df.columns:
        vc = df["layout"].fillna("未知").astype(str).value_counts().head(15).reset_index()
        vc.columns = ["layout", "count"]
        fig = px.bar(vc, x="layout", y="count", title="户型挂牌量 Top")
        figures["layout_counts"] = fig.to_html(full_html=False, include_plotlyjs=False)

    return figures


def empty_message_figure(message: str, include_plotlyjs: str | bool = "cdn") -> str:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
