from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"
_FONT = dict(family="system-ui, 'Segoe UI', Roboto, 'PingFang SC', sans-serif", size=12)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = str(x)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _style_horizontal_bar(fig: go.Figure, categories_ordered: list[str]) -> None:
    """横向条形图：避免 Plotly 自动抽稀 Y 轴类目标签；按行数加高、加宽左侧给区名。"""
    cats = _dedupe_preserve_order(categories_ordered)
    n = len(cats)
    max_lab = max((len(c) for c in cats), default=4)
    left = int(max(112, min(340, 8 * max_lab + 40)))
    height = int(max(420, min(1800, 24 * n + 140)))
    tick_px = max(9, min(12, 220 // max(n, 1)))

    fig.update_layout(
        template=_TEMPLATE,
        height=height,
        margin=dict(l=left, r=52, t=56, b=72),
        font=_FONT,
        title_font=dict(size=15),
        hoverlabel=dict(font_size=12, font_family=_FONT["family"]),
        bargap=0.28,
    )
    # 分类轴在 y：显式列出每个刻度，防止「隔一行才显示」
    fig.update_yaxes(
        automargin=True,
        type="category",
        categoryorder="array",
        categoryarray=cats,
        tickmode="array",
        tickvals=cats,
        ticktext=cats,
        tickfont=dict(size=tick_px),
        title_standoff=10,
    )
    fig.update_xaxes(separatethousands=True, tickformat=",.0f")


def _style_vertical_bar(fig: go.Figure, *, n_categories: int) -> None:
    fig.update_layout(
        template=_TEMPLATE,
        height=max(400, min(900, 28 * n_categories + 160)),
        margin=dict(l=72, r=48, t=56, b=max(88, min(200, 10 * n_categories))),
        font=_FONT,
        title_font=dict(size=15),
        bargap=0.22,
    )
    if n_categories > 6:
        fig.update_xaxes(tickangle=-35, automargin=True)


def _style_scatter(fig: go.Figure) -> None:
    fig.update_layout(
        template=_TEMPLATE,
        height=460,
        margin=dict(l=72, r=48, t=56, b=72),
        font=_FONT,
        hovermode="closest",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.65, line=dict(width=0)))
    fig.update_xaxes(separatethousands=True, tickformat=",.0f")
    fig.update_yaxes(separatethousands=True, tickformat=",.0f")


def _style_histogram(fig: go.Figure) -> None:
    fig.update_layout(
        template=_TEMPLATE,
        height=400,
        margin=dict(l=72, r=48, t=56, b=72),
        font=_FONT,
        bargap=0.12,
    )
    fig.update_xaxes(separatethousands=True, tickformat=",.0f")


def _style_pie(fig: go.Figure) -> None:
    fig.update_layout(
        template=_TEMPLATE,
        height=440,
        margin=dict(l=48, r=48, t=56, b=48),
        font=_FONT,
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02, font=dict(size=11)),
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", insidetextorientation="radial")


def _style_box(fig: go.Figure, n_x: int) -> None:
    fig.update_layout(
        template=_TEMPLATE,
        height=max(420, min(820, 32 * n_x + 200)),
        margin=dict(l=72, r=48, t=56, b=max(88, min(220, 8 * n_x))),
        font=_FONT,
        showlegend=False,
    )
    if n_x > 5:
        fig.update_xaxes(tickangle=-30, automargin=True)
    fig.update_yaxes(separatethousands=True, tickformat=",.0f")


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
        fig = px.bar(sub, x=y, y=x, orientation="h", title=title, color_discrete_sequence=["#3B82F6"])
        fig.update_layout(xaxis_title=x_title or y, yaxis_title=y_title or x)
        if x in sub.columns:
            _style_horizontal_bar(fig, sub[x].astype(str).tolist())
        else:
            fig.update_layout(template=_TEMPLATE, height=480, margin=dict(l=100, r=48, t=56, b=72), font=_FONT)
    else:
        fig = px.bar(sub, x=x, y=y, title=title, color_discrete_sequence=["#6366F1"])
        fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
        _style_vertical_bar(fig, n_categories=len(sub))
        fig.update_yaxes(separatethousands=True, tickformat=",.0f")
        if len(sub) > 6:
            fig.update_xaxes(tickangle=-30, automargin=True)

    if not horizontal:
        fig.update_xaxes(separatethousands=True, tickformat=",.0f", automargin=True)

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
                    fig = px.pie(
                        sub,
                        names=nc,
                        values=vc,
                        title=title,
                        hole=0.32,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    _style_pie(fig)
                    figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                    first_js = False
            elif kind == "box" and spec.get("records"):
                sub = pd.DataFrame(spec["records"])
                if "category" in sub.columns and "unit_price" in sub.columns:
                    fig = px.box(
                        sub,
                        x="category",
                        y="unit_price",
                        title=title,
                        color="category",
                        color_discrete_sequence=px.colors.qualitative.Pastel1,
                    )
                    _style_box(fig, n_x=sub["category"].nunique())
                    figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                    first_js = False
            elif kind == "scatter" and spec.get("records") and spec.get("x") and spec.get("y"):
                sub = pd.DataFrame(spec["records"])
                fig = px.scatter(
                    sub,
                    x=spec["x"],
                    y=spec["y"],
                    title=title,
                    color_discrete_sequence=["#0EA5E9"],
                )
                _style_scatter(fig)
                figures[name] = fig.to_html(full_html=False, include_plotlyjs=first_js)
                first_js = False
            elif kind == "histogram" and spec.get("records"):
                sub = pd.DataFrame(spec["records"])
                if {"bin", "count"}.issubset(sub.columns):
                    fig = px.bar(
                        sub,
                        x="bin",
                        y="count",
                        title=title,
                        color_discrete_sequence=["#8B5CF6"],
                    )
                    fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="套数")
                    _style_histogram(fig)
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


def _district_horizontal_figure(agg: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        agg,
        x="unit_price",
        y="district",
        orientation="h",
        title="各城区二手房均价排行（元/㎡）",
        color_discrete_sequence=["#2563EB"],
    )
    fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="城区")
    _style_horizontal_bar(fig, agg["district"].astype(str).tolist())
    return fig


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
        fig = _district_horizontal_figure(agg)
        figures["district_avg_unit_price"] = fig.to_html(full_html=False, include_plotlyjs=js)
        js = False
    if "unit_price" in df.columns:
        s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
        if len(s) >= 5:
            fig = px.histogram(s, nbins=40, title="单价分布（元/㎡）", color_discrete_sequence=["#7C3AED"])
            fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="套数")
            _style_histogram(fig)
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
        fig = _district_horizontal_figure(agg)
        figures["district_avg_unit_price"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    if "unit_price" in df.columns:
        s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
        if len(s) >= 5:
            fig = px.histogram(s, nbins=40, title="单价分布（元/㎡）", color_discrete_sequence=["#7C3AED"])
            fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="套数")
            _style_histogram(fig)
            figures["unit_price_hist"] = fig.to_html(full_html=False, include_plotlyjs=False)

    if {"area_m2", "unit_price"}.issubset(df.columns):
        sub = df[["area_m2", "unit_price"]].copy()
        sub["area_m2"] = pd.to_numeric(sub["area_m2"], errors="coerce")
        sub["unit_price"] = pd.to_numeric(sub["unit_price"], errors="coerce")
        sub = sub.dropna()
        if len(sub) >= 5:
            fig = px.scatter(
                sub,
                x="area_m2",
                y="unit_price",
                title="建筑面积 vs 单价",
                color_discrete_sequence=["#0EA5E9"],
            )
            fig.update_layout(xaxis_title="建筑面积（㎡）", yaxis_title="单价（元/㎡）")
            _style_scatter(fig)
            figures["area_vs_unit_price"] = fig.to_html(full_html=False, include_plotlyjs=False)

    if "layout" in df.columns:
        vc = df["layout"].fillna("未知").astype(str).value_counts().head(15).reset_index()
        vc.columns = ["layout", "count"]
        fig = px.bar(vc, x="layout", y="count", title="户型挂牌量 Top", color_discrete_sequence=["#6366F1"])
        _style_vertical_bar(fig, n_categories=len(vc))
        fig.update_layout(xaxis_title="户型", yaxis_title="套数")
        fig.update_yaxes(separatethousands=True, tickformat=",.0f")
        if len(vc) > 6:
            fig.update_xaxes(tickangle=-30, automargin=True)
        figures["layout_counts"] = fig.to_html(full_html=False, include_plotlyjs=False)

    return figures


def empty_message_figure(message: str, include_plotlyjs: str | bool = "cdn") -> str:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=_TEMPLATE, font=_FONT, height=320, margin=dict(t=48, b=48))
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
