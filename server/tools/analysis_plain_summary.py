"""基于 analysis / task_results 生成 300～500 字中文纯文本分析总结（供首屏展示）。"""

from __future__ import annotations

from typing import Any

import pandas as pd

_MIN = 300
_MAX = 500


def _clip(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    chunk = s[: max_len + 1]
    cut = chunk.rfind("。")
    if cut >= max_len // 2:
        return chunk[: cut + 1]
    return chunk[:max_len].rstrip() + "…"


def _pad(s: str, min_len: int) -> str:
    tail = (
        "后续可结合户型、面积段、装修与楼层档对单价做交叉下钻，并结合城区与小区维度观察供应结构，以支撑定价与去化判断。"
    )
    out = s.strip()
    while len(out) < min_len:
        out = out + " " + tail
        if len(out) > _MAX + 200:
            break
    return out


def build_analysis_plain_summary(
    df: pd.DataFrame,
    analysis: dict[str, Any],
    task_results: dict[str, Any] | None,
) -> str:
    n = int(len(df))
    cols = list(df.columns)
    col_preview = "、".join(cols[:10]) + (" 等" if len(cols) > 10 else "")

    sec1 = f"【数据概览】当前样本共 {n} 条挂牌记录，结构化字段主要包括：{col_preview}。"

    sec2 = "【价格分析】"
    uq = analysis.get("unit_price_quantiles") if isinstance(analysis, dict) else None
    if isinstance(uq, dict) and uq:
        sec2 += (
            f"单价（元/㎡）大致分布为：最低约 {uq.get('min', 0):.0f}，"
            f"中位数约 {uq.get('p50', 0):.0f}，"
            f"上四分位约 {uq.get('p75', 0):.0f}，"
            f"最高约 {uq.get('max', 0):.0f}。"
            f"整体离散程度可结合 p25～p75 区间理解市场主流价位带。"
        )
    elif "unit_price" in df.columns:
        s = pd.to_numeric(df["unit_price"], errors="coerce").dropna()
        if len(s) >= 5:
            sec2 += f"单价样本量 {len(s)}，均值约 {float(s.mean()):.0f} 元/㎡，中位数约 {float(s.median()):.0f} 元/㎡。"
        else:
            sec2 += "单价字段有效样本较少，价格结论仅供参考。"
    else:
        sec2 += "缺少单价或分位数信息，未做细粒度价格分布解读。"

    sec3 = "【供应分析】"
    ds = analysis.get("district_summary") if isinstance(analysis, dict) else None
    if isinstance(ds, list) and ds:
        ranked = sorted(
            (x for x in ds if isinstance(x, dict)),
            key=lambda x: float(x.get("listings") or 0),
            reverse=True,
        )[:5]
        bits = [
            f"{x.get('district', '?')}约 {int(float(x.get('listings') or 0))} 套"
            + (
                f"（均价约 {float(x.get('avg_unit_price') or 0):.0f} 元/㎡）"
                if x.get("avg_unit_price") not in (None, "", "nan")
                else ""
            )
            for x in ranked
        ]
        sec3 += "城区挂牌量前列包括：" + "；".join(bits) + "。"
    else:
        sec3 += "缺少按城区聚合的摘要，供应结构以全表列分布为准。"

    buckets = analysis.get("area_buckets") if isinstance(analysis, dict) else None
    if isinstance(buckets, dict) and buckets:
        top_b = sorted(buckets.items(), key=lambda kv: int(kv[1]), reverse=True)[:3]
        sec3 += "面积段套数占比靠前：" + "，".join(f"{k}约{v}套" for k, v in top_b) + "。"

    sec4 = "【关键发现】"
    if isinstance(task_results, dict) and task_results:
        ok_n = sum(1 for v in task_results.values() if isinstance(v, dict) and v.get("ok"))
        tot = len(task_results)
        sec4 += f"规划分析任务共 {tot} 项，其中 {ok_n} 项成功产出图表或统计。"
        fails = [
            (k, (v or {}).get("reason", ""))
            for k, v in task_results.items()
            if isinstance(v, dict) and not v.get("ok")
        ][:4]
        if fails:
            sec4 += "未完成任务示例：" + "；".join(f"{k}（{r}）" for k, r in fails if r) + "。"
        if ok_n < tot:
            sec4 += "若多项任务失败，通常与清洗后仍缺少 layout、area_m2、decoration、followers 等维度有关，可检查原始表复合列是否已拆分。"
    else:
        sec4 += "当前未附带任务级结果对象，分析以聚合指标与兜底图表为主。"

    text = "\n".join([sec1, sec2, sec3, sec4])
    text = _pad(text, _MIN)
    text = _clip(text, _MAX)
    return text
