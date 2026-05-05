"""L3：对 description_raw 抽样用 LLM 推断弱特征（地铁/学区相关度），失败则原样返回。"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from server.core.config import Settings


def enrich_description_columns(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if df.empty or not settings.dashscope_api_key or not settings.houseinsight_description_enrich:
        return df
    if "description_raw" not in df.columns:
        return df
    out = df.copy()
    for c in ("description_hint_subway", "description_hint_school"):
        if c not in out.columns:
            out[c] = pd.Series(np.nan, index=out.index, dtype="float64")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    n = min(settings.houseinsight_description_sample_n, len(out))
    if n < 5:
        return out
    rng = np.random.default_rng(42)
    positions = rng.choice(len(out), size=n, replace=False).tolist()
    rows: list[dict[str, Any]] = []
    for pos in positions:
        raw = out.iloc[int(pos)].get("description_raw")
        snip = (str(raw) if raw is not None and not (isinstance(raw, float) and np.isnan(raw)) else "")[:400]
        rows.append({"pos": int(pos), "snippet": snip})
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.houseinsight_llm_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=0.0,
    )
    prompt = (
        "你是文本弱标注器。下面是二手房「描述」片段（带行号 pos）。"
        "请仅根据文本是否**显式或较强暗示**提及地铁/轨道交通、学校/学区/教育配套，给出 0～1 的连续分数（无证据则为 0）。\n"
        "只输出合法 JSON，不要 markdown：\n"
        '{"scores":[{"pos":整型,"subway":0到1,"school":0到1},...]}\n'
        "scores 条数与输入一致，pos 必须与输入相同。\n输入：\n"
        + json.dumps(rows, ensure_ascii=False)
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = getattr(resp, "content", str(resp)).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
        arr = data.get("scores")
        if not isinstance(arr, list):
            return out
        for item in arr:
            if not isinstance(item, dict):
                continue
            pos = item.get("pos")
            if pos is None:
                continue
            p = int(pos)
            if p < 0 or p >= len(out):
                continue
            try:
                sw = float(item.get("subway", 0) or 0)
                sc = float(item.get("school", 0) or 0)
            except (TypeError, ValueError):
                continue
            sw = max(0.0, min(1.0, sw))
            sc = max(0.0, min(1.0, sc))
            out.iat[p, out.columns.get_loc("description_hint_subway")] = sw
            out.iat[p, out.columns.get_loc("description_hint_school")] = sc
    except Exception:
        return df
    return out
