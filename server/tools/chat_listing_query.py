"""对话中受控查询清洗表：结构化意图 + 相关度打分排序，仅暴露白名单列。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

# 允许返回给模型与用户的列（存在则输出）
_LISTING_DISPLAY_COLS: tuple[str, ...] = (
    "district",
    "community",
    "layout",
    "layout_normalized",
    "area_m2",
    "total_price",
    "unit_price",
    "floor",
    "floor_band",
    "orientation",
    "decoration",
    "build_year",
    "building_type",
    "listing_title",
    "location_raw",
    "house_info_raw",
    "follow_info_raw",
    "listing_id",
    "tag_near_subway",
    "tag_near_bus",
    "tag_has_balcony",
    "tag_lighting",
    "tag_subway_station_hint",
    "tag_elevator",
)

_GEO_TEXT_COLUMNS: tuple[str, ...] = (
    "district",
    "location_raw",
    "listing_title",
    "community",
)

# 用于软匹配的正文拼接（覆盖 district 未写但描述里出现的场景）
_SOFT_TEXT_COLUMNS: tuple[str, ...] = (
    "description_raw",
    "house_info_raw",
    "location_raw",
    "listing_title",
    "follow_info_raw",
    "community",
)

# 打分权重（相对尺度，可日后调参）
_W_GEO = 120.0
_W_COMMUNITY = 95.0
_W_LAYOUT = 75.0
_W_PRICE_IN_RANGE = 55.0
_W_PRICE_NEAR_RANGE = 22.0
_W_SUBWAY_STATION = 100.0
_W_SUBWAY_TAG = 55.0
_W_SUBWAY_TEXT = 50.0
_W_SUBWAY_DIM_CAP = 85.0
_W_BALCONY_TAG = 45.0
_W_BALCONY_TEXT = 40.0
_W_BALCONY_CAP = 70.0
_W_LIGHT_TAG = 40.0
_W_LIGHT_TEXT = 35.0
_W_LIGHT_CAP = 65.0
_W_YEAR_OK = 45.0
_W_YEAR_PARTIAL = 18.0
# 噪声/异常：强罚分（与正常维度同量纲，保证排后）
_W_NOISE_PARKING = 600.0
_W_NOISE_ABSURD_LAYOUT = 550.0
# 在已匹配 layout_contains 时，对「正常成套住宅」额外套利
_W_RESIDENTIAL_FAMILY = 110.0

# 从候选中硬剔除（若剔空则回退为仅罚分）
_NOISE_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "车位",
    "停车位",
    "车库出售",
    "产权车位",
    "人防车位",
)
# 异常户型：解析到的「室」数上限（住宅极少超过）
_ABSURD_MIN_ROOMS = 10


def _max_room_count(text: str) -> int:
    """从户型串中取最大的「n室」里的 n。"""
    if not text or not isinstance(text, str):
        return 0
    found = re.findall(r"(\d+)\s*室", text)
    if not found:
        return 0
    return max(int(x) for x in found)


def _noise_parking_mask(df: pd.DataFrame) -> pd.Series:
    """疑似车位/非成套住宅出让。"""
    blob = _soft_text_blob(df)
    m = pd.Series(False, index=df.index)
    for sub in _NOISE_EXCLUDE_SUBSTRINGS:
        m = m | blob.str.contains(sub, case=False, na=False, regex=False)
    return m


def _absurd_layout_mask(df: pd.DataFrame) -> pd.Series:
    """户型字段出现过大「室」数（如 20室9厅），多为噪声或异常录入。"""
    parts: list[pd.Series] = []
    for col in ("layout_normalized", "layout", "house_info_raw"):
        if col in df.columns:
            n = df[col].astype(str).map(_max_room_count)
            parts.append(n)
    if not parts:
        return pd.Series(False, index=df.index)
    mx = parts[0]
    for p in parts[1:]:
        mx = pd.concat([mx, p], axis=1).max(axis=1)
    return mx >= _ABSURD_MIN_ROOMS


def _normal_family_layout_mask(df: pd.DataFrame) -> pd.Series:
    """典型商品住宅：layout_normalized/main 串里室数为 2～6。"""
    def _ok_one(s: str) -> bool:
        n = _max_room_count(s)
        return 2 <= n <= 6

    m = pd.Series(False, index=df.index)
    for col in ("layout_normalized", "layout"):
        if col in df.columns:
            m = m | df[col].astype(str).map(_ok_one)
    return m


def _expand_district_geo_keywords(raw: str) -> tuple[str, ...]:
    """
    用户说的城区子串 + 常见别名，用于在 district/位置/标题/小区 多列 OR 匹配。
    缓解「高新」写在 location、district 标成武侯/天府等」导致的 0 行。
    """
    s = raw.strip()
    if not s:
        return ()
    variants: list[str] = [s]
    if "高新" in s or s in ("高新区", "成都高新区"):
        variants.extend(["高新", "高新区", "成都高新", "高新城南", "新川科技园", "新川"])
    if "天府" in s:
        variants.extend(["天府新区", "天府", "天新"])
    if "温江" in s:
        variants.extend(["温江", "温江区"])
    if "锦江" in s:
        variants.extend(["锦江", "锦江区"])
    if "武侯" in s:
        variants.extend(["武侯", "武侯区"])
    if "青羊" in s:
        variants.extend(["青羊", "青羊区"])
    if "金牛" in s:
        variants.extend(["金牛", "金牛区"])
    if "成华" in s:
        variants.extend(["成华", "成华区"])
    if "郫" in s or "郫县" in s:
        variants.extend(["郫都", "郫都区", "郫县"])
    if "双流" in s:
        variants.extend(["双流", "双流区"])
    if "龙泉" in s:
        variants.extend(["龙泉驿", "龙泉驿区", "龙泉"])
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def _geo_region_mask(df: pd.DataFrame, keywords: tuple[str, ...]) -> pd.Series:
    """任一关键词在任一地理相关文本列中出现即为 True。"""
    cols = [c for c in _GEO_TEXT_COLUMNS if c in df.columns]
    if not cols or not keywords:
        return pd.Series(True, index=df.index)
    m = pd.Series(False, index=df.index)
    for kw in keywords:
        for c in cols:
            m = m | df[c].astype(str).str.contains(kw, case=False, na=False, regex=False)
    return m


def _soft_text_blob(df: pd.DataFrame) -> pd.Series:
    """拼接若干正文列，供地铁/阳台等关键词软命中。"""
    cols = [c for c in _SOFT_TEXT_COLUMNS if c in df.columns]
    if not cols:
        return pd.Series("", index=df.index, dtype=object)
    acc = df[cols[0]].astype(str).fillna("")
    for c in cols[1:]:
        acc = acc + " " + df[c].astype(str).fillna("")
    return acc


def _cap_dim(base: pd.Series, extra: pd.Series, cap: float) -> pd.Series:
    """同一维度内 tag 与正文加分叠加但不超过 cap。"""
    s = base.astype(float) + extra.astype(float)
    return s.clip(upper=cap)


class ListingSearchIntent(BaseModel):
    """由 LLM 从用户话术中抽取；仅当 needs_row_samples 为 True 时执行查表。"""

    needs_row_samples: bool = Field(
        default=False,
        description="用户是否在要房源列表、推荐几套、筛选某区/小区/价位等（需要行级数据时为 true）",
    )
    district_contains: str | None = Field(
        default=None,
        description="城区或板块子串；后端按相关度打分，并在 district/位置/标题/正文等多处软匹配",
    )
    community_contains: str | None = Field(default=None, description="小区名包含的子串")
    layout_contains: str | None = Field(
        default=None,
        description="户型包含子串（三室、3室、四室等）；多人居住需求须映射到此字段以便住宅加权",
    )
    min_unit_price: float | None = Field(
        default=None,
        description="最低单价 元/㎡；仅当用户给出明确数字时填写",
    )
    max_unit_price: float | None = Field(
        default=None,
        description="最高单价 元/㎡；仅当用户给出明确数字时填写，勿用 p25 等统计值代替",
    )
    min_total_price_wan: float | None = Field(
        default=None,
        description="最低总价 万元；仅当用户给出明确数字时填写",
    )
    max_total_price_wan: float | None = Field(
        default=None,
        description="最高总价 万元；仅当用户给出明确数字时填写",
    )
    near_subway: bool | None = Field(
        default=None,
        description="用户重视近地铁/地铁旁时填 true；后端对 tag 与正文「地铁/号线」等联合加分，非唯一硬条件",
    )
    has_balcony: bool | None = Field(
        default=None,
        description="用户明确阳台/露台需求时填 true；标签与正文「阳台」等联合加分",
    )
    lighting_preferred: bool | None = Field(
        default=None,
        description="用户强调采光/通透/全明时填 true；标签与正文关键词联合加分",
    )
    subway_station_contains: str | None = Field(
        default=None,
        description="地铁站名子串；命中提示列或正文则大幅加分",
    )
    min_build_year: int | None = Field(
        default=None,
        description="最低建成年份；命中则加分，略低于阈值给弱分",
    )
    max_rows: int = Field(default=15, ge=1, le=40, description="最多返回条数 1-40")


def _intent_relevance_scores(df: pd.DataFrame, intent: ListingSearchIntent) -> pd.Series:
    """对每一行计算与查询意图的相关度（越高越优先返回）。"""
    score = pd.Series(0.0, index=df.index, dtype=float)
    blob = _soft_text_blob(df)

    if intent.district_contains and str(intent.district_contains).strip():
        kws = _expand_district_geo_keywords(intent.district_contains)
        if kws:
            score = score + _geo_region_mask(df, kws).astype(float) * _W_GEO
            blob_geo = pd.Series(False, index=df.index)
            for kw in kws:
                blob_geo = blob_geo | blob.str.contains(kw, case=False, na=False, regex=False)
            score = score + blob_geo.astype(float) * (_W_GEO * 0.22)

    if intent.community_contains and str(intent.community_contains).strip():
        kw = intent.community_contains.strip()
        if kw and "community" in df.columns:
            score = score + df["community"].astype(str).str.contains(kw, case=False, na=False).astype(float) * (
                _W_COMMUNITY
            )
        score = score + blob.str.contains(kw, case=False, na=False, regex=False).astype(float) * (_W_COMMUNITY * 0.35)

    if intent.layout_contains and str(intent.layout_contains).strip():
        kw = intent.layout_contains.strip()
        if kw:
            comb = pd.Series(False, index=df.index)
            for col in ("layout", "layout_normalized"):
                if col in df.columns:
                    comb = comb | df[col].astype(str).str.contains(kw, case=False, na=False)
            score = score + comb.astype(float) * _W_LAYOUT
            score = score + blob.str.contains(kw, case=False, na=False, regex=False).astype(float) * (_W_LAYOUT * 0.2)

    # 价格：区间内高分，略超出给弱分，不排除
    up = pd.to_numeric(df["unit_price"], errors="coerce") if "unit_price" in df.columns else None
    tp = pd.to_numeric(df["total_price"], errors="coerce") if "total_price" in df.columns else None
    if up is not None:
        if intent.max_unit_price is not None:
            cap = float(intent.max_unit_price)
            score = score + (up <= cap).astype(float) * _W_PRICE_IN_RANGE
            score = score + ((up > cap) & (up <= cap * 1.2)).astype(float) * _W_PRICE_NEAR_RANGE
        if intent.min_unit_price is not None:
            lo = float(intent.min_unit_price)
            score = score + (up >= lo).astype(float) * (_W_PRICE_IN_RANGE * 0.85)
    if tp is not None:
        if intent.max_total_price_wan is not None:
            cap = float(intent.max_total_price_wan)
            score = score + (tp <= cap).astype(float) * _W_PRICE_IN_RANGE
            score = score + ((tp > cap) & (tp <= cap * 1.15)).astype(float) * _W_PRICE_NEAR_RANGE
        if intent.min_total_price_wan is not None:
            lo = float(intent.min_total_price_wan)
            score = score + (tp >= lo).astype(float) * (_W_PRICE_IN_RANGE * 0.85)

    # 地铁：标签与正文并行， capped
    if intent.near_subway is True:
        sub_tag = pd.Series(0.0, index=df.index)
        if "tag_near_subway" in df.columns:
            sub_tag = df["tag_near_subway"].astype(bool).astype(float) * _W_SUBWAY_TAG
        sub_txt = (
            blob.str.contains("地铁", case=False, na=False, regex=False)
            | blob.str.contains("号线", case=False, na=False, regex=False)
            | blob.str.contains("轨道", case=False, na=False, regex=False)
        ).astype(float) * _W_SUBWAY_TEXT
        score = score + _cap_dim(sub_tag, sub_txt, _W_SUBWAY_DIM_CAP)

    station_kw = (intent.subway_station_contains or "").strip()
    if station_kw:
        m_st = blob.str.contains(station_kw, case=False, na=False, regex=False)
        if "tag_subway_station_hint" in df.columns:
            m_st = m_st | df["tag_subway_station_hint"].astype(str).str.contains(
                station_kw, case=False, na=False, regex=False
            )
        score = score + m_st.astype(float) * _W_SUBWAY_STATION

    if intent.has_balcony is True:
        b_tag = (
            df["tag_has_balcony"].astype(bool).astype(float) * _W_BALCONY_TAG
            if "tag_has_balcony" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        b_txt = (
            blob.str.contains("阳台", case=False, na=False, regex=False)
            | blob.str.contains("露台", case=False, na=False, regex=False)
        ).astype(float) * _W_BALCONY_TEXT
        score = score + _cap_dim(b_tag, b_txt, _W_BALCONY_CAP)

    if intent.lighting_preferred is True:
        l_tag = (
            df["tag_lighting"].astype(bool).astype(float) * _W_LIGHT_TAG
            if "tag_lighting" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        l_txt = (
            blob.str.contains("采光", case=False, na=False, regex=False)
            | blob.str.contains("通透", case=False, na=False, regex=False)
            | blob.str.contains("全明", case=False, na=False, regex=False)
        ).astype(float) * _W_LIGHT_TEXT
        score = score + _cap_dim(l_tag, l_txt, _W_LIGHT_CAP)

    if intent.min_build_year is not None and "build_year" in df.columns:
        by = pd.to_numeric(df["build_year"], errors="coerce")
        mn = int(intent.min_build_year)
        score = score + (by >= mn).astype(float) * _W_YEAR_OK
        score = score + ((by >= mn - 8) & (by < mn)).astype(float) * _W_YEAR_PARTIAL

    parking = _noise_parking_mask(df)
    score = score - parking.astype(float) * _W_NOISE_PARKING

    absurd = _absurd_layout_mask(df)
    score = score - absurd.astype(float) * _W_NOISE_ABSURD_LAYOUT

    if intent.layout_contains and str(intent.layout_contains).strip():
        score = score + _normal_family_layout_mask(df).astype(float) * _W_RESIDENTIAL_FAMILY

    return score


_SCORE_RANK_NOTE = (
    "[查询说明] 以下为按与条件的**相关度打分**排序后的样本（已尽量剔除疑似车位、异常大户型，"
    "并对 2～6 室成套住宅额外加权）。标签与正文关键词均参与计分。"
    "价格等为偏好加权而非硬性剔除（单价/总价仍可在样本中高于您口头区间）。\n"
)


def _candidate_frame_for_intent(df: pd.DataFrame, intent: ListingSearchIntent) -> pd.DataFrame:
    """用户指定城区时先收缩候选集，避免低分无关行占满 max_rows。"""
    if not intent.district_contains or not str(intent.district_contains).strip():
        return df
    kws = _expand_district_geo_keywords(intent.district_contains)
    if not kws:
        return df
    gm = _geo_region_mask(df, kws)
    blob = _soft_text_blob(df)
    blob_geo = pd.Series(False, index=df.index)
    for kw in kws:
        blob_geo = blob_geo | blob.str.contains(kw, case=False, na=False, regex=False)
    m = gm | blob_geo
    if not m.any():
        # 用户已限定城区但无一命中：禁止静默退回全表，否则远郊低价盘会占满 topN
        return df.iloc[0:0].copy()
    return df.loc[m].copy()


def apply_listing_search_intent(
    df: pd.DataFrame, intent: ListingSearchIntent
) -> tuple[pd.DataFrame, str | None]:
    """
    按意图对候选集计算相关度分数，按分数降序取前 max_rows 行。
    标签与正文软匹配并行计分；指定城区时仅在命中城区的行内排序。
    """
    if df.empty or not intent.needs_row_samples:
        return df.iloc[0:0].copy(), None

    work = _candidate_frame_for_intent(df, intent)
    if work.empty and intent.district_contains and str(intent.district_contains).strip():
        return (
            work,
            "[查询说明] 城区关键词「"
            + str(intent.district_contains).strip()
            + "」在 district/位置/标题/正文中未命中任何行；**未**放宽到其它区县，避免全市样本干扰。\n",
        )

    pm = _noise_parking_mask(work)
    am = _absurd_layout_mask(work)
    drop_m = pm | am
    if drop_m.any() and (~drop_m).any():
        work = work.loc[~drop_m].copy()
    if work.empty:
        return (
            work,
            "[查询说明] 当前城区候选经剔除车位/异常户型后无剩余行；可尝试放宽条件或检查数据。\n",
        )

    scores = _intent_relevance_scores(work, intent)
    work["_rel_score"] = scores
    tie = (
        pd.to_numeric(work["unit_price"], errors="coerce")
        if "unit_price" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    work["_tie_cheap"] = tie.fillna(1e18)
    work = work.sort_values(by=["_rel_score", "_tie_cheap"], ascending=[False, True], na_position="last")
    out = work.head(int(intent.max_rows)).drop(columns=["_rel_score", "_tie_cheap"], errors="ignore").copy()
    note = _SCORE_RANK_NOTE.rstrip() if len(out) > 0 else None
    return out, note


def listings_to_llm_block(df: pd.DataFrame) -> str:
    use = [c for c in _LISTING_DISPLAY_COLS if c in df.columns]
    if not use:
        return "（无可用列展示）"
    sub = df[use].copy()
    for c in sub.columns:
        if sub[c].dtype == object or pd.api.types.is_string_dtype(sub[c].dtype):
            sub[c] = sub[c].astype(str).str.slice(0, 160)
    records: list[dict[str, Any]] = sub.to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False, indent=2)


_INTENT_SYSTEM = """你是查询意图解析器。输入中可能包含「用户此前若干段话」+「当前输入」，须**综合理解**。
若此前已指定城区/预算/户型等，而当前句只是在补充「采光」「地铁」「便宜点」等，则必须**继承**前述约束（除非当前句明确改掉，如「换成双流」）。
仅当用户明确要「房源列表、推荐几套、筛选某区/小区/价位、有哪些房子」等需要行级数据时，将 needs_row_samples 设为 true。

价格字段（min_unit_price、max_unit_price、min_total_price_wan、max_total_price_wan）：
仅当用户**明确说出数字或明确区间**（如「单价一万二以下」「总价 150 万以内」）时才填写。
禁止把对话摘要里的 p25、p50、中位数、均价等统计值擅自当成筛选上下限——用户只说「不太贵」「中等价位」等模糊话时，这些价格字段应留空。

地理与地铁（后端按**相关度打分**，非硬性一条条过滤）：
- district_contains：区或板块（高新、天府新区等）；系统在 district/位置/标题/正文等多处软匹配。
- 用户提到地铁、号线、步行到站：near_subway=true；正文含「地铁」「号线」也会加分。若能识别站名，填 subway_station_contains。
- 阳台/露台：has_balcony=true（标签与正文「阳台」等并列加分）。
- 采光/通透/全明：lighting_preferred=true。
- 房龄：min_build_year（整数年份），命中加分。

其它：community_contains、layout_contains 按需填写。
用户表达「三室/四室/适合几口人住」时须填写 layout_contains（如 三室、3室），以便返回成套住宅而非车位等。

若用户只是问统计、政策、概念或与具体表无关的问题，needs_row_samples 必须为 false，其它字段留空。"""


def parse_listing_search_intent(
    user_text: str,
    llm: Any,
    *,
    prior_user_messages: Sequence[str] | None = None,
) -> ListingSearchIntent | None:
    """调用带结构化输出的 LLM；失败时返回 None。prior_user_messages 用于多轮继承城区等约束。"""
    try:
        from langchain_core.messages import HumanMessage

        structured = llm.with_structured_output(ListingSearchIntent)
        parts: list[str] = []
        if prior_user_messages:
            parts.append(
                "以下为同一会话中用户**此前**发过的内容（按时间从旧到新），"
                "用于继承城区（district_contains）、户型、预算意图；当前句未否定则必须保留。\n"
            )
            for i, raw in enumerate(prior_user_messages):
                t = str(raw).strip()
                if not t:
                    continue
                parts.append(f"[此前第{i + 1}段] {t}")
        parts.append(f"[当前输入] {user_text.strip()}")
        payload = "\n".join(parts)
        return structured.invoke(
            [
                HumanMessage(content=_INTENT_SYSTEM),
                HumanMessage(content=payload),
            ]
        )
    except Exception:
        return None
