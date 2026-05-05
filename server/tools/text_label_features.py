"""从描述/位置/标题等文本用正则与词表抽取结构化标签（不调用 LLM）。"""

from __future__ import annotations

import re

import pandas as pd

# 成都/温江及周边常见地铁站名（用于首个匹配 hint；可按需扩展）
CHENGDU_SUBWAY_STATION_HINTS: tuple[str, ...] = (
    "凤溪河",
    "南熏大道",
    "光华公园",
    "涌泉",
    "杨柳河",
    "市五医院",
    "金星",
    "明光",
    "非遗博览园",
    "黄田坝",
    "文化宫",
    "中医大省医院",
    "天府广场",
    "春熙路",
    "世纪城",
    "孵化园",
    "金融城",
    "交子大道",
    "石羊",
    "三元",
    "太平园",
    "红牌楼",
    "高升桥",
    "省体育馆",
    "倪家桥",
    "桐梓林",
    "火车南站",
    "三瓦窑",
    "琉璃场",
    "东大路",
    "牛市口",
    "牛王庙",
    "东门大桥",
    "红星桥",
    "前锋路",
    "梁家巷",
    "人民北路",
    "西北桥",
    "九里堤",
    "西南交大",
    "茶店子",
    "羊犀立交",
    "一品天下",
    "金沙博物馆",
    "清江西路",
    "成都西站",
    "中坝",
    "蔡桥",
    "凤凰大街",
    "温泉大道",
    "九江北",
    "白佛桥",
    "康河",
    "万盛",
)

# 近地铁：关键词 + 距站米/步行到轨交/地铁侧米数；否定句压低误报；避免「步行…公交站」「距公交站米」
_RE_SUBWAY_NEG = re.compile(
    r"无地铁|未通地铁|未有地铁|没有地铁|非地铁|远离地铁|"
    r"暂不[临]?地铁|暂无地铁|未临地铁|暂无\s*地铁|不临地铁",
    re.I,
)
_RE_SUBWAY_KW = re.compile(
    r"地铁[房口旁上盖物业站]?|近地铁|临地铁|紧邻地铁|临近地铁|靠近地铁|"
    r"距地铁|出门[即就是]?地铁|轨交|轨道交通|轻轨|号\s*线|轨道\s*交通|"
    r"地铁\s*上盖|地铁\s*直连|地铁\s*入户|地铁\s*出行|地铁\s*通勤|"
    r"距\s*\d{1,2}\s*号?\s*线|"
    r"(?:^|[^a-z])tod(?:$|[^a-z])|"
    r"tod项目|地铁tod|轨交tod",
    re.I,
)
_RE_SUBWAY_DIST_STATION_M = re.compile(
    r"距(?!.{0,14}?(?:公交|巴士|汽车|客运|高铁|动车))"
    r"(?!.{0,14}?(?:火车\s*站|高铁站|火车站|汽车站|客运站))"
    r".{1,22}?"
    r"站\s*\d{2,6}\s*[mM米]",
    re.I,
)
_RE_SUBWAY_DIST_METRO_M = re.compile(
    r"距.{0,24}?(?:地铁|[\d一二三四五六七八九十两〇○百零]+号\s*线).{0,20}?\d{2,6}\s*[mM米]",
    re.I,
)
_RE_SUBWAY_METRO_INLINE_M = re.compile(
    r"地铁.{0,18}\d{2,5}\s*[mM米]|[\d一二三四五六七八九十两〇○百零]+号\s*线.{0,16}\d{2,5}\s*[mM米]",
    re.I,
)
_RE_SUBWAY_WALK = re.compile(
    r"步行\s*\d{1,3}\s*分钟(?:内\s*)?[到至]?"
    r"[^\n。,，;；]{0,26}(?:地铁|号\s*线|轻轨|地铁站|地铁口|轨道交通|轨交)",
    re.I,
)
_RE_BUS = re.compile(r"公交|公交车站|公交站|巴士站|临近\s*公交", re.I)
_RE_BALCONY = re.compile(r"双阳台|生活阳台|观景阳台|阳台|露台|观景台|外挑", re.I)
_RE_LIGHTING = re.compile(
    r"采光|通透|全明|明厨|明卫|视野好|无遮挡|亮堂|日照充足|自然光",
    re.I,
)
_RE_ELEVATOR_YES = re.compile(r"电梯房|有电梯|电梯|梯户|一梯一户|两梯四户", re.I)
_RE_ELEVATOR_NO = re.compile(r"步梯|无电梯|多层", re.I)
_RE_YEAR = re.compile(r"(19\d{2}|20[0-3]\d)")
_RE_LAYOUT_STD = re.compile(r"(\d+)\s*室\s*(\d+)\s*厅")


def _concat_label_text(df: pd.DataFrame) -> pd.Series:
    cols = ("description_raw", "location_raw", "listing_title", "house_info_raw")
    parts: list[pd.Series] = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    s = parts[0].copy()
    for p in parts[1:]:
        s = s + " " + p
    return s


def normalize_layout_cell(value: object) -> str:
    """将常见户型别名归一为「N室M厅」形式；无法识别则返回去空白后的原文截断。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    t = str(value).strip()
    if not t:
        return ""
    m = _RE_LAYOUT_STD.search(t.replace(" ", ""))
    if m:
        return f"{int(m.group(1))}室{int(m.group(2))}厅"
    tl = t.lower()
    # 两居/二房/2房 → 2 室
    n_rooms: int | None = None
    if re.search(r"两居|二居|两房|二房|两室|二室|2\s*居|2\s*房", tl):
        n_rooms = 2
    elif re.search(r"三居|三房|3\s*居|3\s*房|三室", tl):
        n_rooms = 3
    elif re.search(r"一居|一房|1\s*居|1\s*房|一室", tl):
        n_rooms = 1
    elif re.search(r"四居|四房|4\s*居|4\s*房|四室", tl):
        n_rooms = 4
    if n_rooms is not None:
        hm = re.search(r"(\d+)\s*厅", t)
        halls = int(hm.group(1)) if hm else 1
        return f"{n_rooms}室{halls}厅"
    return t[:80] if len(t) > 80 else t


def _station_pattern() -> str:
    # 长站名优先：按长度降序，避免短名抢先匹配
    ordered = sorted(CHENGDU_SUBWAY_STATION_HINTS, key=len, reverse=True)
    return "|".join(re.escape(x) for x in ordered)


def _near_subway_mask(text: pd.Series) -> pd.Series:
    """合并多类正例；否定句单独剔除。"""
    tl = text.str.lower()
    pos = (
        tl.str.contains(_RE_SUBWAY_KW, regex=True, na=False)
        | tl.str.contains(_RE_SUBWAY_DIST_STATION_M, regex=True, na=False)
        | tl.str.contains(_RE_SUBWAY_DIST_METRO_M, regex=True, na=False)
        | tl.str.contains(_RE_SUBWAY_METRO_INLINE_M, regex=True, na=False)
        | tl.str.contains(_RE_SUBWAY_WALK, regex=True, na=False)
    )
    neg = tl.str.contains(_RE_SUBWAY_NEG, regex=True, na=False)
    return pos & ~neg


def apply_text_label_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    写入 tag_*、layout_normalized；对 build_year 仍缺失时从拼接文本中提取四位年份。
    原地修改 df 的副本：调用方应传入 copy。
    """
    out = df
    text = _concat_label_text(out)
    text_lower = text.str.lower()

    out["tag_near_subway"] = _near_subway_mask(text)
    out["tag_near_bus"] = text_lower.str.contains(_RE_BUS, regex=True, na=False)
    out["tag_has_balcony"] = text_lower.str.contains(_RE_BALCONY, regex=True, na=False)
    out["tag_lighting"] = text_lower.str.contains(_RE_LIGHTING, regex=True, na=False)

    pat = _station_pattern()
    extracted = text.str.extract(f"({pat})", expand=False)
    hint = extracted.fillna("").astype(str)
    hint = hint.mask(hint.str.lower() == "nan", "")
    out["tag_subway_station_hint"] = hint

    el_yes = text_lower.str.contains(_RE_ELEVATOR_YES, regex=True, na=False)
    el_no = text_lower.str.contains(_RE_ELEVATOR_NO, regex=True, na=False)
    out["tag_elevator"] = False
    out.loc[el_yes, "tag_elevator"] = True
    out.loc[el_no & ~el_yes, "tag_elevator"] = False

    if "layout" in out.columns:
        out["layout_normalized"] = out["layout"].map(normalize_layout_cell)
    else:
        out["layout_normalized"] = ""

    if "build_year" in out.columns:
        by = pd.to_numeric(out["build_year"], errors="coerce")
        need = by.isna()
        if need.any():
            years = text.str.extract(_RE_YEAR, expand=False)
            ynum = pd.to_numeric(years, errors="coerce")
            fill_mask = need & ynum.notna() & (ynum >= 1900) & (ynum <= 2035)
            out.loc[fill_mask, "build_year"] = ynum.loc[fill_mask].astype(int)
    return out
