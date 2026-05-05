import pandas as pd

from server.tools.chat_listing_query import (
    ListingSearchIntent,
    _expand_district_geo_keywords,
    apply_listing_search_intent,
    listings_to_llm_block,
)


def test_expand_district_keywords_suffix_any_city() -> None:
    """「区」与不带「区」互列为关键词，不限定单一城市。"""
    kws = _expand_district_geo_keywords("新都区")
    assert "新都" in kws and "新都区" in kws
    kws2 = _expand_district_geo_keywords("锦江")
    assert "锦江" in kws2 and "锦江区" in kws2


def test_district_short_form_row_matches_user_saying_full_suffix() -> None:
    """表中 district 只有「新都」时，用户说「新都区」也应命中候选。"""
    df = pd.DataFrame(
        {
            "district": ["新都", "新都", "武侯区"],
            "location_raw": ["", "", ""],
            "listing_title": ["", "", ""],
            "community": ["", "", ""],
            "description_raw": ["", "", ""],
            "house_info_raw": ["", "", ""],
            "follow_info_raw": ["", "", ""],
            "unit_price": [1.0, 2.0, 3.0],
        }
    )
    intent = ListingSearchIntent(needs_row_samples=True, district_contains="新都区", max_rows=10)
    out, _ = apply_listing_search_intent(df, intent)
    assert len(out) == 2


def test_hard_require_tags_intersection_like_excel_and() -> None:
    """必选标签：pandas 硬筛 AND，再排序取 TopN。"""
    df = pd.DataFrame(
        {
            "district": ["新都区", "新都区", "新都区"],
            "location_raw": ["", "", ""],
            "listing_title": ["", "", ""],
            "community": ["A", "B", "C"],
            "description_raw": ["", "", ""],
            "house_info_raw": ["", "", ""],
            "follow_info_raw": ["", "", ""],
            "tag_near_subway": [True, True, False],
            "tag_lighting": [True, False, True],
            "unit_price": [10000.0, 9000.0, 8000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="新都",
        require_tag_near_subway=True,
        require_tag_lighting=True,
        max_rows=10,
    )
    out, note = apply_listing_search_intent(df, intent)
    assert len(out) == 1
    assert out.iloc[0]["community"] == "A"
    assert note is not None
    assert "硬条件" in note or "硬筛选" in note


def test_district_no_geo_match_returns_empty_not_city_wide() -> None:
    """限定城区但数据中无一命中时不得退回全市（否则会都江堰刷屏）。"""
    df = pd.DataFrame(
        {
            "district": ["锦江区"],
            "location_raw": [""],
            "listing_title": [""],
            "community": [""],
            "description_raw": [""],
            "house_info_raw": [""],
            "follow_info_raw": [""],
            "unit_price": [1.0],
        }
    )
    intent = ListingSearchIntent(needs_row_samples=True, district_contains="成华", max_rows=10)
    out, note = apply_listing_search_intent(df, intent)
    assert len(out) == 0
    assert note is not None
    assert "未命中" in note or "未放宽" in note


def test_apply_district_filter() -> None:
    df = pd.DataFrame(
        {
            "district": ["温江区", "锦江区", "温江区"],
            "community": ["A", "B", "C"],
            "unit_price": [10000.0, 20000.0, 12000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="温江",
        max_rows=10,
    )
    out, note = apply_listing_search_intent(df, intent)
    assert note is not None and "相关度" in note
    assert len(out) == 2
    assert all("温江" in str(x) for x in out["district"])


def test_needs_row_samples_false_returns_empty() -> None:
    df = pd.DataFrame({"district": ["A"]})
    intent = ListingSearchIntent(needs_row_samples=False)
    out, _ = apply_listing_search_intent(df, intent)
    assert len(out) == 0


def test_listings_to_llm_block_json() -> None:
    df = pd.DataFrame({"district": ["X"], "unit_price": [1.0]})
    s = listings_to_llm_block(df)
    assert "district" in s
    assert "X" in s


def test_near_subway_and_layout_normalized_filter() -> None:
    df = pd.DataFrame(
        {
            "district": ["温江区", "温江区"],
            "layout": ["两室一厅", "三室两厅"],
            "layout_normalized": ["2室1厅", "3室2厅"],
            "tag_near_subway": [True, False],
            "tag_subway_station_hint": ["凤溪河", ""],
            "build_year": [2010, 2000],
            "unit_price": [10000.0, 12000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="温江",
        near_subway=True,
        layout_contains="2室",
        min_build_year=2005,
        max_rows=10,
    )
    out, note = apply_listing_search_intent(df, intent)
    assert len(out) == 2
    assert out.iloc[0]["layout_normalized"] == "2室1厅"
    assert note is not None


def test_district_geo_matches_location_raw_not_only_district() -> None:
    """「高新」写在 location_raw 而 district 为武侯时仍能命中。"""
    df = pd.DataFrame(
        {
            "district": ["武侯区", "锦江区"],
            "location_raw": ["大源\u677f\u5757\u6210\u90fd\u9ad8\u65b0\u5357\u533a", ""],
            "listing_title": ["", ""],
            "community": ["", ""],
            "unit_price": [11000.0, 9000.0],
        }
    )
    intent = ListingSearchIntent(needs_row_samples=True, district_contains="高新", max_rows=10)
    out, note = apply_listing_search_intent(df, intent)
    assert len(out) == 1
    assert "武侯" in str(out.iloc[0]["district"])
    assert note is not None


def test_scoring_keeps_row_when_price_above_cap() -> None:
    """打分模式下单价超出上限仍会返回样本（偏好降权而非剔除）。"""
    df = pd.DataFrame(
        {
            "district": ["武侯区"],
            "location_raw": ["\u6210\u90fd\u9ad8\u65b0"],
            "listing_title": [""],
            "community": [""],
            "description_raw": [""],
            "house_info_raw": [""],
            "follow_info_raw": [""],
            "tag_near_subway": [True],
            "unit_price": [15000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="高新",
        near_subway=True,
        max_unit_price=10000.0,
        max_rows=10,
    )
    out, note = apply_listing_search_intent(df, intent)
    assert len(out) == 1
    assert note is not None
    assert "\u76f8\u5173\u5ea6" in note


def test_relax_drops_station_hint_keeps_near_subway() -> None:
    df = pd.DataFrame(
        {
            "district": ["温江区", "温江区"],
            "tag_near_subway": [True, True],
            "tag_subway_station_hint": ["南熏大道", "南熏大道"],
            "unit_price": [1.0, 2.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="温江",
        near_subway=True,
        subway_station_contains="不存在的站",
        max_rows=10,
    )
    out, note = apply_listing_search_intent(df, intent)
    assert len(out) == 2
    assert note is not None


def test_parking_rows_dropped_before_ranking() -> None:
    """含「车位」的挂牌应从候选中剔除，避免占满前 N 条。"""
    df = pd.DataFrame(
        {
            "district": ["高新区"] * 3,
            "location_raw": ["", "", ""],
            "listing_title": ["", "", ""],
            "community": ["A", "B", "C"],
            "description_raw": ["产权车位转让", "精装三室", "地铁旁"],
            "house_info_raw": ["", "3室2厅 | 95平米", ""],
            "follow_info_raw": ["", "", ""],
            "layout_normalized": ["", "3室2厅", ""],
            "layout": ["", "3室2厅", ""],
            "tag_near_subway": [False, True, True],
            "unit_price": [5000.0, 18000.0, 17000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="高新",
        near_subway=True,
        layout_contains="3室",
        max_rows=5,
    )
    out, _ = apply_listing_search_intent(df, intent)
    assert len(out) == 2
    assert "B" in set(out["community"])
    assert "A" not in set(out["community"])


def test_absurd_layout_dropped_or_heavily_penalized() -> None:
    """异常多「室」户型不与正常住宅抢前排。"""
    df = pd.DataFrame(
        {
            "district": ["高新", "高新"],
            "location_raw": ["", ""],
            "listing_title": ["", ""],
            "community": ["X", "Y"],
            "description_raw": ["", ""],
            "house_info_raw": ["20室9厅 | 500平米", "3室2厅 | 96平米"],
            "follow_info_raw": ["", ""],
            "layout_normalized": ["20室9厅", "3室2厅"],
            "layout": ["20室9厅", "3室2厅"],
            "tag_near_subway": [True, True],
            "unit_price": [8000.0, 20000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="高新",
        layout_contains="3室",
        max_rows=5,
    )
    out, _ = apply_listing_search_intent(df, intent)
    assert len(out) >= 1
    assert out.iloc[0]["community"] == "Y"


def test_scoring_subway_text_when_tag_false() -> None:
    """正文含地铁关键词时，即便 tag_near_subway 为 False 也应排在同类前列。"""
    df = pd.DataFrame(
        {
            "district": ["高新区", "高新区"],
            "location_raw": ["", ""],
            "listing_title": ["", ""],
            "community": ["A", "B"],
            "description_raw": ["临近地铁1号线", "南北通透户型"],
            "house_info_raw": ["", ""],
            "follow_info_raw": ["", ""],
            "tag_near_subway": [False, False],
            "unit_price": [20000.0, 10000.0],
        }
    )
    intent = ListingSearchIntent(
        needs_row_samples=True,
        district_contains="高新",
        near_subway=True,
        max_rows=2,
    )
    out, _ = apply_listing_search_intent(df, intent)
    assert len(out) == 2
    assert out.iloc[0]["community"] == "A"
