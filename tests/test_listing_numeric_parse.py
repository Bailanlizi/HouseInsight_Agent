import pandas as pd

from server.tools.listing_numeric_parse import (
    finalize_listing_dataframe,
    parse_area_m2,
    parse_total_price_wan,
    parse_unit_price_yuan,
    slim_cleaned_export_dataframe,
)


def test_parse_chengdu_style_strings() -> None:
    assert abs(parse_unit_price_yuan("17190元/平米") - 17190.0) < 0.01
    assert abs(parse_total_price_wan("153万") - 153.0) < 0.01
    assert abs(parse_area_m2("89平米") - 89.0) < 0.01


def test_finalize_promotes_area_str_and_fills_unit_price() -> None:
    df = pd.DataFrame(
        {
            "total_price": ["153万"],
            "area_m2_str": ["89平米"],
            "unit_price": [pd.NA],
        }
    )
    out = finalize_listing_dataframe(df)
    assert abs(float(out.loc[0, "area_m2"]) - 89.0) < 0.01
    assert abs(float(out.loc[0, "total_price"]) - 153.0) < 0.01
    derived = 153 * 10_000 / 89
    assert abs(float(out.loc[0, "unit_price"]) - derived) < 1.0


def test_slim_cleaned_export_drops_redundant_staging() -> None:
    df = pd.DataFrame(
        {
            "district": ["高新"],
            "area_m2": [91.68],
            "area_m2_str": ["91.68平米"],
            "layout": ["3室2厅"],
            "layout_normalized": ["3室2厅"],
            "orientation": ["东北"],
            "orientation_str": ["东北"],
            "floor": ["高楼层(共32层)"],
            "floor_text": ["高楼层(共32层)"],
            "listing_time": ["一年前发布"],
            "publish_time_raw": ["一年前发布"],
            "house_info_raw": ["…"],
        }
    )
    slim = slim_cleaned_export_dataframe(df)
    assert "area_m2_str" not in slim.columns
    assert "layout" not in slim.columns
    assert "layout_normalized" in slim.columns
    assert "orientation_str" not in slim.columns
    assert "floor_text" not in slim.columns
    assert "publish_time_raw" not in slim.columns
    assert "area_m2" in slim.columns and slim.loc[0, "area_m2"] == 91.68


def test_slim_cleaned_export_keeps_staging_when_no_canonical() -> None:
    df = pd.DataFrame({"area_m2_str": ["90平米"], "x": [1]})
    slim = slim_cleaned_export_dataframe(df)
    assert "area_m2_str" in slim.columns
    assert "x" in slim.columns
