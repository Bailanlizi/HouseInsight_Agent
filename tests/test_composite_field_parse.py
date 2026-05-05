import pandas as pd

from server.tools.cleaning import apply_default_cleaning_pipeline
from server.tools.composite_field_parse import expand_composite_listing_columns


def test_expand_house_info_pipe_delimited() -> None:
    df = pd.DataFrame(
        {
            "house_info_raw": [
                "3室2厅|89平米|南|精装|中楼层(共26层)|2016年建|板楼",
            ],
            "total_price": [200.0],
            "unit_price": [22000.0],
        }
    )
    out = expand_composite_listing_columns(df)
    assert out.loc[0, "layout_str"] == "3室2厅"
    assert "89" in str(out.loc[0, "area_m2_str"])
    assert out.loc[0, "orientation_str"] == "南"
    assert out.loc[0, "decoration_str"] == "精装"
    assert "中楼层" in str(out.loc[0, "floor_text"])
    assert str(out.loc[0, "build_year"]) == "2016"
    assert out.loc[0, "building_type_str"] == "板楼"


def test_expand_follow_info_slash() -> None:
    df = pd.DataFrame({"follow_info_raw": ["6人关注 / 11天以前发布"]})
    out = expand_composite_listing_columns(df)
    assert float(out.loc[0, "followers"]) == 6.0
    assert "11天以前发布" in str(out.loc[0, "listing_time"]) or "11天以前发布" in str(out.loc[0, "publish_time_raw"])


def test_apply_default_pipeline_runs_expand() -> None:
    df = pd.DataFrame(
        {
            "district": ["DemoDistrict"],
            "house_info_raw": ["2室1厅|75平米|北|简装|低楼层(共6层)|2010年建|砖混"],
            "follow_info_raw": ["12人关注/3个月以前发布"],
            "total_price": [100.0],
            "area_m2": [75.0],
            "unit_price": [13000.0],
        }
    )
    out, _note = apply_default_cleaning_pipeline(df)
    assert "layout" in out.columns or "layout_str" in out.columns
    up = pd.to_numeric(out["unit_price"], errors="coerce")
    assert up.notna().any()


def test_expand_does_not_overwrite_existing_layout() -> None:
    df = pd.DataFrame(
        {
            "layout": ["4室2厅"],
            "house_info_raw": ["1室1厅|40平米|南|毛坯|低层|2000年建|塔楼"],
        }
    )
    out = expand_composite_listing_columns(df)
    assert str(out.loc[0, "layout"]).strip() == "4室2厅"
