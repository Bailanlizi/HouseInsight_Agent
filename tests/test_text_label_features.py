import numpy as np
import pandas as pd

from server.tools.text_label_features import (
    apply_text_label_features,
    normalize_layout_cell,
)


def test_tag_near_subway_and_station_hint() -> None:
    # 出门即地铁4号线凤溪河站 / 普通住宅无地铁词
    df = pd.DataFrame(
        {
            "description_raw": [
                "\u51fa\u95e8\u5373\u5730\u94c14\u53f7\u7ebf\u51e4\u6eaa\u6cb3\u7ad9",
                "\u666e\u901a\u4f4f\u5b85\u7eff\u5316\u597d",
            ],
            "location_raw": ["", ""],
            "listing_title": ["", ""],
        }
    )
    out = apply_text_label_features(df.copy())
    assert bool(out.loc[0, "tag_near_subway"])
    assert "\u51e4\u6eaa\u6cb3" in str(out.loc[0, "tag_subway_station_hint"])
    assert not bool(out.loc[1, "tag_near_subway"])


def test_tag_has_balcony_and_lighting() -> None:
    # 南北通透 带大阳台 / 采光好全明户型
    df = pd.DataFrame(
        {
            "description_raw": [
                "\u5357\u5317\u901a\u900f \u5e26\u5927\u9633\u53f0",
                "",
            ],
            "location_raw": ["", "\u91c7\u5149\u597d\u5168\u660e\u6237\u578b"],
            "listing_title": ["", ""],
        }
    )
    out = apply_text_label_features(df.copy())
    assert bool(out.loc[0, "tag_has_balcony"])
    assert bool(out.loc[0, "tag_lighting"])
    assert bool(out.loc[1, "tag_lighting"])


def test_normalize_layout_aliases() -> None:
    assert normalize_layout_cell("\u4e24\u5ba4\u4e00\u5385") == "2\u5ba41\u5385"
    assert normalize_layout_cell("3\u5ba42\u5385") == "3\u5ba42\u5385"
    assert normalize_layout_cell("\u4e8c\u623f\u4e00\u5385") == "2\u5ba41\u5385"


def test_near_subway_recall_and_bus_exclusion() -> None:
    """地铁房/TOD/步行到地铁/距站米；排除纯公交步行与否定句。"""
    df = pd.DataFrame(
        {
            "description_raw": [
                "\u5730\u94c1\u623f\u7cbe\u88c5",  # 地铁房精装
                "TOD\u7efc\u5408\u4f53",  # TOD综合体
                "\u6b65\u884c8\u5206\u949f\u5230\u5730\u94c1\u7ad9",  # 步行8分钟到地铁站
                "\u8ddd\u5357\u718f\u5927\u9053\u7ad9600\u7c73",  # 距南熏大道站600米
                "\u6b65\u884c5\u5206\u949f\u5230\u516c\u4ea4\u7ad9",  # 步行5分钟到公交站
                "\u65e0\u5730\u94c1\u516c\u4ea4\u4fbf\u5229",  # 无地铁公交便利
            ],
            "location_raw": ["", "", "", "", "", ""],
            "listing_title": ["", "", "", "", "", ""],
        }
    )
    out = apply_text_label_features(df.copy())
    assert bool(out.loc[0, "tag_near_subway"])
    assert bool(out.loc[1, "tag_near_subway"])
    assert bool(out.loc[2, "tag_near_subway"])
    assert bool(out.loc[3, "tag_near_subway"])
    assert "\u5357\u718f\u5927\u9053" in str(out.loc[3, "tag_subway_station_hint"])
    assert not bool(out.loc[4, "tag_near_subway"])
    assert not bool(out.loc[5, "tag_near_subway"])


def test_extended_balcony_lighting_keywords() -> None:
    df = pd.DataFrame(
        {
            "description_raw": ["\u53cc\u9633\u53f0\u6237\u578b", ""],  # 双阳台户型
            "location_raw": ["", "\u4eae\u5802\u5357\u5411"],  # 亮堂南向
            "listing_title": ["", ""],
        }
    )
    out = apply_text_label_features(df.copy())
    assert bool(out.loc[0, "tag_has_balcony"])
    assert bool(out.loc[1, "tag_lighting"])


def test_build_year_backfill_from_text() -> None:
    df = pd.DataFrame(
        {
            "description_raw": ["\u5efa\u4e8e2012\u5e74 \u7cbe\u88c5", ""],
            "build_year": [np.nan, np.nan],
            "location_raw": ["", "1998\u5e74\u5c0f\u533a"],
            "listing_title": ["", ""],
        }
    )
    out = apply_text_label_features(df.copy())
    assert int(out.loc[0, "build_year"]) == 2012
    assert int(out.loc[1, "build_year"]) == 1998
