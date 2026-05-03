import pandas as pd

from server.tools.cleaning import apply_default_cleaning_pipeline, resolve_listing_dedup_subset


def test_resolve_dedup_subset_requires_two_informative_columns() -> None:
    df = pd.DataFrame({"total_price": [100.0] * 50, "desc": range(50)})
    assert resolve_listing_dedup_subset(df) is None


def test_resolve_dedup_subset_with_district_and_price() -> None:
    df = pd.DataFrame(
        {
            "district": ["金堂"] * 20,
            "total_price": [80.0] * 20,
        }
    )
    assert set(resolve_listing_dedup_subset(df) or []) == {"district", "total_price"}


def test_default_pipeline_no_single_column_collapse() -> None:
    """缺少户型/面积/小区时，不得仅按总价去重把多行压成一行。"""
    df = pd.DataFrame(
        {
            "district": ["金堂"] * 100,
            "total_price": [85.5] * 100,
            "house_info_raw": [f"房源{i}" for i in range(100)],
        }
    )
    out, note = apply_default_cleaning_pipeline(df)
    assert len(out) == 100, note
    assert "去重" in note
