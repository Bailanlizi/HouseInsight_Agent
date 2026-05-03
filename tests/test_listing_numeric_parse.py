import pandas as pd

from server.tools.listing_numeric_parse import (
    finalize_listing_dataframe,
    parse_area_m2,
    parse_total_price_wan,
    parse_unit_price_yuan,
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
