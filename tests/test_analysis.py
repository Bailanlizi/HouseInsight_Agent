import pandas as pd

from server.tools.analysis import analyze_second_hand_listings


def test_analyze_second_hand_listings_smoke() -> None:
    df = pd.DataFrame(
        {
            "district": ["海淀区", "海淀区", "朝阳区"],
            "unit_price": [70000.0, 72000.0, 65000.0],
            "area_m2": [88.0, 92.0, 101.0],
            "build_year": [2005, 2012, 1999],
        }
    )
    out = analyze_second_hand_listings(df)
    assert out["row_count"] == 3
    assert "district_summary" in out
    assert "unit_price_quantiles" in out
