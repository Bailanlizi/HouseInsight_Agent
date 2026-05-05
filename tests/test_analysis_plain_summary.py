import pandas as pd

from server.tools.analysis_plain_summary import build_analysis_plain_summary


def test_plain_summary_length_bounds() -> None:
    df = pd.DataFrame(
        {
            "district": ["A区", "B区"],
            "unit_price": [10000.0, 20000.0],
            "layout": ["2室1厅", "3室2厅"],
        }
    )
    analysis = {
        "row_count": 2,
        "unit_price_quantiles": {"min": 1e4, "p25": 1.2e4, "p50": 1.5e4, "p75": 1.8e4, "max": 2e4},
        "district_summary": [
            {"district": "A区", "listings": 1, "avg_unit_price": 10000.0},
            {"district": "B区", "listings": 1, "avg_unit_price": 20000.0},
        ],
        "area_buckets": {"60-90": 1, "90-120": 1},
    }
    tr = {"t1": {"ok": True}, "t2": {"ok": False, "reason": "缺列"}}
    text = build_analysis_plain_summary(df, analysis, tr)
    assert 300 <= len(text) <= 500
    assert "数据概览" in text or "样本" in text
