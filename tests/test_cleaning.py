import pandas as pd

from server.tools.cleaning import apply_default_cleaning_pipeline


def test_apply_default_cleaning_pipeline_basic() -> None:
    df = pd.DataFrame(
        {
            "community": ["A", "A", "B"],
            "layout": ["2室1厅", "2室1厅", "3室2厅"],
            "area_m2": [90.0, 90.0, 110.0],
            "total_price": [600.0, 600.0, 800.0],
            "unit_price": [70000.0, 70000.0, 75000.0],
        }
    )
    out, note = apply_default_cleaning_pipeline(df)
    assert len(out) <= len(df)
    assert "去重" in note or "行数" in note
