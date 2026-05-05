import numpy as np
import pandas as pd

from server.core.config import Settings
from server.pipeline.listing_etl import run_listing_etl
from server.tools.analysis_plan import compact_plan_payload
from server.tools.dataset_profile import build_dataset_profile


def test_run_listing_etl_smoke() -> None:
    df = pd.DataFrame(
        {
            "district": ["A", "A"],
            "house_info_raw": ["2室1厅|90平米|南|精装|中层(共20层)|2015年建|板楼", "2室1厅|90平米|南|精装|中层(共20层)|2015年建|板楼"],
            "total_price": [300.0, 300.0],
            "area_m2": [90.0, 90.0],
            "unit_price": [33000.0, 33000.0],
        }
    )
    out, note = run_listing_etl(df)
    assert len(out) >= 1
    assert "去重" in note


def test_promote_skips_when_staging_all_na_for_float_target() -> None:
    """目标 area_m2 全为 NaN 且 area_m2_str 也全 NA 时不应抛错。"""
    df = pd.DataFrame(
        {
            "listing_id": ["a", "b"],
            "area_m2": [np.nan, np.nan],
            "area_m2_str": [pd.NA, pd.NA],
        }
    )
    out, _ = run_listing_etl(df)
    assert len(out) == 2


def test_compact_plan_payload_truncates_columns() -> None:
    wide = pd.DataFrame({f"c{i}": [1] for i in range(80)})
    settings = Settings(houseinsight_plan_profile_max_cols=10)
    prof = build_dataset_profile(wide, sample_per_col=1)
    payload = compact_plan_payload(wide, prof, settings)
    cols = payload["profile"]["columns"]
    assert len(cols) <= 10
    assert payload["profile"].get("profile_truncated") is True
