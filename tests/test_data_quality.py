import pandas as pd

from server.core.config import Settings
from server.tools.data_quality import assess_clean_quality


def _settings() -> Settings:
    return Settings(
        dashscope_api_key="",
        quality_min_rows=10,
        quality_min_retention_ratio=0.02,
        quality_min_unit_price_coverage=0.15,
        quality_min_geo_coverage=0.15,
    )


def test_assess_passes_on_typical_clean_sample() -> None:
    raw = pd.DataFrame({"x": range(40)})
    clean = pd.DataFrame(
        {
            "district": ["海淀区"] * 40,
            "unit_price": [50000.0 + i for i in range(40)],
            "total_price": [400.0] * 40,
            "area_m2": [80.0] * 40,
        }
    )
    r = assess_clean_quality(raw, clean, _settings())
    assert r["passed"] is True
    assert r["metrics"]["clean_rows"] == 40


def test_assess_fails_retention_critical() -> None:
    raw = pd.DataFrame({"x": range(200)})
    clean = pd.DataFrame(
        {
            "district": ["a"] * 3,
            "unit_price": [5000.0] * 3,
            "total_price": [50.0] * 3,
            "area_m2": [80.0] * 3,
        }
    )
    r = assess_clean_quality(raw, clean, _settings())
    assert r["passed"] is False
    assert "row_retention_critical" in r["failures"]


def test_assess_passes_with_price_warning_only() -> None:
    raw = pd.DataFrame({"x": range(40)})
    clean = pd.DataFrame(
        {
            "district": ["a"] * 40,
            "unit_price": [1.0] * 40,
            "total_price": [400.0] * 40,
            "area_m2": [80.0] * 40,
        }
    )
    r = assess_clean_quality(raw, clean, _settings())
    assert r["passed"] is True
    codes = r.get("warning_codes") or []
    assert "unit_price_sparse" in codes or "price_fields_weak" in codes


def test_tag_near_subway_sparse_warning() -> None:
    raw = pd.DataFrame({"x": range(40)})
    clean = pd.DataFrame(
        {
            "district": ["a"] * 40,
            "unit_price": [50000.0] * 40,
            "total_price": [400.0] * 40,
            "area_m2": [80.0] * 40,
            "tag_near_subway": [False] * 40,
        }
    )
    r = assess_clean_quality(raw, clean, _settings())
    assert r["passed"] is True
    assert "tag_near_subway_sparse" in (r.get("warning_codes") or [])
    assert r["metrics"].get("tag_near_subway_true_ratio") == 0.0


def test_assess_fails_when_clean_empty() -> None:
    raw = pd.DataFrame({"x": range(10)})
    clean = pd.DataFrame()
    r = assess_clean_quality(raw, clean, _settings())
    assert r["passed"] is False
    assert "clean_empty" in r["failures"]
