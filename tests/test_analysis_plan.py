import pandas as pd

from server.core.config import Settings
from server.tools.analysis_plan import (
    AnalysisTaskType,
    PlannedTask,
    execute_task,
    fallback_plan,
    plan_analysis_with_llm,
    run_planned_analysis,
)


def test_fallback_plan_covers_columns() -> None:
    df = pd.DataFrame(
        {
            "district": ["A", "B"],
            "layout": ["2室", "3室"],
            "area_m2": [80.0, 100.0],
            "unit_price": [50000.0, 60000.0],
            "decoration": ["精装", "毛坯"],
            "floor_band": ["低层", "高层"],
            "build_year": [2010, 2005],
            "community": ["X", "Y"],
            "followers": [100.0, 200.0],
            "total_price": [400.0, 600.0],
        }
    )
    tasks = fallback_plan(df)
    types = {t.type for t in tasks}
    assert AnalysisTaskType.district_price_rank in types
    assert AnalysisTaskType.decoration_price_compare in types
    assert AnalysisTaskType.floor_band_price in types
    assert AnalysisTaskType.community_followers_rank in types


def test_execute_decoration_compare() -> None:
    df = pd.DataFrame(
        {
            "decoration": ["精装", "毛坯", "精装", "简装", "精装", "毛坯"],
            "unit_price": [7e4, 6e4, 8e4, 65e3, 72e3, 61e3],
        }
    )
    r = execute_task(df, PlannedTask(type=AnalysisTaskType.decoration_price_compare))
    assert r["ok"] is True
    assert r["chart_kind"] == "bar"


def test_plan_analysis_without_api_key_uses_fallback() -> None:
    df = pd.DataFrame({"district": ["a"], "unit_price": [1.0]})
    tasks, raw = plan_analysis_with_llm(df, Settings(dashscope_api_key=""))
    assert raw == ""
    assert len(tasks) >= 1


def test_run_planned_analysis_has_summary_and_tasks() -> None:
    df = pd.DataFrame(
        {
            "district": ["海淀区"] * 3 + ["朝阳区"] * 3,
            "unit_price": [70000.0, 72000.0, 71000.0, 65000.0, 66000.0, 64000.0],
            "layout": ["2室"] * 4 + ["3室"] * 2,
            "area_m2": [88.0, 92.0, 90.0, 101.0, 99.0, 102.0],
        }
    )
    tasks = fallback_plan(df)
    out = run_planned_analysis(df, tasks)
    assert out["row_count"] == 6
    assert "task_results" in out
    assert out["analysis_summary_markdown"]
    assert any(v.get("ok") for v in out["task_results"].values())
