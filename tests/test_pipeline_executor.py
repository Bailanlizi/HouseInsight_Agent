"""在无 LLM 密钥时跑通整条 LangGraph（默认规则清洗）。"""

from pathlib import Path

import pandas as pd

from server.agent.house_agent import build_pipeline_graph
from server.core.config import Settings
from server.core.paths import ProjectPaths
from server.core.session_store import SessionStore


def test_pipeline_end_to_end_without_llm(tmp_path: Path) -> None:
    root = tmp_path
    raw = root / "data" / "raw" / "s1"
    raw.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "district": ["海淀区", "海淀区"],
            "community": ["X", "X"],
            "layout": ["2室1厅", "2室1厅"],
            "area_m2": [90.0, 90.0],
            "total_price": [650.0, 650.0],
            "unit_price": [72000.0, 72000.0],
        }
    )
    df.to_csv(raw / "a.csv", index=False, encoding="utf-8-sig")

    paths = ProjectPaths(root)
    store = SessionStore()
    st = store.create_session()
    sid = st.session_id
    dest_raw = paths.raw_dir(sid)
    pd.read_csv(raw / "a.csv", encoding="utf-8-sig").to_csv(dest_raw / "a.csv", index=False, encoding="utf-8-sig")

    settings = Settings(dashscope_api_key="")
    g = build_pipeline_graph(store, paths, settings)
    g.invoke({"session_id": sid})

    st2 = store.require(sid)
    assert st2.stage == "done"
    assert "report.xlsx" in st2.artifacts
    assert Path(st2.artifacts["report.xlsx"]).exists()
    assert "report.html" not in st2.artifacts
    assert "report.pdf" not in st2.artifacts


def test_cleaned_csv_when_flag_enabled(tmp_path: Path) -> None:
    root = tmp_path
    raw = root / "data" / "raw" / "s1"
    raw.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "district": ["海淀区", "海淀区"],
            "community": ["X", "X"],
            "layout": ["2室1厅", "2室1厅"],
            "area_m2": [90.0, 90.0],
            "total_price": [650.0, 650.0],
            "unit_price": [72000.0, 72000.0],
        }
    )
    df.to_csv(raw / "b.csv", index=False, encoding="utf-8-sig")

    paths = ProjectPaths(root)
    store = SessionStore()
    st = store.create_session()
    sid = st.session_id
    st.return_cleaned_file = True
    dest_raw = paths.raw_dir(sid)
    pd.read_csv(raw / "b.csv", encoding="utf-8-sig").to_csv(dest_raw / "b.csv", index=False, encoding="utf-8-sig")

    settings = Settings(dashscope_api_key="")
    g = build_pipeline_graph(store, paths, settings)
    g.invoke({"session_id": sid})

    st2 = store.require(sid)
    assert st2.stage == "done"
    assert "cleaned.csv" in st2.artifacts
    assert Path(st2.artifacts["cleaned.csv"]).exists()
    assert "report.html" not in st2.artifacts


def test_main_app_import() -> None:
    from server.main import app as real_app

    assert real_app.title == "HouseInsight Agent"
