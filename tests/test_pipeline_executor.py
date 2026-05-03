"""在无 LLM 密钥时跑通整条 LangGraph（默认规则清洗）。"""

import shutil
from pathlib import Path

import pandas as pd

from server.agent.house_agent import build_pipeline_graph
from server.core.config import Settings
from server.core.paths import ProjectPaths
from server.core.session_store import SessionStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_end_to_end_without_llm(tmp_path: Path) -> None:
    root = tmp_path
    tpl_dir = root / "templates"
    tpl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "templates" / "report_template.html", tpl_dir / "report_template.html")
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
    # 把文件写到期望 raw 目录（模拟上传）
    dest_raw = paths.raw_dir(sid)
    pd.read_csv(raw / "a.csv", encoding="utf-8-sig").to_csv(dest_raw / "a.csv", index=False, encoding="utf-8-sig")

    settings = Settings(dashscope_api_key="")
    g = build_pipeline_graph(store, paths, settings)
    g.invoke({"session_id": sid})

    st2 = store.require(sid)
    assert st2.stage == "done"
    assert "report.html" in st2.artifacts
    assert Path(st2.artifacts["report.html"]).exists()


def test_main_app_import() -> None:
    from server.main import app as real_app

    assert real_app.title == "HouseInsight Agent"
