from fastapi.testclient import TestClient

from server.main import create_app


def test_health() -> None:
    client = TestClient(create_app())
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_session() -> None:
    client = TestClient(create_app())
    r = client.post("/api/v1/sessions")
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert len(body["session_id"]) >= 16


def test_post_run_json_sets_session_flags() -> None:
    client = TestClient(create_app())
    sid = client.post("/api/v1/sessions").json()["session_id"]
    store = client.app.state.store
    assert store.require(sid).return_cleaned_file is False
    r = client.post(
        f"/api/v1/sessions/{sid}/run",
        json={"return_cleaned_file": True, "skip_full_report_export": False},
    )
    assert r.status_code == 200
    st = store.require(sid)
    assert st.return_cleaned_file is True
    assert st.skip_full_report_export is False


def test_run_result_empty_session() -> None:
    client = TestClient(create_app())
    sid = client.post("/api/v1/sessions").json()["session_id"]
    r = client.get(f"/api/v1/sessions/{sid}/run_result")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["stage"] == "idle"
    assert body["progress_events"] == []
    assert "analysis_summary_plain" in body
    assert "figures_keys" not in body
    assert "figures_payload_chars" not in body
    assert "figures_too_large_for_inline" not in body
