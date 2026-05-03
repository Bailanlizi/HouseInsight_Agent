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
