"""HTTP layer: health, triage, validation (422), rate limit, CORS, audit."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import MAX_TICKET_CHARS
from src.config import Settings


@pytest.fixture
def client(offline_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(offline_settings)) as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and "version" in data
    assert data["model"] == "claude-sonnet-4-5-20250929"
    assert data["llm_enabled"] is False
    assert data["classifier_backend"] == "keyword_heuristic"
    assert data["retrieval_backend"] == "memory" and data["persistence_backend"] == "memory"


def test_triage_happy_path(client: TestClient) -> None:
    r = client.post("/api/triage", json={"ticket_id": "t-1", "body": "Where is my order 123?"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "track_order"
    assert data["decision"]["decision"] in ("auto_resolve", "suggest", "escalate")
    assert len(data["trace_id"]) == 32
    assert data["llm_used"] is False and data["cost_usd"] == 0.0
    assert data["similar_resolved"] and data["draft_response"]


def test_triage_empty_body_is_422(client: TestClient) -> None:
    r = client.post("/api/triage", json={"ticket_id": "t-1", "body": ""})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "body"]


def test_triage_oversized_body_is_422(client: TestClient) -> None:
    r = client.post("/api/triage", json={"ticket_id": "t-1", "body": "x" * (MAX_TICKET_CHARS + 1)})
    assert r.status_code == 422


def test_triage_malformed_json_is_422(client: TestClient) -> None:
    r = client.post(
        "/api/triage", content=b'{"ticket_id": ', headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_triage_missing_ticket_id_is_422(client: TestClient) -> None:
    r = client.post("/api/triage", json={"body": "hello"})
    assert r.status_code == 422


def test_triage_invalid_channel_is_422(client: TestClient) -> None:
    r = client.post("/api/triage", json={"ticket_id": "t", "body": "hi", "channel": "fax"})
    assert r.status_code == 422


def test_audit_recent_lists_triaged_tickets(client: TestClient) -> None:
    client.post("/api/triage", json={"ticket_id": "a-1", "body": "Cancel my order please"})
    client.post("/api/triage", json={"ticket_id": "a-2", "body": "I want a refund"})
    rows = client.get("/api/audit/recent", params={"limit": 10}).json()
    assert [r["ticket_id"] for r in rows[:2]] == ["a-2", "a-1"]
    assert rows[0]["decision"] and rows[0]["trace_id"]
    assert client.get("/api/audit/recent", params={"limit": 0}).status_code == 422


def test_cors_allows_configured_origin_only(offline_settings: Settings) -> None:
    settings = offline_settings.model_copy(update={"cors_origins": "https://demo.example, *"})
    with TestClient(create_app(settings)) as c:
        ok = c.options(
            "/api/triage",
            headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"},
        )
        assert ok.headers.get("access-control-allow-origin") == "https://demo.example"
        denied = c.options(
            "/api/triage",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in denied.headers
    assert settings.cors_origin_list == ["https://demo.example"]


def test_rate_limit_returns_429(offline_settings: Settings) -> None:
    settings = offline_settings.model_copy(update={"rate_limit": "2/minute"})
    with TestClient(create_app(settings)) as c:
        codes = [
            c.post("/api/triage", json={"ticket_id": f"r-{i}", "body": "hello"}).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 429]


def test_eval_endpoint_returns_summary(client: TestClient) -> None:
    data = client.get("/api/eval/run").json()
    assert data["n"] >= 100 and "macro_f1" in data and "cases" not in data
