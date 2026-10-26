import asyncio

import pytest
from fastapi.testclient import TestClient

from conduit.api import create_app
from conduit.llm import LLMError
from conduit.mcp_server import build_server

TICKET = {
    "account_id": "ACC-1001",
    "subject": "Send SSO root-cause analysis to Northwind CIO",
    "description": "CIO asked for a written RCA before renewal talks.",
    "priority": "P2",
}


@pytest.fixture
def client(make_conduit):
    c = make_conduit()
    return TestClient(create_app(c)), c


def as_user(uid):
    return {"X-Conduit-User": uid}


def test_requests_need_a_known_user(client):
    http, _ = client
    assert http.post("/ask", json={"question": "hi"}).status_code == 401
    assert http.post("/ask", json={"question": "hi"}, headers=as_user("mallory")).status_code == 401


def test_health_reports_the_backend(client):
    http, _ = client
    body = http.get("/health").json()
    assert body["bm25_backend"] in {"cpp", "python"}
    assert body["chunks_indexed"] >= 12


def test_ask_returns_citations(client):
    http, _ = client
    body = http.post("/ask", json={"question": "P1 open for more than 8 hours"}, headers=as_user("arjun")).json()
    assert body["grounded"] and body["citations"][0]["doc_id"] == "wiki:support-sla"


def test_search_respects_permissions(client):
    http, _ = client
    q = {"q": "Compensation Bands salary lakh"}
    sales = http.get("/search", params=q, headers=as_user("priya")).json()
    hr = http.get("/search", params=q, headers=as_user("meera")).json()
    assert all("Compensation" not in r["title"] for r in sales)
    assert hr[0]["title"].startswith("Compensation")
    assert "score" not in hr[0]


def test_sync_and_audit_need_leadership(client):
    http, _ = client
    assert http.post("/sync", headers=as_user("priya")).status_code == 403
    assert http.post("/sync", headers=as_user("anita")).status_code == 200
    assert http.get("/audit", headers=as_user("arjun")).status_code == 403


def test_approval_flow_over_http(client):
    http, c = client
    queued = c.tools.call(c.user("priya"), "create_ticket", TICKET)
    aid = queued["action_id"]
    assert [a["id"] for a in http.get("/actions", headers=as_user("priya")).json()] == [aid]
    assert http.get("/actions", headers=as_user("arjun")).json() == []  # not theirs, not an approver
    assert http.post(f"/actions/{aid}/approve", headers=as_user("arjun")).status_code == 403
    assert http.post(f"/actions/{aid}/approve", headers=as_user("anita")).json()["status"] == "executed"
    assert http.post(f"/actions/{aid}/approve", headers=as_user("anita")).status_code == 409
    assert http.post("/actions/999/approve", headers=as_user("anita")).status_code == 404


def test_provider_outage_is_a_503(make_conduit):
    class DownLLM:
        model_id = "down"
        supports_tools = False

        def chat(self, messages, tools=None):
            raise LLMError("model provider returned 500")

    http = TestClient(create_app(make_conduit(llm=DownLLM())))
    resp = http.post("/ask", json={"question": "refund policy"}, headers=as_user("priya"))
    assert resp.status_code == 503


def test_ui_is_served(client):
    http, _ = client
    assert "<title>Conduit</title>" in http.get("/").text


def test_mcp_server_exposes_tools_with_the_same_permissions(make_conduit):
    c = make_conduit()
    server = build_server(c, "arjun")

    async def go():
        names = {t.name for t in await server.list_tools()}
        denied = await server.call_tool("get_account", {"account": "ACC-1001"})
        queued = await server.call_tool("create_ticket", TICKET)
        return names, denied, queued

    names, denied, queued = asyncio.run(go())
    assert {"search_knowledge", "ask", "get_account", "list_tickets", "create_ticket", "post_slack_message"} <= names
    assert "permission denied" in str(denied)
    assert "pending_approval" in str(queued)
    assert c.store.list_actions("pending")[0]["requested_by"] == "arjun"
