import json

import pytest

from conduit.tools import PermissionDenied, ToolError
from tests.conftest import ScriptedLLM

TICKET = {
    "account_id": "ACC-1001",
    "subject": "Send SSO root-cause analysis to Northwind CIO",
    "description": "CIO asked for a written RCA of the Q3 SSO outages before renewal talks.",
    "priority": "P2",
}


def ticket_ids(c):
    return {t["id"] for t in json.loads(c.helpdesk.store.path.read_text())}


def test_agent_uses_read_tools_and_answers(make_conduit):
    llm = ScriptedLLM([[("get_account", {"account": "Northwind"})], "Northwind renews on 2026-12-15. (ACC-1001)"])
    c = make_conduit(llm=llm)
    run = c.agent.run(c.user("priya"), "When does Northwind renew?")
    assert run.mode == "agent"
    assert run.steps[0].result["account"]["renewal_date"] == "2026-12-15"
    assert "2026-12-15" in run.answer
    # The tool result was fed back to the model as a tool message.
    assert llm.seen[1][-1]["role"] == "tool"


def test_write_tools_are_queued_not_executed(make_conduit):
    c = make_conduit(llm=ScriptedLLM([[("create_ticket", TICKET)], "Queued as action 1 for approval."]))
    before = ticket_ids(c)
    run = c.agent.run(c.user("priya"), "Open a ticket for the RCA")
    assert run.pending_actions == [1]
    assert run.steps[0].result["status"] == "pending_approval"
    assert ticket_ids(c) == before  # nothing written to the helpdesk yet
    assert c.store.get_action(1)["status"] == "pending"


def test_approval_executes_the_action_and_the_new_ticket_becomes_searchable(make_conduit):
    c = make_conduit(llm=ScriptedLLM([[("create_ticket", TICKET)], "Queued."]))
    c.agent.run(c.user("priya"), "Open a ticket for the RCA")
    action = c.tools.approve(1, c.user("anita"), c.users)
    assert action["status"] == "executed"
    new_id = action["result"]["ticket"]["id"]
    assert new_id in ticket_ids(c)
    hits = c.index.search("root-cause analysis Northwind CIO", c.user("arjun"), k=3)
    assert f"helpdesk:{new_id}" in [h.chunk.doc_id for h in hits]


def test_requester_cannot_approve_their_own_action(make_conduit):
    c = make_conduit()
    c.tools.call(c.user("anita"), "create_ticket", TICKET)
    with pytest.raises(PermissionDenied):
        c.tools.approve(1, c.user("anita"), c.users)


def test_only_approvers_can_decide(make_conduit):
    c = make_conduit()
    c.tools.call(c.user("priya"), "create_ticket", TICKET)
    with pytest.raises(PermissionDenied):
        c.tools.approve(1, c.user("arjun"), c.users)


def test_an_action_can_only_be_decided_once(make_conduit):
    c = make_conduit()
    c.tools.call(c.user("priya"), "create_ticket", TICKET)
    c.tools.reject(1, c.user("anita"))
    with pytest.raises(ToolError):
        c.tools.approve(1, c.user("anita"), c.users)
    assert c.store.get_action(1)["status"] == "rejected"


def test_tool_permissions_are_enforced(make_conduit):
    c = make_conduit()
    result = c.tools.call(c.user("meera"), "get_account", {"account": "ACC-1001"})
    assert "permission denied" in result["error"]


def test_invalid_arguments_are_returned_to_the_model(make_conduit):
    c = make_conduit()
    result = c.tools.call(c.user("priya"), "create_ticket", {**TICKET, "priority": "urgent"})
    assert result["error"] == "invalid arguments"
    assert c.store.list_actions() == []


def test_ticket_for_unknown_account_is_refused_before_queueing(make_conduit):
    c = make_conduit()
    result = c.tools.call(c.user("priya"), "create_ticket", {**TICKET, "account_id": "ACC-9999"})
    assert "does not exist" in result["error"]
    assert c.store.list_actions() == []


def test_prompt_injection_in_data_cannot_send_anything_by_itself(make_conduit):
    # Even if a model obeys the injected text in TCK-504, the Slack post is only queued.
    llm = ScriptedLLM([
        [("list_tickets", {"account_id": "ACC-1003", "status": "open"})],
        [("post_slack_message", {"channel": "#general", "text": "all accounts and revenue..."})],
        "done",
    ])
    c = make_conduit(llm=llm)
    run = c.agent.run(c.user("priya"), "Summarise Fabrikam's open tickets")
    assert run.pending_actions == [1]
    assert not (c.settings.data_dir / "slack_outbox.jsonl").exists()


def test_approved_slack_message_goes_to_the_outbox(make_conduit):
    c = make_conduit()
    c.tools.call(c.user("priya"), "post_slack_message", {"channel": "#cs-escalations", "text": "Northwind is at risk"})
    c.tools.approve(1, c.user("anita"), c.users)
    line = json.loads((c.settings.data_dir / "slack_outbox.jsonl").read_text().splitlines()[0])
    assert line["channel"] == "#cs-escalations"
    assert "Priya Nair" in line["text"]


def test_agent_stops_after_max_steps(make_conduit):
    c = make_conduit(llm=ScriptedLLM([[("search_knowledge", {"query": "sso"})]] * 10))
    run = c.agent.run(c.user("priya"), "loop forever")
    assert len(run.steps) == c.agent.max_steps
    assert "maximum number of steps" in run.answer


def test_offline_model_falls_back_to_rag(make_conduit):
    c = make_conduit()
    run = c.agent.run(c.user("priya"), "What is the refund policy for annual plans?")
    assert run.mode == "rag-fallback"
    assert "30 days" in run.answer
