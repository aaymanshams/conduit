"""HTTP integrations, tested against fake servers (httpx.MockTransport): GitHub, embeddings, chat, Slack."""

import json

import httpx
import numpy as np
import pytest

from conduit.connectors.github import GitHubError, GitHubIssuesConnector
from conduit.embeddings import OpenAICompatibleEmbedder
from conduit.llm import LLMError, OpenAICompatibleChat
from conduit.notify import NotifyError, SlackNotifier


def issue(n, updated="2026-10-20T10:00:00Z", pr=False):
    d = {"number": n, "title": f"Bug {n}", "body": "Steps to reproduce", "state": "open",
         "labels": [{"name": "bug"}], "updated_at": updated, "html_url": f"https://github.com/o/r/issues/{n}"}
    if pr:
        d["pull_request"] = {}
    return d


def github(handler):
    sleeps = []
    client = httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(handler))
    return GitHubIssuesConnector("o/r", token="t", client=client, sleep=sleeps.append), sleeps


def test_github_follows_pagination_skips_prs_and_passes_the_cursor():
    seen = []

    def handler(req):
        seen.append(req)
        if req.url.params.get("page") == "2":
            return httpx.Response(200, json=[issue(3)])
        return httpx.Response(200, json=[issue(1), issue(2, pr=True)],
                              headers={"Link": '<https://api.github.com/repos/o/r/issues?page=2>; rel="next"'})

    gh, _ = github(handler)
    docs = list(gh.fetch(since="2026-10-01T00:00:00Z"))
    assert [d.external_id for d in docs] == ["1", "3"]
    assert seen[0].url.params["since"] == "2026-10-01T00:00:00Z"
    assert seen[0].headers["Authorization"] == "Bearer t"
    assert docs[0].url == "https://github.com/o/r/issues/1"


def test_github_waits_on_rate_limit_then_succeeds():
    responses = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=[issue(1)])])
    gh, sleeps = github(lambda req: next(responses))
    assert len(list(gh.fetch(None))) == 1
    assert sleeps == [7.0]


def test_github_retries_server_errors_with_backoff():
    responses = iter([httpx.Response(502), httpx.Response(503), httpx.Response(200, json=[])])
    gh, sleeps = github(lambda req: next(responses))
    assert list(gh.fetch(None)) == []
    assert sleeps == [1.0, 2.0]


def test_github_does_not_retry_client_errors():
    gh, sleeps = github(lambda req: httpx.Response(404))
    with pytest.raises(GitHubError):
        list(gh.fetch(None))
    assert sleeps == []


def test_embeddings_are_reordered_by_index_and_normalised():
    def handler(req):
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 2.0]},
            {"index": 0, "embedding": [3.0, 4.0]},
        ]})

    emb = OpenAICompatibleEmbedder("https://x/v1", "k", "m", client=httpx.Client(base_url="https://x/v1", transport=httpx.MockTransport(handler)))
    out = emb.embed(["a", "b"])
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)


def chat_client(handler):
    return OpenAICompatibleChat("https://x/v1", "secret-key", "m",
                                client=httpx.Client(base_url="https://x/v1", transport=httpx.MockTransport(handler)))


def test_chat_parses_tool_calls_and_survives_bad_json_arguments():
    def handler(req):
        body = json.loads(req.content)
        assert body["tools"][0]["function"]["name"] == "f"
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "1", "function": {"name": "f", "arguments": '{"q": "x"}'}},
            {"id": "2", "function": {"name": "f", "arguments": "{not json"}},
        ]}}]})

    res = chat_client(handler).chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "f"}}])
    assert res.tool_calls[0].arguments == {"q": "x"}
    assert res.tool_calls[1].arguments is None


def test_chat_errors_never_leak_the_api_key():
    with pytest.raises(LLMError) as exc:
        chat_client(lambda req: httpx.Response(401, json={"error": "bad key"})).chat([{"role": "user", "content": "hi"}])
    assert "secret-key" not in str(exc.value)


def test_slack_webhook_success_and_failure(tmp_path):
    ok = SlackNotifier("https://hooks.slack.test/x", tmp_path / "o.jsonl",
                       client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    assert ok.post("#a", "hi")["delivered_to"] == "slack"
    bad = SlackNotifier("https://hooks.slack.test/x", tmp_path / "o.jsonl",
                        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(NotifyError):
        bad.post("#a", "hi")
