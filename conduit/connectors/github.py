"""GitHub Issues connector: a live REST API integration.

Handles the things every real SaaS integration has to: cursor-based
incremental sync (`since`), Link-header pagination, rate limits (429/403
with Retry-After or X-RateLimit-Reset) and transient 5xx errors with
exponential backoff.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import httpx

from conduit.connectors.base import Connector
from conduit.models import Document

GITHUB_GROUPS = ["everyone"]  # a public repo's issues are visible to all staff
MAX_RETRIES = 4
MAX_WAIT_SECONDS = 60.0


class GitHubError(RuntimeError):
    pass


class GitHubIssuesConnector(Connector):
    name = "github"

    def __init__(
        self,
        repo: str,
        token: str | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if repo.count("/") != 1:
            raise ValueError("repo must look like 'owner/name'")
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.repo = repo
        self.client = client or httpx.Client(base_url="https://api.github.com", timeout=20.0)
        self.client.headers.update(headers)
        self.sleep = sleep

    def fetch(self, since: str | None) -> Iterator[Document]:
        params = {"state": "all", "per_page": 100, "sort": "updated", "direction": "asc"}
        if since:
            params["since"] = since
        for issue in self._paginate(f"/repos/{self.repo}/issues", params):
            if "pull_request" in issue:  # the issues endpoint also returns PRs
                continue
            yield self._to_document(issue)

    def list_ids(self) -> set[str]:
        params = {"state": "all", "per_page": 100}
        return {
            str(i["number"])
            for i in self._paginate(f"/repos/{self.repo}/issues", params)
            if "pull_request" not in i
        }

    def _paginate(self, path: str, params: dict) -> Iterator[dict]:
        url: str | None = path
        while url:
            resp = self._get(url, params)
            yield from resp.json()
            url = resp.links.get("next", {}).get("url")
            params = None  # the next link already carries the query string

    def _get(self, url: str, params: dict | None) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.client.get(url, params=params)
            except httpx.TransportError as exc:
                if attempt == MAX_RETRIES:
                    raise GitHubError(f"network error talking to GitHub: {exc}") from exc
                self.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp
            if attempt < MAX_RETRIES and (wait := self._retry_after(resp, attempt)) is not None:
                self.sleep(wait)
                continue
            raise GitHubError(f"GitHub returned {resp.status_code} for {url}")
        raise AssertionError("unreachable")

    @staticmethod
    def _retry_after(resp: httpx.Response, attempt: int) -> float | None:
        """Seconds to wait before retrying, or None if the error is not retryable."""
        rate_limited = resp.status_code == 429 or (
            resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0"
        )
        if rate_limited:
            if "Retry-After" in resp.headers:
                return min(float(resp.headers["Retry-After"]), MAX_WAIT_SECONDS)
            if "X-RateLimit-Reset" in resp.headers:
                reset = float(resp.headers["X-RateLimit-Reset"]) - time.time()
                return min(max(reset, 1.0), MAX_WAIT_SECONDS)
            return min(2.0**attempt, MAX_WAIT_SECONDS)
        if resp.status_code >= 500:
            return min(2.0**attempt, MAX_WAIT_SECONDS)
        return None

    def _to_document(self, issue: dict) -> Document:
        labels = ", ".join(lbl["name"] for lbl in issue.get("labels", [])) or "none"
        text = (
            f"GitHub issue #{issue['number']} in {self.repo}. State: {issue['state']}. "
            f"Labels: {labels}.\n\n{issue.get('body') or ''}"
        )
        updated = datetime.fromisoformat(issue["updated_at"].replace("Z", "+00:00"))
        return Document(
            source="github",
            external_id=str(issue["number"]),
            title=f"Issue #{issue['number']}: {issue['title']}",
            text=text,
            allowed_groups=GITHUB_GROUPS,
            updated_at=updated.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            url=issue.get("html_url"),
            metadata={"state": issue["state"]},
        )
