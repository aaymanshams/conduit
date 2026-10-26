"""Tools the agent (and MCP clients) can call, with permission checks and an approval gate for writes.

Read tools run immediately, as the calling user. Write tools never run when
the model asks: they are queued as a pending action, and only execute after a
human in the `approver` group (who is not the requester) approves them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from conduit.connectors.crm import CRMConnector
from conduit.connectors.helpdesk import HelpdeskConnector
from conduit.models import User
from conduit.notify import SlackNotifier
from conduit.search.hybrid import HybridIndex
from conduit.store import Store

APPROVER_GROUP = "approver"


class ToolError(Exception):
    """An error the model is allowed to see and recover from."""


class PermissionDenied(Exception):
    pass


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="What to look for")


class AccountArgs(BaseModel):
    account: str = Field(min_length=1, max_length=200, description="Account id (ACC-1001) or company name")


class TicketsArgs(BaseModel):
    account_id: str = Field(pattern=r"^ACC-\d+$")
    status: Literal["open", "solved", "all"] = "open"


class CreateTicketArgs(BaseModel):
    account_id: str = Field(pattern=r"^ACC-\d+$")
    subject: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=10, max_length=4000)
    priority: Literal["P1", "P2", "P3", "P4"]


class SlackArgs(BaseModel):
    channel: str = Field(pattern=r"^#[a-z0-9_-]{1,80}$", description="Channel name such as #cs-escalations")
    text: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    kind: Literal["read", "write"]
    groups: tuple[str, ...]  # caller must be in at least one
    run: Callable[[User, BaseModel], dict]
    precheck: Callable[[BaseModel], None] | None = None  # cheap validation before queueing a write


class ToolRegistry:
    def __init__(
        self,
        store: Store,
        index: HybridIndex,
        crm: CRMConnector,
        helpdesk: HelpdeskConnector,
        slack: SlackNotifier,
        after_write: Callable[[str], None] = lambda source: None,
    ):
        self.store = store
        self.index = index
        self.crm = crm
        self.helpdesk = helpdesk
        self.slack = slack
        self.after_write = after_write  # e.g. re-sync the helpdesk so a new ticket becomes searchable
        tools = [
            Tool("search_knowledge", "Hybrid keyword + semantic search over every connected source the user can read "
                 "(wiki policies, CRM accounts, helpdesk tickets, GitHub issues).",
                 SearchArgs, "read", ("everyone",), self._search),
            Tool("get_account", "Fetch one CRM account: revenue, owner, renewal date, health and notes.",
                 AccountArgs, "read", ("sales", "leadership"), self._get_account),
            Tool("list_tickets", "List helpdesk tickets for an account.",
                 TicketsArgs, "read", ("support", "sales", "leadership"), self._list_tickets),
            Tool("create_ticket", "Create a helpdesk ticket. Needs human approval before it is created.",
                 CreateTicketArgs, "write", ("support", "sales", "leadership"), self._create_ticket,
                 precheck=self._check_account_exists),
            Tool("post_slack_message", "Post a message to a Slack channel. Needs human approval before it is sent.",
                 SlackArgs, "write", ("support", "sales", "leadership"), self._post_slack),
        ]
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        """OpenAI-style function schemas generated from the pydantic models."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_model.model_json_schema(),
                },
            }
            for t in self.tools.values()
        ]

    def call(self, user: User, name: str, raw_args: dict | None) -> dict:
        """Run a read tool or queue a write tool. Returns a JSON-able dict, including errors for the model."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'"}
        if raw_args is None:
            return {"error": "arguments were not valid JSON"}
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return {"error": "invalid arguments", "details": exc.errors(include_url=False, include_context=False)}
        if not user.in_any(*tool.groups):
            self.store.audit(user.id, "tool.denied", {"tool": name})
            return {"error": f"permission denied: {name} needs one of {list(tool.groups)}"}

        try:
            if tool.kind == "read":
                result = tool.run(user, args)
                self.store.audit(user.id, "tool.read", {"tool": name, "args": args.model_dump()})
                return result
            if tool.precheck:
                tool.precheck(args)
        except ToolError as exc:
            return {"error": str(exc)}

        action_id = self.store.create_action(user.id, name, args.model_dump())
        self.store.audit(user.id, "action.requested", {"action_id": action_id, "tool": name, "args": args.model_dump()})
        return {
            "status": "pending_approval",
            "action_id": action_id,
            "message": "Queued for human approval. It has NOT been executed yet.",
        }

    # --- approval queue ---------------------------------------------------

    def approve(self, action_id: int, approver: User, users: dict[str, User]) -> dict:
        action = self._decidable(action_id, approver)
        if not self.store.decide_action(action_id, approver.id, "approved"):
            raise ToolError(f"action {action_id} is no longer pending")
        tool = self.tools[action["tool"]]
        requester = users.get(action["requested_by"])
        try:
            # Re-check at execution time: the requester's access may have changed while it waited.
            if requester is None or not requester.in_any(*tool.groups):
                raise PermissionDenied("requester no longer has permission for this tool")
            args = tool.args_model.model_validate(action["args"])
            result = tool.run(requester, args)
        except Exception as exc:  # noqa: BLE001 - record any failure on the action
            self.store.finish_action(action_id, "failed", {"error": str(exc)})
            self.store.audit(approver.id, "action.failed", {"action_id": action_id, "error": str(exc)})
            return self.store.get_action(action_id)
        self.store.finish_action(action_id, "executed", result)
        self.store.audit(approver.id, "action.executed", {"action_id": action_id, "tool": tool.name})
        return self.store.get_action(action_id)

    def reject(self, action_id: int, approver: User) -> dict:
        self._decidable(action_id, approver)
        if not self.store.decide_action(action_id, approver.id, "rejected"):
            raise ToolError(f"action {action_id} is no longer pending")
        self.store.audit(approver.id, "action.rejected", {"action_id": action_id})
        return self.store.get_action(action_id)

    def _decidable(self, action_id: int, approver: User) -> dict:
        action = self.store.get_action(action_id)
        if action is None:
            raise KeyError(action_id)
        if APPROVER_GROUP not in approver.groups:
            raise PermissionDenied("only members of the approver group can decide actions")
        if action["requested_by"] == approver.id:
            raise PermissionDenied("you cannot approve or reject your own request")
        return action

    # --- tool implementations ---------------------------------------------

    def _search(self, user: User, args: SearchArgs) -> dict:
        hits = self.index.search(args.query, user, k=5)
        return {
            "results": [
                {"doc_id": h.chunk.doc_id, "title": h.chunk.title, "source": h.chunk.source, "text": h.chunk.text}
                for h in hits
            ]
        }

    def _get_account(self, user: User, args: AccountArgs) -> dict:
        acc = self.crm.find_account(args.account)
        if acc is None:
            raise ToolError(f"no account matches '{args.account}'")
        return {"account": acc}

    def _list_tickets(self, user: User, args: TicketsArgs) -> dict:
        status = None if args.status == "all" else args.status
        return {"tickets": self.helpdesk.list_tickets(args.account_id, status)}

    def _check_account_exists(self, args: CreateTicketArgs) -> None:
        if self.crm.find_account(args.account_id) is None:
            raise ToolError(f"account {args.account_id} does not exist")

    def _create_ticket(self, user: User, args: CreateTicketArgs) -> dict:
        self._check_account_exists(args)
        ticket = self.helpdesk.create_ticket(args.account_id, args.subject, args.description, args.priority)
        self.after_write("helpdesk")
        return {"ticket": ticket}

    def _post_slack(self, user: User, args: SlackArgs) -> dict:
        return self.slack.post(args.channel, f"{args.text}\n— sent via Conduit for {user.name}")
