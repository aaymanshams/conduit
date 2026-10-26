"""MCP server: exposes Conduit's tools to any MCP client (Claude Desktop, Cursor, other agents).

The server acts as one configured user, and every call goes through the same
permission checks and approval queue as the REST API.
"""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

from conduit.service import Conduit


def build_server(conduit: Conduit, user_id: str) -> MCPServer:
    user = conduit.user(user_id)
    server = MCPServer(
        name="conduit",
        instructions=(
            f"Enterprise knowledge and actions for {user.name}. Write tools only queue a request "
            "for human approval; they never act immediately."
        ),
    )

    @server.tool(description="Hybrid search over wiki, CRM, helpdesk and GitHub sources the user can read.")
    def search_knowledge(query: str) -> dict:
        return conduit.tools.call(user, "search_knowledge", {"query": query})

    @server.tool(description="Answer a question from connected sources, with verified citations.")
    def ask(question: str) -> dict:
        return asdict(conduit.rag.ask(question, user))

    @server.tool(description="Fetch a CRM account by id (ACC-1001) or company name.")
    def get_account(account: str) -> dict:
        return conduit.tools.call(user, "get_account", {"account": account})

    @server.tool(description="List helpdesk tickets for an account. status: open, solved or all.")
    def list_tickets(account_id: str, status: str = "open") -> dict:
        return conduit.tools.call(user, "list_tickets", {"account_id": account_id, "status": status})

    @server.tool(description="Request a new helpdesk ticket. Queued for human approval.")
    def create_ticket(account_id: str, subject: str, description: str, priority: str) -> dict:
        return conduit.tools.call(
            user, "create_ticket",
            {"account_id": account_id, "subject": subject, "description": description, "priority": priority},
        )

    @server.tool(description="Request a Slack message. Queued for human approval.")
    def post_slack_message(channel: str, text: str) -> dict:
        return conduit.tools.call(user, "post_slack_message", {"channel": channel, "text": text})

    return server
