"""Command-line entry point: `conduit <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from conduit.config import Settings
from conduit.service import Conduit


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="conduit", description="Permission-aware enterprise AI hub")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="pull changes from connected sources")
    s.add_argument("--source")
    sub.add_parser("status", help="indexed documents and last sync per source")

    for name, help_text in (("ask", "grounded answer with citations"), ("agent", "run the tool-calling agent"), ("search", "hybrid search")):
        a = sub.add_parser(name, help=help_text)
        a.add_argument("text")
        a.add_argument("--user", "-u", default="priya")

    a = sub.add_parser("actions", help="list actions waiting for approval")
    a.add_argument("--status", default="pending")
    for name in ("approve", "reject"):
        a = sub.add_parser(name, help=f"{name} a pending action")
        a.add_argument("action_id", type=int)
        a.add_argument("--user", "-u", default="anita")

    a = sub.add_parser("serve", help="run the REST API and web UI")
    a.add_argument("--port", type=int, default=8000)
    a = sub.add_parser("mcp", help="run the MCP server over stdio")
    a.add_argument("--user", "-u", default="priya")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        from conduit.api import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
        return 0

    svc = Conduit(Settings.from_env())
    if args.cmd == "sync":
        _print([asdict(st) for st in svc.sync(args.source)])
    elif args.cmd == "status":
        _print({"documents": svc.store.counts(), "sources": svc.store.sync_status()})
    elif args.cmd == "search":
        hits = svc.index.search(args.text, svc.user(args.user))
        _print([{"title": h.chunk.title, "text": h.chunk.text} for h in hits])
    elif args.cmd == "ask":
        _print(asdict(svc.rag.ask(args.text, svc.user(args.user))))
    elif args.cmd == "agent":
        _print(asdict(svc.agent.run(svc.user(args.user), args.text, role=svc.roles[args.user])))
    elif args.cmd == "actions":
        _print(svc.store.list_actions(args.status))
    elif args.cmd in ("approve", "reject"):
        approver = svc.user(args.user)
        if args.cmd == "approve":
            _print(svc.tools.approve(args.action_id, approver, svc.users))
        else:
            _print(svc.tools.reject(args.action_id, approver))
    elif args.cmd == "mcp":
        from conduit.mcp_server import build_server

        build_server(svc, args.user).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
