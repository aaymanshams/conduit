"""REST API (FastAPI).

Authentication is a demo stand-in: the caller names a user in the
`X-Conduit-User` header. In production this would be the company's SSO
(OIDC/SAML) and the groups would come from the identity provider.
"""

# No `from __future__ import annotations` here: FastAPI must evaluate the
# Annotated[..., Depends(...)] hints on the nested route functions.
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from conduit.config import Settings
from conduit.embeddings import EmbeddingError
from conduit.llm import LLMError
from conduit.models import User
from conduit.search.bm25 import BACKEND
from conduit.service import Conduit
from conduit.tools import PermissionDenied, ToolError

STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AgentBody(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def create_app(conduit: Conduit | None = None) -> FastAPI:
    app = FastAPI(title="Conduit", version="0.1.0")
    svc = conduit or Conduit(Settings.from_env())

    def current_user(x_conduit_user: Annotated[str | None, Header()] = None) -> User:
        if not x_conduit_user or x_conduit_user not in svc.users:
            raise HTTPException(401, "send a known user id in the X-Conduit-User header")
        return svc.user(x_conduit_user)

    def require(group: str):
        def check(user: Annotated[User, Depends(current_user)]) -> User:
            if group not in user.groups:
                raise HTTPException(403, f"requires the '{group}' group")
            return user
        return check

    @app.exception_handler(LLMError)
    @app.exception_handler(EmbeddingError)
    async def provider_down(_, exc):
        return JSONResponse(status_code=503, content={"detail": f"AI provider unavailable: {exc}"})

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "chunks_indexed": len(svc.index),
            "bm25_backend": BACKEND,
            "llm": svc.llm.model_id,
            "embedder": svc.embedder.model_id,
            "sources": list(svc.connectors),
        }

    @app.get("/users")
    def users():
        return [{"id": u.id, "name": u.name, "role": svc.roles[u.id], "groups": sorted(u.groups)} for u in svc.users.values()]

    @app.post("/sync")
    def sync(user: Annotated[User, Depends(require("leadership"))], source: str | None = None):
        try:
            stats = svc.sync(source)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        svc.store.audit(user.id, "sync", {"source": source or "all"})
        return [asdict(s) for s in stats]

    @app.get("/sync/status")
    def sync_status(user: Annotated[User, Depends(current_user)]):
        return {"documents": svc.store.counts(), "sources": svc.store.sync_status()}

    @app.get("/search")
    def search(user: Annotated[User, Depends(current_user)], q: str = Query(min_length=1, max_length=500), k: int = Query(5, ge=1, le=20)):
        hits = svc.index.search(q, user, k=k)
        svc.store.audit(user.id, "search", {"q": q, "results": [h.chunk.doc_id for h in hits]})
        # Raw BM25/cosine scores are not returned: they are computed with corpus-wide statistics.
        return [
            {
                "rank": i + 1,
                "doc_id": h.chunk.doc_id,
                "title": h.chunk.title,
                "source": h.chunk.source,
                "text": h.chunk.text,
                "matched_by": [m for m, r in (("keyword", h.keyword_rank), ("vector", h.vector_rank)) if r is not None],
            }
            for i, h in enumerate(hits)
        ]

    @app.post("/ask")
    def ask(body: AskBody, user: Annotated[User, Depends(current_user)]):
        return asdict(svc.rag.ask(body.question, user))

    @app.post("/agent")
    def agent(body: AgentBody, user: Annotated[User, Depends(current_user)]):
        return asdict(svc.agent.run(user, body.message, role=svc.roles[user.id]))

    @app.get("/actions")
    def actions(user: Annotated[User, Depends(current_user)], status: str | None = None):
        rows = svc.store.list_actions(status)
        if "approver" not in user.groups:
            rows = [a for a in rows if a["requested_by"] == user.id]
        return rows

    def decide(action_id: int, user: User, approve: bool):
        try:
            if approve:
                return svc.tools.approve(action_id, user, svc.users)
            return svc.tools.reject(action_id, user)
        except KeyError as exc:
            raise HTTPException(404, f"no action {action_id}") from exc
        except PermissionDenied as exc:
            raise HTTPException(403, str(exc)) from exc
        except ToolError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/actions/{action_id}/approve")
    def approve(action_id: int, user: Annotated[User, Depends(current_user)]):
        return decide(action_id, user, approve=True)

    @app.post("/actions/{action_id}/reject")
    def reject(action_id: int, user: Annotated[User, Depends(current_user)]):
        return decide(action_id, user, approve=False)

    @app.get("/audit")
    def audit(user: Annotated[User, Depends(require("leadership"))], limit: int = Query(100, ge=1, le=1000)):
        return svc.store.audit_entries(limit)

    return app
