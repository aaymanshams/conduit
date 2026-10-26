"""Wires connectors, storage, search, RAG, tools and the agent into one object."""

from __future__ import annotations

import json
import logging
import shutil
import threading

from conduit.agent import Agent
from conduit.config import Settings
from conduit.connectors import CRMConnector, GitHubIssuesConnector, HelpdeskConnector, WikiConnector
from conduit.connectors.base import Connector
from conduit.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from conduit.llm import ChatModel, OfflineChat, OpenAICompatibleChat
from conduit.models import User
from conduit.notify import SlackNotifier
from conduit.rag import RagService
from conduit.search.hybrid import HybridIndex
from conduit.store import Store
from conduit.sync import SyncStats, sync_source
from conduit.tools import ToolRegistry

log = logging.getLogger(__name__)


def default_embedder(s: Settings) -> Embedder:
    if s.embed_model == "local":
        return HashingEmbedder()
    if not s.embed_api_key:
        raise ValueError("CONDUIT_EMBED_MODEL is set but CONDUIT_EMBED_API_KEY is missing")
    return OpenAICompatibleEmbedder(s.embed_base_url, s.embed_api_key, s.embed_model)


def default_llm(s: Settings) -> ChatModel:
    if s.llm_api_key:
        return OpenAICompatibleChat(s.llm_base_url, s.llm_api_key, s.llm_model)
    return OfflineChat()


class Conduit:
    def __init__(self, settings: Settings, llm: ChatModel | None = None, embedder: Embedder | None = None):
        self.settings = settings
        self._sync_lock = threading.Lock()
        sources = self._prepare_sources()

        self.users = self._load_users()
        self.roles = {
            uid: u["role"] for uid, u in json.loads((settings.seed_dir / "users.json").read_text()).items()
        }
        self.store = Store(settings.db_path)
        self.embedder = embedder or default_embedder(settings)
        if self.store.get_meta("embed_model") != self.embedder.model_id:
            # Vectors from different models are not comparable; rebuild from scratch.
            log.info("embedding model changed to %s; the next sync re-indexes everything", self.embedder.model_id)
            self.store.reset_index()
            self.store.set_meta("embed_model", self.embedder.model_id)

        self.crm = CRMConnector(sources / "crm_accounts.json")
        self.helpdesk = HelpdeskConnector(sources / "helpdesk_tickets.json")
        self.connectors: dict[str, Connector] = {
            "wiki": WikiConnector(sources / "wiki"),
            "crm": self.crm,
            "helpdesk": self.helpdesk,
        }
        if settings.github_repo:
            self.connectors["github"] = GitHubIssuesConnector(settings.github_repo, settings.github_token)

        self.index = HybridIndex(self.embedder)
        self.reload_index()
        self.llm = llm or default_llm(settings)
        self.rag = RagService(self.index, self.llm, self.store)
        self.slack = SlackNotifier(settings.slack_webhook_url, settings.data_dir / "slack_outbox.jsonl")
        self.tools = ToolRegistry(
            self.store, self.index, self.crm, self.helpdesk, self.slack,
            after_write=lambda source: self.sync(source),
        )
        self.agent = Agent(self.llm, self.tools, self.rag, self.store)

    def _prepare_sources(self):
        """Copy the seed data into the runtime data dir once, so approved writes never touch the repo."""
        target = self.settings.data_dir / "sources"
        if not target.exists():
            target.mkdir(parents=True)
            for name in ("crm_accounts.json", "helpdesk_tickets.json"):
                shutil.copy(self.settings.seed_dir / name, target / name)
            shutil.copytree(self.settings.seed_dir / "wiki", target / "wiki")
        return target

    def _load_users(self) -> dict[str, User]:
        raw = json.loads((self.settings.seed_dir / "users.json").read_text())
        return {uid: User(id=uid, name=u["name"], groups=frozenset(u["groups"])) for uid, u in raw.items()}

    def user(self, user_id: str) -> User:
        return self.users[user_id]

    def reload_index(self) -> None:
        chunks, vectors = self.store.load_chunks()
        self.index.rebuild(chunks, vectors)

    def sync(self, source: str | None = None) -> list[SyncStats]:
        names = [source] if source else list(self.connectors)
        unknown = [n for n in names if n not in self.connectors]
        if unknown:
            raise KeyError(f"unknown source(s): {', '.join(unknown)}")
        with self._sync_lock:
            stats = [sync_source(self.store, self.connectors[n], self.embedder) for n in names]
            self.reload_index()
        return stats
