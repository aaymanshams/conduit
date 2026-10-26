"""Settings, read from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    seed_dir: Path
    data_dir: Path
    # Chat model: any OpenAI-compatible endpoint (OpenAI, Groq, Gemini, Ollama...).
    # With no API key, conduit runs fully offline with an extractive fallback.
    llm_base_url: str
    llm_api_key: str | None
    llm_model: str
    # Embeddings: "local" (hashing, no network) or an OpenAI-compatible model name.
    embed_model: str
    embed_base_url: str
    embed_api_key: str | None
    github_repo: str | None
    github_token: str | None
    slack_webhook_url: str | None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "conduit.db"

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ.get
        llm_base = env("CONDUIT_LLM_BASE_URL", "https://api.groq.com/openai/v1")
        values = dict(
            seed_dir=Path(env("CONDUIT_SEED_DIR", PROJECT_ROOT / "data" / "acme")),
            data_dir=Path(env("CONDUIT_DATA_DIR", PROJECT_ROOT / "var")),
            llm_base_url=llm_base,
            llm_api_key=env("CONDUIT_LLM_API_KEY") or None,
            llm_model=env("CONDUIT_LLM_MODEL", "llama-3.3-70b-versatile"),
            embed_model=env("CONDUIT_EMBED_MODEL", "local"),
            embed_base_url=env("CONDUIT_EMBED_BASE_URL", "https://api.openai.com/v1"),
            embed_api_key=env("CONDUIT_EMBED_API_KEY") or None,
            github_repo=env("CONDUIT_GITHUB_REPO") or None,
            github_token=env("CONDUIT_GITHUB_TOKEN") or None,
            slack_webhook_url=env("CONDUIT_SLACK_WEBHOOK_URL") or None,
        )
        values.update(overrides)
        return cls(**values)
