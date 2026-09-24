"""Centralized runtime configuration.

Every notebook, script, and the FastAPI service can import `get_settings()`
instead of reading `os.environ` and hardcoding a base URL / model name
directly. Swapping providers (NVIDIA NIM, OpenAI, Anthropic) or models then
means editing `.env`, not the notebooks.

    from ai_engineering.config import get_settings, make_chat_client, NO_THINK

    settings = get_settings()
    client = make_chat_client(settings)
    client.chat.completions.create(model=settings.llm_model, extra_body=NO_THINK, ...)
"""
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, regardless of the caller's working directory — same convention
# the scripts use for resolving data/ paths from __file__.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# One base_url/model default per provider. `llm_provider` picks which of
# these `Settings.llm_base_url` / `Settings.llm_model` fall back to when
# that field isn't set explicitly in `.env`.
_PROVIDER_DEFAULTS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "embedding_model": "nvidia/nemotron-3-embed-1b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-5",
        "embedding_model": None,  # Anthropic has no embeddings endpoint
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: Literal["nvidia", "openai", "anthropic"] = "nvidia"

    # Leave unset to use the provider's default (see _PROVIDER_DEFAULTS below);
    # set explicitly to point at a different deployment/model without code changes.
    llm_base_url: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None

    nvidia_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @field_validator(
        "llm_base_url", "llm_model", "embedding_model",
        "nvidia_api_key", "openai_api_key", "anthropic_api_key",
        "langfuse_public_key", "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, v):
        # `.env.example` ships `NVIDIA_API_KEY=` etc.; a blank line copied from it
        # must mean "not configured" (offline fallback), not an empty key -> 401.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def _defaults(self) -> dict:
        return _PROVIDER_DEFAULTS[self.llm_provider]

    @property
    def resolved_base_url(self) -> str:
        return self.llm_base_url or self._defaults["base_url"]

    @property
    def resolved_model(self) -> str:
        return self.llm_model or self._defaults["model"]

    @property
    def resolved_embedding_model(self) -> str | None:
        return self.embedding_model or self._defaults["embedding_model"]

    @property
    def api_key(self) -> SecretStr | None:
        return {
            "nvidia": self.nvidia_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }[self.llm_provider]

    @property
    def has_llm_credentials(self) -> bool:
        """False when no key is configured for the active provider — callers
        should fall back to an offline/mock path instead of erroring."""
        return self.api_key is not None

    def require_api_key(self) -> str:
        if self.api_key is None:
            raise RuntimeError(
                f"{self.llm_provider.upper()}_API_KEY is not set. "
                f"Set it in .env, or set LLM_PROVIDER to a provider you do have a key for."
            )
        return self.api_key.get_secret_value()

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def get_settings() -> "Settings":
    """Re-reads .env and the environment on every call — deliberately not
    cached, since env vars can change within a process (tests monkeypatch
    them; a long-lived app might reload .env). Construction is cheap."""
    return Settings()


# Nemotron (and most reasoning-tuned NIM models) narrate their reasoning by
# default; every direct-answer call in this repo passes this to suppress it.
# Harmless no-op on providers that don't recognize the field.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def make_chat_client(settings: "Settings | None" = None, max_retries: int = 2):
    """OpenAI-compatible client (chat + embeddings) for the configured provider.

    `max_retries` is the SDK's own exponential backoff on 408/409/429/5xx and
    connection errors. Eval harnesses that make hundreds of calls should raise
    it — hosted NIM endpoints return transient 503 "overloaded" often enough
    that one unlucky call otherwise kills a whole run.

    Anthropic's API isn't OpenAI-compatible for tool calling / embeddings, so
    this raises for llm_provider="anthropic" — use langchain-anthropic or the
    Anthropic SDK directly for that provider instead.
    """
    from openai import OpenAI

    s = settings or get_settings()
    if s.llm_provider == "anthropic":
        raise ValueError(
            "make_chat_client() only supports OpenAI-compatible providers (nvidia, openai). "
            "Use the Anthropic SDK or langchain-anthropic directly for llm_provider='anthropic'."
        )
    return OpenAI(base_url=s.resolved_base_url, api_key=s.require_api_key(), max_retries=max_retries)
