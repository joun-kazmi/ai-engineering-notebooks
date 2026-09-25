"""Centralized runtime configuration.

Every notebook, script, and the FastAPI service can import `get_settings()`
instead of reading `os.environ` and hardcoding a base URL / model name
directly. Swapping between OpenAI-compatible providers (NVIDIA NIM, OpenAI)
or models then means editing `.env`, not the notebooks.

Provider differences beyond base URL and model name live in `ProviderAdapter`
(`Settings.adapter`), not at call sites:

    from ai_engineering.config import get_settings, make_chat_client

    settings = get_settings()
    client = make_chat_client(settings)
    client.chat.completions.create(model=settings.resolved_model,
                                   extra_body=settings.adapter.chat_extra_body(), ...)
"""
import threading
import time
from dataclasses import dataclass
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
}

# Nemotron (and most reasoning-tuned NIM models) narrate their reasoning by
# default. This is a NIM chat-template field, not part of OpenAI's schema —
# send it via `ProviderAdapter.chat_extra_body()`, which only does so for nvidia.
# Kept as a constant for the older NVIDIA-only notebooks that import it.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class ProviderAdapter:
    """Request fields that differ between OpenAI-compatible providers.

    NIM accepts extras OpenAI doesn't (and OpenAI rejects unknown request
    arguments rather than ignoring them), so these must never be sent
    unconditionally.
    """
    provider: str

    def chat_extra_body(self) -> dict | None:
        """Extra chat-completions fields: NIM's thinking switch, else nothing."""
        return NO_THINK if self.provider == "nvidia" else None

    def embedding_extra_body(self, input_type: str) -> dict | None:
        """NIM's asymmetric embedders need `input_type` ("query" vs "passage");
        OpenAI's embeddings are symmetric and take no such parameter."""
        return {"input_type": input_type} if self.provider == "nvidia" else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Only OpenAI-compatible providers: everything here goes through the
    # OpenAI SDK (chat, tool calling, embeddings, json_schema output).
    llm_provider: Literal["nvidia", "openai"] = "nvidia"

    # Leave unset to use the provider's default (see _PROVIDER_DEFAULTS above);
    # set explicitly to point at a different deployment/model without code changes.
    llm_base_url: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None

    nvidia_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Client-side request pacing, shared by every client make_chat_client()
    # builds in this process. Unset = no pacing. NIM's free tier allows about
    # 40 requests/minute; eval runs that exceed it get 429s the SDK's short
    # backoff can't ride out.
    llm_max_rpm: int | None = None

    # Optional USD prices per million prompt / completion tokens, for run cost
    # budgets (ai_engineering.tool_runtime.RunBudget). Unset = cost is reported
    # as "unpriced" and not capped; NIM's free tier has no price to put here.
    llm_usd_per_mtok_in: float | None = None
    llm_usd_per_mtok_out: float | None = None

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @field_validator(
        "llm_base_url", "llm_model", "embedding_model", "llm_max_rpm",
        "llm_usd_per_mtok_in", "llm_usd_per_mtok_out",
        "nvidia_api_key", "openai_api_key",
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
    def adapter(self) -> ProviderAdapter:
        return ProviderAdapter(self.llm_provider)

    @property
    def resolved_base_url(self) -> str:
        return self.llm_base_url or self._defaults["base_url"]

    @property
    def resolved_model(self) -> str:
        return self.llm_model or self._defaults["model"]

    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model or self._defaults["embedding_model"]

    @property
    def api_key(self) -> SecretStr | None:
        return {"nvidia": self.nvidia_api_key, "openai": self.openai_api_key}[self.llm_provider]

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


class RateLimiter:
    """Spaces requests at least 60/rpm seconds apart. Thread-safe. Keeps a
    running total of time spent waiting so latency measurements can exclude
    self-imposed pacing (see `pacing_wait_seconds`)."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.waited = 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self, *_):  # usable directly as an httpx request event hook
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
            if delay > 0:
                self.waited += delay
        if delay > 0:
            time.sleep(delay)


_limiters: dict[int, RateLimiter] = {}


def pacing_wait_seconds() -> float:
    """Total time this process has spent in client-side pacing. Take a delta
    around a timed block and subtract it to report provider latency only."""
    return sum(lim.waited for lim in _limiters.values())


def paced_http_client(rpm: int | None):
    """An httpx client whose every request — SDK retries included — passes
    through one process-wide RateLimiter per rpm value, so separate OpenAI
    clients (chat, embeddings, a judge) share a single budget. None = no pacing."""
    if not rpm:
        return None
    import httpx

    limiter = _limiters.setdefault(rpm, RateLimiter(rpm))
    return httpx.Client(event_hooks={"request": [limiter.wait]}, timeout=httpx.Timeout(120.0, connect=10.0))


def make_chat_client(settings: "Settings | None" = None, max_retries: int = 2, max_rpm: int | None = None,
                     client_cls=None):
    """OpenAI-compatible client (chat + embeddings) for the configured provider.

    `max_retries` is the SDK's own exponential backoff on 408/409/429/5xx and
    connection errors. Eval harnesses that make hundreds of calls should raise
    it — hosted NIM endpoints return transient 503 "overloaded" often enough
    that one unlucky call otherwise kills a whole run. `max_rpm` (default:
    LLM_MAX_RPM) paces requests client-side so a long run doesn't hit 429s in
    the first place. `client_cls` swaps in a drop-in wrapper such as
    `langfuse.openai.OpenAI`.
    """
    if client_cls is None:
        from openai import OpenAI as client_cls

    s = settings or get_settings()
    kwargs = {}
    http_client = paced_http_client(max_rpm or s.llm_max_rpm)
    if http_client is not None:
        kwargs["http_client"] = http_client
    return client_cls(base_url=s.resolved_base_url, api_key=s.require_api_key(), max_retries=max_retries, **kwargs)
