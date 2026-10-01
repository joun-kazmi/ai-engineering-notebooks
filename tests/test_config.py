"""Offline tests for src/ai_engineering/config.py."""
import pytest
from pydantic import ValidationError

from ai_engineering.config import NO_THINK, RateLimiter, Settings


def test_blank_key_counts_as_unset(monkeypatch):
    # .env.example ships `NVIDIA_API_KEY=`; copying it must mean OFFLINE, not a 401
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "  ")
    s = Settings(_env_file=None)
    assert s.nvidia_api_key is None
    assert not s.has_llm_credentials
    assert s.resolved_model == "nvidia/nemotron-3-super-120b-a12b"


def test_real_key_and_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    s = Settings(_env_file=None)
    assert s.has_llm_credentials
    assert s.require_api_key() == "sk-test"
    assert s.resolved_model == "gpt-test"
    assert s.resolved_base_url == "https://api.openai.com/v1"


def test_provider_specific_fields_only_go_to_nvidia(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    nv = Settings(_env_file=None).adapter
    assert nv.chat_extra_body() == NO_THINK
    assert nv.embedding_extra_body("query") == {"input_type": "query"}

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    oa = Settings(_env_file=None).adapter
    assert oa.chat_extra_body() is None
    assert oa.embedding_extra_body("query") is None


def test_unsupported_provider_is_rejected_at_config_time(monkeypatch):
    # Anthropic isn't OpenAI-compatible for tools/embeddings; selecting it used to
    # pass config, flip the evals to LIVE, then fail inside make_chat_client().
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rate_limiter_spaces_requests_and_records_wait(monkeypatch):
    clock = {"t": 100.0}
    sleeps = []
    monkeypatch.setattr("ai_engineering.config.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("ai_engineering.config.time.sleep", lambda s: sleeps.append(s))
    lim = RateLimiter(rpm=60)  # one request per second
    for _ in range(3):
        lim.wait()
    assert sleeps == [1.0, 2.0]  # back-to-back calls queue up one interval apart
    assert lim.waited == 3.0


def test_blank_auth_settings_mean_the_defaults(monkeypatch):
    for name in ("AUTH_MODE", "AUTH_ISSUER", "AUTH_AUDIENCE", "AUTH_JWKS_URL", "AUTH_ALGORITHMS",
                 "AUTH_SCOPE_CLAIM", "AUTH_GROUPS_CLAIM", "AUTH_APPROVER_GROUP"):
        monkeypatch.setenv(name, " ")
    s = Settings(_env_file=None)
    assert s.auth_mode == "oidc"  # fails closed unless issuer, audience and JWKS URL are set
    assert (s.auth_issuer, s.auth_audience, s.auth_jwks_url) == (None, None, None)
    assert s.resolved_auth_algorithms == ("RS256", "ES256")
    assert (s.auth_scope_claim, s.auth_groups_claim, s.auth_approver_group) == ("scope", "groups", "sre")

    monkeypatch.setenv("AUTH_ALGORITHMS", "ES256, RS256")
    assert Settings(_env_file=None).resolved_auth_algorithms == ("ES256", "RS256")
    monkeypatch.setenv("AUTH_MODE", "off")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
